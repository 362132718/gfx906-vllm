#!/usr/bin/env bash
# p1_backend_ab.sh -- P1 driver: TRITON_ATTN vs ROCM_ATTN on the delivery
# geometry, prefill AND decode, at 2K/8K/16K context.
#
# Runs on the GPU host. Builds a FRESH container from vllm-gfx906:latest (nothing
# hand-installed, so the result describes the delivered image), then runs one
# process per arm -- the engine's attention backend cannot be changed at runtime.
#
# Clock: AUTO, deliberately. The old bucket table was pin=1425MHz (DPM level 4)
# and is not a valid denominator for the delivery path. `power_dpm_force_performance_level`
# must read `auto`; this driver records what it actually saw.
#
# The host has a history of silent power loss (init-time thermtrip at DPM level
# 8), so everything is detached and every step lands in $DIR/driver.log.
#
# Usage:  bash p1_backend_ab.sh [arm] [arm] ...     (default: TRITON_ATTN ROCM_ATTN)
set -uo pipefail

DIR=/home/x/vllm-work/p1
CTR=p1-ab
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
MODEL_IN=/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
ARMS=${*:-"TRITON_ATTN ROCM_ATTN"}
REPS=${REPS:-2}
CTXS=${CTXS:-2048,8192,16256}
mkdir -p "$DIR"
LOG="$DIR/driver.log"

{
  echo "=== P1 BACKEND A/B $(date -Iseconds) ==="
  echo "arms: $ARMS | reps=$REPS | ctxs=$CTXS | image=$IMAGE"
  echo "clock: force_level=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null)"
} | tee -a "$LOG"

# --- refuse to stack on a live engine (zombies do not hold the GPUs) ---------
BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore" | grep -v grep \
       | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "P1_ABORT busy=$BUSY vllm processes still alive; not stacking a run" | tee -a "$LOG"
  ps -eo pid,stat,args | grep -E "vllm serve|EngineCore" | grep -v grep | head -5
  exit 1
fi

docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v "$MODELS_HOST":/models:ro \
  -v "$DIR":/work \
  "$IMAGE" sleep infinity >/dev/null || {
    echo "P1_ABORT docker run failed" | tee -a "$LOG"; exit 1; }

# --- host-side clock/temp sampler (hwmon; rocm-smi's sclk is unreliable) -----
HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ 2>/dev/null | head -1)
CLK="$DIR/clocks.txt"; : > "$CLK"
(
  while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

for ARM in $ARMS; do
  O="$DIR/out-$ARM"; rm -rf "$O"; mkdir -p "$O"
  echo "=== arm $ARM start $(date -Iseconds) ===" | tee -a "$LOG"
  docker exec -d "$CTR" bash -lc \
    "ATTN_BACKEND=$ARM OUT=/work/out-$ARM MODEL=$MODEL_IN REPS=$REPS CTXS=$CTXS \
     python3 -u /work/p1_backend_ab.py > /work/out-$ARM/probe.log 2>&1"

  for i in $(seq 1 300); do           # up to 75 min per arm
    sleep 15
    if grep -qE "P1_ARM_DONE|Traceback|SystemExit|FATAL|Error:" "$O/probe.log" 2>/dev/null; then
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
      echo "!! container died during arm $ARM" | tee -a "$LOG"; break
    fi
  done
  echo "=== arm $ARM done $(date -Iseconds) (rc-marker: $(grep -c P1_ARM_DONE "$O/probe.log" 2>/dev/null)) ===" | tee -a "$LOG"
  tail -25 "$O/probe.log" 2>/dev/null | tee -a "$LOG"
  cp "$O/probe.log" "$DIR/probe-$ARM.log" 2>/dev/null
done

kill $SAMPLER 2>/dev/null; trap - EXIT
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" \
  | tee -a "$LOG"
echo "P1_ALL_DONE $(date -Iseconds)" | tee -a "$LOG"
