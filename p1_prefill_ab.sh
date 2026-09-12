#!/usr/bin/env bash
# p1_prefill_ab.sh -- corrected P1 prefill A/B (Prefix-Cache trap removed).
#
# Runs on the GPU host. Fresh container from vllm-gfx906:latest. One process per
# arm. Uses p1_prefill_ab.py (enable_prefix_caching=False + unique nonce per
# request), NOT p1_backend_ab.py -- that one measured a cached prefill 6x..30x
# too fast on 2026-09-11.
#
# Usage: bash p1_prefill_ab.sh            # TRITON_ATTN then ROCM_ATTN
#        bash p1_prefill_ab.sh ROCM_ATTN  # single arm
set -uo pipefail

DIR=/home/x/vllm-work/p1pf
CTR=p1-pf
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
MODEL_IN=/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
ARMS=${*:-"TRITON_ATTN ROCM_ATTN"}
REPS=${REPS:-3}
CTXS=${CTXS:-2048,8192,16256}
mkdir -p "$DIR"
LOG="$DIR/driver.log"

{ echo "=== P1 PREFILL A/B (corrected) $(date -Iseconds) ==="
  echo "arms: $ARMS | reps=$REPS | ctxs=$CTXS"
  echo "clock: force_level=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null)"
} | tee -a "$LOG"

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore" | grep -v grep \
       | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "P1PF_ABORT busy=$BUSY vllm processes alive; not stacking" | tee -a "$LOG"
  ps -eo pid,stat,args | grep -E "vllm serve|EngineCore" | grep -v grep | head -5
  exit 1
fi

docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v "$MODELS_HOST":/models:ro -v "$DIR":/work \
  "$IMAGE" sleep infinity >/dev/null || {
    echo "P1PF_ABORT docker run failed" | tee -a "$LOG"; exit 1; }

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ 2>/dev/null | head -1)
CLK="$DIR/clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

for ARM in $ARMS; do
  O="$DIR/out-$ARM"; rm -rf "$O"; mkdir -p "$O"
  echo "=== arm $ARM start $(date -Iseconds) ===" | tee -a "$LOG"
  docker exec -d "$CTR" bash -lc \
    "ATTN_BACKEND=$ARM OUT=/work/out-$ARM MODEL=$MODEL_IN REPS=$REPS CTXS=$CTXS \
     python3 -u /work/p1_prefill_ab.py > /work/out-$ARM/probe.log 2>&1"
  for i in $(seq 1 300); do
    sleep 15
    if grep -qE "P1PF_ARM_DONE|Traceback|SystemExit|FATAL" "$O/probe.log" 2>/dev/null; then break; fi
    docker ps --format '{{.Names}}' | grep -qx "$CTR" || {
      echo "!! container died during $ARM" | tee -a "$LOG"; break; }
  done
  echo "=== arm $ARM done $(date -Iseconds) ===" | tee -a "$LOG"
  grep -E "^  ctx|^    rep|engine ready|P1PF_ARM_DONE|Error|Traceback" "$O/probe.log" 2>/dev/null | tail -20 | tee -a "$LOG"
  # did it reuse the other arm's AOT cache? (cache hash may omit the backend)
  echo "   recompiled: $(grep -c 'Compiling a graph for compile range' "$O/probe.log" 2>/dev/null) graph(s)" | tee -a "$LOG"
done

kill $SAMPLER 2>/dev/null; trap - EXIT
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" | tee -a "$LOG"
echo "P1PF_ALL_DONE $(date -Iseconds)" | tee -a "$LOG"
