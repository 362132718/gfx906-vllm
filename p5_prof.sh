#!/usr/bin/env bash
# p5_prof.sh -- run the MoE decode profiler on the GPU host, one arm per spec.
#
# Usage:  bash p5_prof.sh "eagerprof:EAGER=1;MAX_LEN=16384;MAX_NUM_SEQS=48"
#
# Arms (separator ';'): EAGER, MAX_LEN, MAX_NUM_SEQS, ATTN_BACKEND,
#                       DELAY_ITERS, ACTIVE_ITERS, CTX.
#
# Profile EAGER on purpose: graphs only remove CPU launch gaps, kernel times are
# unchanged, and eager keeps every kernel individually visible to the profiler.
# Comparing the eager kernel-sum against the graph-mode wall number tells us
# whether the step is GPU-work-bound or launch-bound.
#
# Expects the container to already exist (run p5_moe.sh once first, or set CTR).
set -uo pipefail

DIR=/home/x/vllm-work/p5
CTR=${CTR:-p5-moe}
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
MODEL_IN=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
SPECS=("$@")
mkdir -p "$DIR"
LOG="$DIR/profdriver.log"

if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=("eagerprof:EAGER=1;MAX_LEN=16384;MAX_NUM_SEQS=48;ACTIVE_ITERS=48")
fi

# Guard: never stack two GPU jobs. Two drivers sharing a container name silently
# destroy each other (burned 2026-09-11).
BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore|p5_moe\.py|p5_prof\.py" \
       | grep -v grep | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "P5PROF_ABORT busy=$BUSY; not stacking" | tee -a "$LOG"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  docker run -d --name "$CTR" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --shm-size 16g --ipc=host \
    -v "$MODELS_HOST":/models:ro -v "$DIR":/work \
    "$IMAGE" sleep infinity >/dev/null || {
      echo "P5PROF_ABORT docker run failed" | tee -a "$LOG"; exit 1; }
fi

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ 2>/dev/null | head -1)
CLK="$DIR/prof-clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

{
  echo "=== P5 PROF $(date -Iseconds) ==="
  echo "specs: ${SPECS[*]}"
  echo "clock: force_level=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null)"
} | tee -a "$LOG"

for SPEC in "${SPECS[@]}"; do
  LABEL="${SPEC%%:*}"; VARS="${SPEC#*:}"
  O="$DIR/$LABEL"; rm -rf "$O"; mkdir -p "$O/prof"
  echo "=== prof arm $LABEL start $(date -Iseconds) vars='$VARS' ===" | tee -a "$LOG"

  {
    echo "OUT=/work/$LABEL"
    echo "MODEL=$MODEL_IN"
    if [ -n "$VARS" ]; then
      IFS=';' read -ra KVS <<< "$VARS"
      for kv in "${KVS[@]}"; do echo "$kv"; done
    fi
  } > "$O/env.sh"

  docker exec -d "$CTR" bash -lc \
    "set -a; . /work/$LABEL/env.sh; set +a; \
     python3 -u /work/p5_prof.py > /work/$LABEL/prof.log 2>&1"

  for i in $(seq 1 200); do            # up to 50 min
    sleep 15
    grep -qE "P5PROF_DONE_OK|Traceback|SystemExit" "$O/prof.log" 2>/dev/null && break
    docker ps --format '{{.Names}}' | grep -qx "$CTR" || {
      echo "!! container died during $LABEL" | tee -a "$LOG"; break; }
  done
  echo "=== prof arm $LABEL done $(date -Iseconds) ===" | tee -a "$LOG"
  grep -E "^P5PROF_STEP|P5PROF_DONE|P5PROF_ABORT|ValueError|Error" "$O/prof.log" \
    2>/dev/null | head -20 | tee -a "$LOG"

  echo "--- buckets ---" | tee -a "$LOG"
  python3 /work/p5_kernels.py "/work/$LABEL/prof" "/work/$LABEL/p5_wall.json" 2>&1 \
    | tee -a "$LOG"
done

kill $SAMPLER 2>/dev/null; trap - EXIT
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" \
  | tee -a "$LOG"
echo "P5PROF_ALL_DONE $(date -Iseconds)" | tee -a "$LOG"
