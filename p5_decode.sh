#!/usr/bin/env bash
# p5_decode.sh -- honest decode-step measurement, one engine per spec.
#
# Usage:  bash p5_decode.sh "tp2graph:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=48;MM=0;GMU=0.92"
# Arm vars (separator ';'): EAGER, MAX_LEN, MAX_NUM_SEQS, GMU, PP, MM,
#                           ATTN_BACKEND, N_GEN, CTX, CONC.
set -uo pipefail

DIR=/home/x/vllm-work/p5
CTR=${CTR:-p5-moe}
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
MODEL_IN=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
SPECS=("$@")
mkdir -p "$DIR"
LOG="$DIR/decodedriver.log"

if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=("tp2graph:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=48;MM=0;GMU=0.92")
fi

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore|p5_moe\.py|p5_prof\.py|p5_decode\.py" \
       | grep -v grep | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "DECODE_ABORT busy=$BUSY; not stacking" | tee -a "$LOG"
  ps -eo pid,etime,args | grep -E "[v]llm serve|[E]ngineCore|[p]5_" | cut -c1-120
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
      echo "DECODE_ABORT docker run failed" | tee -a "$LOG"; exit 1; }
fi

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ 2>/dev/null | head -1)
CLK="$DIR/decode-clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

{
  echo "=== P5 DECODE $(date -Iseconds) ==="
  echo "specs: ${SPECS[*]}"
  echo "clock: force_level=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null)"
} | tee -a "$LOG"

for SPEC in "${SPECS[@]}"; do
  LABEL="${SPEC%%:*}"; VARS="${SPEC#*:}"
  O="$DIR/dec-$LABEL"; rm -rf "$O"; mkdir -p "$O"
  echo "=== dec arm $LABEL start $(date -Iseconds) vars='$VARS' ===" | tee -a "$LOG"

  {
    echo "OUT=/work/dec-$LABEL"
    echo "MODEL=$MODEL_IN"
    if [ -n "$VARS" ]; then
      IFS=';' read -ra KVS <<< "$VARS"
      for kv in "${KVS[@]}"; do echo "$kv"; done
    fi
  } > "$O/env.sh"

  docker exec -d "$CTR" bash -lc \
    "set -a; . /work/dec-$LABEL/env.sh; set +a; \
     python3 -u /work/p5_decode.py > /work/dec-$LABEL/decode.log 2>&1"

  for i in $(seq 1 160); do            # up to 40 min
    sleep 15
    grep -qE "DECODE_DONE_OK|DECODE_FAIL|Traceback|SystemExit" "$O/decode.log" 2>/dev/null && break
    docker ps --format '{{.Names}}' | grep -qx "$CTR" || {
      echo "!! container died during $LABEL" | tee -a "$LOG"; break; }
  done
  echo "=== dec arm $LABEL done $(date -Iseconds) ===" | tee -a "$LOG"
  grep -E "^DECODE_STEP|DECODE_DONE|DECODE_FAIL|ValueError|Error" "$O/decode.log" \
    2>/dev/null | head -12 | tee -a "$LOG"
done

kill $SAMPLER 2>/dev/null; trap - EXIT
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" \
  | tee -a "$LOG"
echo "DECODE_ALL_DONE $(date -Iseconds)" | tee -a "$LOG"
