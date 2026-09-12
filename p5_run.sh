#!/usr/bin/env bash
# p5_run.sh -- generic one-arm GPU probe runner (fresh-or-existing container).
#
# Usage:  SCRIPT=p5_lin_census.py bash p5_run.sh "label:VAR=1;VAR2=2"
#   SCRIPT   host path under /home/x/vllm-work/p5 (default p5_lin_census.py)
#   VAR=V   arm env; separator is ';' because values may contain commas/JSON
set -uo pipefail

DIR=/home/x/vllm-work/p5
CTR=${CTR:-p5-moe}
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
MODEL_IN=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
SCRIPT=${SCRIPT:-p5_lin_census.py}
SPEC=${1:-"census:EAGER=1;MAX_LEN=16384;MAX_NUM_SEQS=48;MM=0;GMU=0.92;N_GEN=4"}
LOG="$DIR/rundriver.log"

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore|p5_moe\.py|p5_prof\.py|p5_decode\.py|p5_lin_census\.py" \
       | grep -v grep | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "RUN_ABORT busy=$BUSY; not stacking" | tee -a "$LOG"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  docker run -d --name "$CTR" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --shm-size 16g --ipc=host \
    -v "$MODELS_HOST":/models:ro -v "$DIR":/work \
    "$IMAGE" sleep infinity >/dev/null || { echo "RUN_ABORT docker run failed" | tee -a "$LOG"; exit 1; }
fi

LABEL="${SPEC%%:*}"; VARS="${SPEC#*:}"
O="$DIR/run-$LABEL"; rm -rf "$O"; mkdir -p "$O"
{
  echo "OUT=/work/run-$LABEL"
  echo "MODEL=$MODEL_IN"
  if [ -n "$VARS" ]; then
    IFS=';' read -ra KVS <<< "$VARS"
    for kv in "${KVS[@]}"; do echo "$kv"; done
  fi
} > "$O/env.sh"

echo "=== RUN $LABEL $(date -Iseconds) script=$SCRIPT vars='$VARS' ===" | tee -a "$LOG"
HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ | head -1)
CLK="$DIR/run-clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

docker exec -d "$CTR" bash -lc \
  "set -a; . /work/run-$LABEL/env.sh; set +a; \
   python3 -u /work/$SCRIPT > /work/run-$LABEL/run.log 2>&1"

for i in $(seq 1 160); do
  sleep 15
  grep -qE "CENSUS_DONE_OK|LIN_DONE_OK|LIN_FAIL|DECODE_DONE_OK|P5PROF_DONE_OK|P5_DONE_OK|Traceback|SystemExit" \
    "$O/run.log" 2>/dev/null && break
  docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "!! container died" | tee -a "$LOG"; break; }
done

kill $SAMPLER 2>/dev/null; trap - EXIT
echo "=== RUN $LABEL done $(date -Iseconds) ===" | tee -a "$LOG"
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" | tee -a "$LOG"
echo "RUN_ALL_DONE" | tee -a "$LOG"
