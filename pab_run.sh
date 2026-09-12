#!/usr/bin/env bash
# pab_run.sh -- run N engine arms sequentially in ONE container and summarise.
#
# Runs on the GPU host. One arm = one process = one engine build (attention
# backend / kv-cache-dtype are fixed at construction), but all arms share a
# single container so only the engine init is paid per arm.
#
# Usage:
#   bash pab_run.sh "baseline:" "gdn_on:VLLM_GFX906_GDN_OUTPUT_NORM=1"
#   bash pab_run.sh "fp16kv:" "fp8kv:KV_CACHE_DTYPE=fp8_e4m3"
#
# Each spec is  <label>:<ENV=VAL[,ENV=VAL...]>   (extra vars are exported into
# the probe process only).  Global knobs come from the environment:
#   CTXS REPS MAX_LEN MAX_NUM_SEQS DO_PREFILL
#
# Clock: AUTO and recorded. The host has a silent-power-loss history, so the
# driver runs detached and everything lands in $DIR/driver.log.
set -uo pipefail

DIR=/home/x/vllm-work/pab
CTR=pab-ab
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
SPECS=("$@")
mkdir -p "$DIR"
LOG="$DIR/driver.log"

if [ ${#SPECS[@]} -eq 0 ]; then
  echo "usage: pab_run.sh \"label:ENV=VAL,...\" ..." >&2; exit 2
fi

{
  echo "=== PAB RUN $(date -Iseconds) ==="
  echo "specs: ${SPECS[*]}"
  echo "global: CTXS=${CTXS:-2048,8192} REPS=${REPS:-2} MAX_LEN=${MAX_LEN:-16384} MAX_NUM_SEQS=${MAX_NUM_SEQS:-48} DO_PREFILL=${DO_PREFILL:-1}"
  echo "clock: force_level=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null)"
} | tee -a "$LOG"

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore" | grep -v grep \
       | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "PAB_ABORT busy=$BUSY vllm processes alive; not stacking" | tee -a "$LOG"
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
    echo "PAB_ABORT docker run failed" | tee -a "$LOG"; exit 1; }

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ 2>/dev/null | head -1)
CLK="$DIR/clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

for SPEC in "${SPECS[@]}"; do
  LABEL="${SPEC%%:*}"
  VARS="${SPEC#*:}"
  O="$DIR/$LABEL"; rm -rf "$O"; mkdir -p "$O"
  echo "=== arm $LABEL start $(date -Iseconds)  vars='$VARS' ===" | tee -a "$LOG"

  ENVSTR=""
  if [ -n "$VARS" ]; then
    IFS=',' read -ra KVS <<< "$VARS"
    for kv in "${KVS[@]}"; do ENVSTR="$ENVSTR $kv"; done
  fi

  docker exec -d "$CTR" bash -lc \
    "LABEL=$LABEL OUT=/work $ENVSTR \
     CTXS=${CTXS:-2048,8192} REPS=${REPS:-2} MAX_LEN=${MAX_LEN:-16384} \
     MAX_NUM_SEQS=${MAX_NUM_SEQS:-48} DO_PREFILL=${DO_PREFILL:-1} \
     python3 -u /work/pab_probe.py > /work/$LABEL/probe.log 2>&1"

  for i in $(seq 1 320); do           # up to 80 min per arm
    sleep 15
    if grep -qE "PAB_ARM_DONE|Traceback|SystemExit|FATAL|Error:" "$O/probe.log" 2>/dev/null; then
      break
    fi
    docker ps --format '{{.Names}}' | grep -qx "$CTR" || {
      echo "!! container died during $LABEL" | tee -a "$LOG"; break; }
  done
  echo "=== arm $LABEL done $(date -Iseconds) (done-marker=$(grep -c PAB_ARM_DONE "$O/probe.log" 2>/dev/null)) ===" | tee -a "$LOG"
  grep -E "^  ctx|engine ready|kv_capacity|PAB_ARM_DONE|Error|Traceback" "$O/probe.log" 2>/dev/null | tee -a "$LOG"
done

kill $SAMPLER 2>/dev/null; trap - EXIT
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" \
  | tee -a "$LOG"

echo "--- summary ---" | tee -a "$LOG"
python3 "$DIR/pab_compare.py" "$DIR" 2>&1 | tee -a "$LOG"
echo "PAB_ALL_DONE $(date -Iseconds)" | tee -a "$LOG"
