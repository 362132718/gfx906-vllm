#!/usr/bin/env bash
# p5_moe.sh -- MoE smoke on the delivery image, one engine per spec.
#
# Runs on the GPU host. Fresh container from vllm-gfx906:latest, model mounted
# read-only. Each spec is a separate engine build (max_len / eager are fixed at
# construction).
#
# Usage:
#   bash p5_moe.sh "graph16k:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=53" \
#                  "pp2:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=53;PP=2"
#
# Arm vars (separator ';'): EAGER, MAX_LEN, MAX_NUM_SEQS, MM, PP, SPEC, OUT_TOK.
# SPEC takes raw JSON, e.g.
#   SPEC={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}
#
# TP1 is not an option: 24.3 GB of weights do not fit a 16 GB card.
# PP2 = layer split (TP1+PP2) -- no per-layer all-reduce, the llama.cpp shape.
# Clock AUTO, recorded from hwmon (rocm-smi's sclk is unreliable on this box).
set -uo pipefail

DIR=/home/x/vllm-work/p5
CTR=p5-moe
IMAGE=vllm-gfx906:latest
MODELS_HOST=/home/x/models
MODEL_IN=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
SPECS=("$@")
mkdir -p "$DIR"
LOG="$DIR/driver.log"

if [ ${#SPECS[@]} -eq 0 ]; then
  # 1st: the config that mirrors gfx906-serve (mm on, TRITON_ATTN encoder).
  # 2nd: same but with CUDA graphs (FULL_DECODE_ONLY) and the GDN-mandated
  #      max_num_seqs, i.e. the first honest measurement of the delivery path.
  # 3rd: vision fully disabled -- guarantees a text-only MoE verdict even if the
  #      vision tower turns out to be the broken part on gfx906.
  SPECS=("eager8k:EAGER=1;MAX_LEN=8192"
         "graph16k:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=53"
         "novis8k:MM=0;EAGER=1;MAX_LEN=8192")
fi

{
  echo "=== P5 MOE SMOKE $(date -Iseconds) ==="
  echo "specs: ${SPECS[*]}"
  echo "model: $MODEL_IN"
  echo "clock: force_level=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null)"
} | tee -a "$LOG"

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore" | grep -v grep \
       | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "P5_ABORT busy=$BUSY vllm processes alive; not stacking" | tee -a "$LOG"
  exit 1
fi

docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v "$MODELS_HOST":/models:ro -v "$DIR":/work \
  "$IMAGE" sleep infinity >/dev/null || {
    echo "P5_ABORT docker run failed" | tee -a "$LOG"; exit 1; }

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ 2>/dev/null | head -1)
CLK="$DIR/clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

for SPEC in "${SPECS[@]}"; do
  LABEL="${SPEC%%:*}"; VARS="${SPEC#*:}"
  O="$DIR/$LABEL"; rm -rf "$O"; mkdir -p "$O"
  echo "=== arm $LABEL start $(date -Iseconds) vars='$VARS' ===" | tee -a "$LOG"

  # Env goes through a sourced file, not a command line: SPEC carries raw JSON
  # (commas and braces), which the old "IFS=',' + inline assignment" form would
  # have shredded. Separator is ';' for the same reason.
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
     python3 -u /work/p5_moe.py > /work/$LABEL/probe.log 2>&1"

  for i in $(seq 1 240); do            # up to 60 min per arm
    sleep 15
    if grep -qE "P5_DONE_OK|P5_DONE_FAIL|Traceback|SystemExit" "$O/probe.log" 2>/dev/null; then
      break
    fi
    docker ps --format '{{.Names}}' | grep -qx "$CTR" || {
      echo "!! container died during $LABEL" | tee -a "$LOG"; break; }
  done
  echo "=== arm $LABEL done $(date -Iseconds) ===" | tee -a "$LOG"
  grep -E "^P5_STEP|P5_DONE|Error|Traceback|out of memory|No available memory" \
    "$O/probe.log" 2>/dev/null | head -20 | tee -a "$LOG"
  cp "$O/probe.log" "$DIR/probe-$LABEL.log" 2>/dev/null
done

kill $SAMPLER 2>/dev/null; trap - EXIT
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "clock samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK" \
  | tee -a "$LOG"
echo "P5_ALL_DONE $(date -Iseconds)" | tee -a "$LOG"
