#!/usr/bin/env bash
# c3_video_budget_probe.sh -- find a video budget that lets the vision dummy
# profile fit in 16GB, then check what the resulting max video tokens are.
#
# Mechanism (vllm/model_executor/models/qwen3_vl.py:991 get_max_video_tokens):
# the profile's dummy video size comes from
#   mm_processor_kwargs["size"]["longest_edge"]   (fallback: video processor
#   config, 16777216 px here), divided by temporal_patch_size, at num_frames=2.
# So `--mm-processor-kwargs '{"size":{...}}'` is the knob to sweep.
#
# EAGER=1 keeps each attempt to weight load + profile (~2 min).
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8160
export EAGER=1
export MM='{"image":1,"video":1}'

try() {
  local label=$1 proc_kwargs=$2 log=/tmp/vprobe.log
  echo "########## $label  size_kwargs=${proc_kwargs:-<model default>}"
  if [ -n "$proc_kwargs" ]; then
    gfx906-serve "$MODEL" --mm-processor-kwargs "$proc_kwargs" > "$log" 2>&1 &
  else
    gfx906-serve "$MODEL" > "$log" 2>&1 &
  fi
  local srv=$! rc=1
  for i in $(seq 1 36); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then rc=0; break; fi
    kill -0 $srv 2>/dev/null || { rc=2; break; }
    sleep 10
  done
  case $rc in
    0) echo "${label}_RESULT=STARTED";;
    1) echo "${label}_RESULT=TIMEOUT";;
    *) echo "${label}_RESULT=CRASHED";;
  esac
  grep -a -oE "Encoder cache will be initialized with a budget of [0-9]+ tokens, and profiled with [0-9]+ [a-z]+ items of the maximum feature size" "$log" | tail -1
  grep -a -oE "Tried to allocate [0-9.]+ [KM]iB.*" "$log" | tail -1
  grep -a -oE "ValueError: .*|maximum number of tokens.*" "$log" | tail -1
  bash /work/kill_serve.sh >/dev/null 2>&1
  sleep 6
}

try baseline ""
for L in 4194304 2097152 1048576; do
  try "L$L" "{\"size\": {\"longest_edge\": $L, \"shortest_edge\": 65536}}"
done
echo VPROBE_DONE
