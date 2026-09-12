#!/usr/bin/env bash
# lcpp_ref.sh -- the reference the user actually cares about: llama.cpp on the
# SAME box, SAME models, measured with llama-bench (self-reported pp/tg, fixed
# methodology) instead of hand-timed generation.
#
# Why this matters (2026-09-11): the user's llama-swap MoE launch recipe is
#   -sm layer --tensor-split 44/56   <- LAYER split == pipeline parallel:
#                                       zero per-layer cross-GPU communication
#   --spec-type ngram-mod            <- n-gram speculative decoding
#   --cache-type-k/v q8_0            <- quantized KV
# vLLM TP2 instead all-reduces twice per transformer block. So the comparison is
# NOT apples-to-apples on either communication or speculation; this script
# isolates the engine difference by using plain tg (no spec decode) as well as
# the spec-decode number.
#
# MUST run with vLLM stopped (both want the same 32 GB).
set -uo pipefail

BIN=/home/x/llamacpp-gfx906-slot-save/build-rocm724-fix/bin/llama-bench
MOE=/home/x/models/Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-LynnStyle-Q4-imatrix.gguf
D27=/home/x/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q6_K.gguf
OUT=/home/x/vllm-work/lcpp-ref
mkdir -p "$OUT"

BUSY=$(ps -eo stat,args | grep -E "vllm serve|EngineCore|p5_moe\.py|p5_prof\.py|llama-server" \
       | grep -v grep | awk '$1 !~ /^Z/' | wc -l)
if [ "$BUSY" != 0 ]; then
  echo "LCPP_ABORT busy=$BUSY; stop vLLM/llama first"
  ps -eo pid,etime,args | grep -E "[v]llm serve|[E]ngineCore|[p]5_(moe|prof)\.py|[l]lama-server" | cut -c1-120
  exit 1
fi

HWM=$(ls -d /sys/class/drm/card0/device/hwmon/hwmon*/ | head -1)
CLK="$OUT/clocks.txt"; : > "$CLK"
( while :; do
    echo "$(date +%s) $(cat $HWM/freq1_input 2>/dev/null) $(cat $HWM/temp1_input 2>/dev/null)" >> "$CLK"
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run() {   # run <label> <model> [extra args...]
  local label="$1" model="$2"; shift 2
  echo "=== llama-bench $label $(date -Iseconds) ==="
  echo "cmd: llama-bench -m $(basename "$model") -ngl 999 -sm layer -ts 44/56 -fa 1 $*"
  "$BIN" -m "$model" -ngl 999 -sm layer -ts 44/56 -fa 1 \
         -p 512 -n 128 -r 2 "$@" 2>&1 | tee "$OUT/$label.txt"
  echo
}

# tg is the number that matters; -r 2 keeps it short (each load is ~1 min).
run moe-35ba3b "$MOE"
run dense-27b  "$D27"

kill $SAMPLER 2>/dev/null; trap - EXIT
echo "=== 时钟 ==="
awk '{f=$2/1e6; if(f>mx)mx=f; t=$3/1000; if(t>mt)mt=t; n++}
     END{printf "samples=%d max_sclk=%.0fMHz max_temp=%.0fC\n", n, mx, mt}' "$CLK"
echo "LCPP_ALL_DONE $(date -Iseconds)"
