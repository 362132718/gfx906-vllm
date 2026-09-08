#!/usr/bin/env bash
# c1_textpath_test.sh -- second branch of the entry point: a text-only model on
# ONE GPU, discovered via MODELS_ROOT (no explicit path, no quantization flag,
# no multimodal flags). Proves the defaults are safe for non-VL users too.
set -uo pipefail
export PORT="${PORT:-8121}"
export TP=1
export MODELS_ROOT=/tmp/c1_models
export SERVED_NAME=small

rm -rf "$MODELS_ROOT"
mkdir -p "$MODELS_ROOT"
ln -s /home/x/models/vllm/Qwen3-0.6B-real "$MODELS_ROOT/Qwen3-0.6B-real"

LOG=/tmp/c1_text_server.log
gfx906-serve > "$LOG" 2>&1 &
SRV=$!
echo "server pid=$SRV"
sleep 6
CMD=$(tr '\0' ' ' < /proc/$SRV/cmdline 2>/dev/null)
echo "RESOLVED_CMD: $CMD"
echo "$CMD" | grep -q 'limit-mm-per-prompt' && echo "UNEXPECTED_MM_FLAGS=1" || echo "MM_FLAGS_ABSENT=1"
echo "$CMD" | grep -q ' --quantization ' && echo "UNEXPECTED_QUANT_FLAG=1" || echo "QUANT_FLAG_ABSENT=1"
echo "$CMD" | grep -q -- '--tensor-parallel-size 1' && echo "TP1=1" || echo "TP1=0"

READY=0
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; echo "READY_AFTER=$((i*5))s"; break; fi
  kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -40 "$LOG"; exit 1; }
  sleep 5
done
[ "$READY" = 1 ] || { echo TIMEOUT; tail -40 "$LOG"; exit 1; }

curl -s "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d '{"model":"small","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("TEXT_OUT:", repr(d["choices"][0]["text"]))'

kill $SRV 2>/dev/null; sleep 6; kill -9 $SRV 2>/dev/null
rm -rf "$MODELS_ROOT"
echo C1_TEXT_TEST_DONE
