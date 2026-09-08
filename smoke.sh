#!/bin/bash
# smoke.sh -- vllm serve startup + one completion under sitecustomize .pth
set -uo pipefail
export VLLM_WNA16_EXLLAMA=1
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
vllm serve "$MODEL" --served-model-name q --dtype float16 -tp 2 \
  --gpu-memory-utilization 0.85 --max-model-len 512 --max-num-seqs 48 \
  --attention-backend TRITON_ATTN --quantization compressed-tensors \
  --disable-custom-all-reduce --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48]}' \
  > /tmp/smoke_srv.log 2>&1 &
S=$!
READY=0
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then READY=1; break; fi
  kill -0 $S 2>/dev/null || { echo SERVER_DIED; tail -6 /tmp/smoke_srv.log; exit 1; }
  sleep 10
done
[ $READY = 1 ] || { echo TIMEOUT; tail -6 /tmp/smoke_srv.log; exit 1; }
curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"q","prompt":"The capital of France is","max_tokens":12,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("SMOKE_TEXT:", repr(d["choices"][0]["text"]))'
kill $S 2>/dev/null
sleep 5; kill -9 $S 2>/dev/null
echo SMOKE_DONE
