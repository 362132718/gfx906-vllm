#!/usr/bin/env bash
# a2_run.sh -- long-context measurement: serve 27B W4A16 TP2, hit it with
# 2K/8K/16K prompts via streaming completions, report prefill/decode tok/s.
# Run inside container:  bash /work/a2_run.sh   (detached via docker exec -d)
set -uo pipefail
LOG=/tmp/a2_server.log
PORT=8100
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16

export VLLM_WNA16_EXLLAMA=1
python3 -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen38 \
  --dtype float16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 18432 \
  --attention-backend TRITON_ATTN --quantization compressed-tensors \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --disable-custom-all-reduce --enforce-eager \
  --no-enable-prefix-caching \
  --port $PORT > $LOG 2>&1 &
SRV=$!
echo "server pid=$SRV"

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo READY_AFTER=${i}0s; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo SERVER_DIED; tail -30 $LOG; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo SERVER_TIMEOUT; tail -30 $LOG; exit 1; }

python3 -u /work/a2_client.py $PORT
RC=$?
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "A2_DONE rc=$RC"
exit $RC
