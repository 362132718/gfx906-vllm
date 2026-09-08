#!/usr/bin/env bash
# a3_probe.sh -- bring up the VL model with image modality enabled and run
# an end-to-end image Q&A. Video stays off in this first probe (its dummy
# profile is what OOMs at 128GiB).
set -uo pipefail
LOG=/tmp/a3_server.log
PORT=8101
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16

export VLLM_WNA16_EXLLAMA=1
python3 -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen38 \
  --dtype float16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --attention-backend TRITON_ATTN --quantization compressed-tensors \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-encoder-attn-backend TRITON_ATTN \
  --disable-custom-all-reduce --enforce-eager \
  --port $PORT > $LOG 2>&1 &
SRV=$!
echo "server pid=$SRV"
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo READY_AFTER=${i}0s; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo SERVER_DIED; tail -40 $LOG; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo SERVER_TIMEOUT; tail -40 $LOG; exit 1; }

python3 -u /work/a3_client.py $PORT
RC=$?
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "A3_PROBE_DONE rc=$RC"
exit $RC
