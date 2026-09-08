#!/usr/bin/env bash
# c1_coda.sh -- remaining C1 verifications, one detached run:
#   STEP 1: 27B with VLLM_WNA16_EXLLAMA=0 -> must start on the Triton kernel
#           from its OWN cache dir (previously crashed with a stride mismatch
#           because both modes shared ~/.cache/vllm).
#   STEP 2: text-only model, TP=1, discovered via MODELS_ROOT -> proves the
#           defaults are safe for non-VL / non-quantized users (no mm flags,
#           no --quantization flag).
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8130
export SERVED_NAME=qwen38

wait_ready() {
  local srv=$1 port=$2 label=$3 i
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then echo "${label}_READY_AFTER=$((i*10))s"; return 0; fi
    kill -0 $srv 2>/dev/null || { echo "${label}_SERVER_DIED"; return 1; }
    sleep 10
  done
  echo "${label}_TIMEOUT"; return 1
}

ask() {
  local port=$1 name=$2 mt=${3:-24}
  python3 - "$port" "$name" "$mt" <<'PY'
import json, sys, time, urllib.request
port, name, mt = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.dumps({"model": name, "prompt": "The capital of France is",
                   "max_tokens": mt, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                             data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=600))
el = time.time() - t0
n = d["usage"]["completion_tokens"]
print("DECODE: %d tok in %.1fs = %.1f tok/s" % (n, el, n / el))
print("OUT:", repr(d["choices"][0]["text"][:60]))
PY
}

echo "########## STEP1_27B_TRITON_MODE"
export VLLM_WNA16_EXLLAMA=0
gfx906-serve "$MODEL" > /tmp/c1_c1.log 2>&1 &
S=$!
sleep 6
grep -a -oE "\[serve\].*" /tmp/c1_c1.log | head -1
if wait_ready $S $PORT STEP1; then
  grep -a -oE "Using (TritonW4A16LinearKernel|WNA16ExllamaKernel) for CompressedTensorsWNA16" /tmp/c1_c1.log | sort -u
  ask $PORT qwen38
else
  grep -a -m2 -E "Error in op|expected size" /tmp/c1_c1.log
fi
kill $S 2>/dev/null; sleep 8; kill -9 $S 2>/dev/null
bash /work/kill_serve.sh >/dev/null 2>&1
unset VLLM_WNA16_EXLLAMA
sleep 5

echo "########## STEP2_TEXT_ONLY_TP1"
export PORT=8131
export SERVED_NAME=small
export TP=1
export MODELS_ROOT=/tmp/c1_models
rm -rf "$MODELS_ROOT"; mkdir -p "$MODELS_ROOT"
ln -s /home/x/models/vllm/Qwen3-0.6B-real "$MODELS_ROOT/Qwen3-0.6B-real"
gfx906-serve > /tmp/c1_c2.log 2>&1 &
S2=$!
sleep 6
CMD=$(tr '\0' ' ' < /proc/$S2/cmdline 2>/dev/null)
echo "RESOLVED_CMD: $CMD"
echo "$CMD" | grep -q 'limit-mm-per-prompt' && echo "STEP2_UNEXPECTED_MM=1" || echo "STEP2_MM_ABSENT=1"
echo "$CMD" | grep -q -- '--quantization' && echo "STEP2_UNEXPECTED_QUANT=1" || echo "STEP2_QUANT_ABSENT=1"
echo "$CMD" | grep -q -- '--tensor-parallel-size 1' && echo "STEP2_TP1=1" || echo "STEP2_TP1=0"
if wait_ready $S2 $PORT STEP2; then ask $PORT small; fi
kill $S2 2>/dev/null; sleep 6; kill -9 $S2 2>/dev/null
bash /work/kill_serve.sh >/dev/null 2>&1
rm -rf "$MODELS_ROOT"
echo C1_CODA_DONE
