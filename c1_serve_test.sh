#!/usr/bin/env bash
# c1_serve_test.sh -- exercise the delivered entry point (gfx906-serve) on the
# 27B W4A16 model with NO extra flags: proves quantization + multimodal
# auto-detection, text and image paths, and graph-mode decode speed.
set -uo pipefail
export PORT="${PORT:-8120}"
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
LOG=/tmp/c1_server.log

# also exercises the SERVED_NAME override (a3_client.py asks for "qwen38")
export SERVED_NAME=qwen38
NAME="$SERVED_NAME"

gfx906-serve "$MODEL" > "$LOG" 2>&1 &
SRV=$!
echo "server pid=$SRV"
sleep 5
echo "RESOLVED_CMD: $(tr '\0' ' ' < /proc/$SRV/cmdline 2>/dev/null)"

READY=0
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; echo "READY_AFTER=$((i*10))s"; break; fi
  kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -50 "$LOG"; exit 1; }
  sleep 10
done
[ "$READY" = 1 ] || { echo TIMEOUT; tail -50 "$LOG"; exit 1; }

echo "--- auto-detected config lines ---"
grep -iE "quantization|compressed.tensors|attention.backend|cudagraph|graph mode|limit_mm|multimodal|KV cache size|Maximum concurrency" "$LOG" | head -20

echo "--- TEXT (greedy) ---"
curl -s "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$NAME\",\"prompt\":\"Paris is the capital of\",\"max_tokens\":16,\"temperature\":0}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("TEXT_OUT:", repr(d["choices"][0]["text"]))'

echo "--- DECODE SPEED (batch1, 64 tok, graph mode expected ~22 tok/s) ---"
PORT=$PORT NAME="$NAME" python3 - <<'PY'
import json, os, time, urllib.request
port, name = os.environ["PORT"], os.environ["NAME"]
body = json.dumps({"model": name, "prompt": "The capital of France is",
                   "max_tokens": 64, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                             data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=600))
el = time.time() - t0
n = d["usage"]["completion_tokens"]
print("DECODE: %d tok in %.1fs = %.1f tok/s" % (n, el, n / el))
PY

echo "--- IMAGE (a3_client keyword check) ---"
python3 -u /work/a3_client.py "$PORT"; echo "IMAGE_RC=$?"

kill $SRV 2>/dev/null
sleep 8
kill -9 $SRV 2>/dev/null
echo C1_TEST_DONE
