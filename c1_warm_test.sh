#!/usr/bin/env bash
# c1_warm_test.sh -- time a same-path warm restart of the delivery defaults
# (cache dir ~/.cache/vllm-w4a16-exllama already populated) and probe decode.
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8140
export SERVED_NAME=qwen38
LOG=/tmp/c1_warm.log
du -sh "$HOME/.cache/vllm-w4a16-exllama" 2>/dev/null
gfx906-serve "$MODEL" > "$LOG" 2>&1 &
S=$!
T0=$(date +%s)
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "WARM_READY_AFTER=$(( $(date +%s) - T0 ))s"; break; }
  kill -0 $S 2>/dev/null || { echo WARM_SERVER_DIED; tail -25 "$LOG"; exit 1; }
  sleep 10
done
grep -a -oE "(torch.compile took [0-9.]+ s|Using cache directory.*|Using (WNA16ExllamaKernel|TritonW4A16LinearKernel))" "$LOG" | sort -u | head -5
python3 - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
def ask(mt):
    body = json.dumps({"model": "qwen38", "prompt": "The capital of France is",
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d, time.time() - t0
d, el = ask(24)
print("REQ1 (cold request): %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
d, el = ask(64)
print("REQ2 (warm): %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
PY
kill $S 2>/dev/null; sleep 8; kill -9 $S 2>/dev/null
bash /work/kill_serve.sh >/dev/null 2>&1
echo WARM_TEST_DONE
