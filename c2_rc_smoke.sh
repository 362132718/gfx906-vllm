#!/usr/bin/env bash
# c2_rc_smoke.sh -- inside a container started from the DELIVERY image
# (vllm-gfx906:rebuilt), with nothing but the documented run flags and a
# read-only model mount: self-check, first start with delivery defaults,
# text + image probes. Also records the cold-cache first-start time that
# QUICKSTART promises.
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8000
export SERVED_NAME=qwen38
NAME="$SERVED_NAME"
LOG=/tmp/rc_serve.log

echo "--- image/env state before serve ---"
ls -a /root/.cache 2>&1 | head -3
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-unset} VLLM_WNA16_EXLLAMA=${VLLM_WNA16_EXLLAMA:-unset}"

gfx906-selfcheck > /tmp/rc_selfcheck.log 2>&1; echo "SELFCHECK_RC=$?"
tail -3 /tmp/rc_selfcheck.log

gfx906-serve "$MODEL" > "$LOG" 2>&1 &
S=$!
T0=$(date +%s)
READY=0
for i in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
  kill -0 $S 2>/dev/null || { echo FIRST_START_DIED; tail -30 "$LOG"; exit 1; }
  sleep 10
done
[ "$READY" = 1 ] || { echo FIRST_START_TIMEOUT; tail -30 "$LOG"; exit 1; }
echo "FIRST_START_READY_AFTER=$(( $(date +%s) - T0 ))s"
grep -a -oE "torch.compile took [0-9.]+ s" "$LOG" | head -1
grep -a -oE "Using (WNA16ExllamaKernel|TritonW4A16LinearKernel) for CompressedTensorsWNA16" "$LOG" | sort -u | head -2
grep -a -oE "GPU KV cache size: [0-9,]+ tokens.*" "$LOG" | head -1
grep -a -oE "\[serve\].*" "$LOG" | head -1

python3 - "$PORT" "$NAME" <<'PY'
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
def ask(mt):
    body = json.dumps({"model": name, "prompt": "The capital of France is",
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d, time.time() - t0
d, el = ask(24)
print("REQ1: %.1f tok/s %r" % (d["usage"]["completion_tokens"] / el, d["choices"][0]["text"][:40]))
d, el = ask(64)
print("REQ2_WARM: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
PY

# a3_client.py is docker cp'd into /tmp before this script runs
test -f /tmp/a3_client.py && python3 -u /tmp/a3_client.py "$PORT" 2>&1 | tail -3 || echo "IMAGE_PROBE_SKIPPED"

kill $S 2>/dev/null; sleep 8; kill -9 $S 2>/dev/null
pgrep -f 'VLLM::Wor' | xargs -r kill -9 2>/dev/null
echo "cache after run:"; du -sh /root/.cache/vllm-w4a16-* 2>/dev/null
echo C2_RC_SMOKE_DONE
