#!/usr/bin/env bash
# c3_graph_video2.sh -- graphs ON but inductor compile skipped (mode=0) so the
# run stays out of the multi-minute CPU compile window that keeps killing this
# host. Purpose: does CUDA-graph decode coexist with an enabled video modality?
set -uo pipefail
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
export PORT=8166
export SERVED_NAME=qwen38
LOG=/tmp/c3gv2.log

gfx906-serve "$MODEL" --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48]}' > "$LOG" 2>&1 &
S=$!
T0=$(date +%s); rc=1
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then rc=0; break; fi
  kill -0 $S 2>/dev/null || { rc=2; break; }
  sleep 10
done
echo "READY_AFTER=$(( $(date +%s) - T0 ))s rc=$rc"
grep -a -oE "\[serve\].*" "$LOG" | head -1
grep -a -oE "Using (WNA16ExllamaKernel|TritonW4A16LinearKernel) for CompressedTensorsWNA16" "$LOG" | sort -u
grep -a -oE "Encoder cache will be initialized with a budget of [0-9]+ tokens[^.]*" "$LOG" | tail -1
grep -a -oE "GPU KV cache size: [0-9,]+ tokens.*" "$LOG" | tail -1
[ $rc = 0 ] || { tail -c 400 "$LOG" | tr -cd "\11\12\15\40-\176" | tail -3; exit 1; }

python3 - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
def ask(mt):
    body = json.dumps({"model": "qwen38", "prompt": "The capital of France is",
                       "max_tokens": mt, "temperature": 0}).encode()
    r = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port, data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=900))
    return d, time.time() - t0
d, el = ask(24); print("TEXT_REQ1: %.1f tok/s" % (d["usage"]["completion_tokens"]/el))
d, el = ask(64); print("TEXT_WARM: %.1f tok/s %r" % (d["usage"]["completion_tokens"]/el, d["choices"][0]["text"][:32]))
PY
echo "--- image ---"; python3 -u /tmp/a3_client.py "$PORT" 2>&1 | grep -aE "SECONDS|KEYWORD_SCORE"
for N in 8 16 32; do echo "--- video frames=$N ---"; python3 -u /tmp/a3_video_client.py "$PORT" qwen38 $N 2>&1 | grep -aE "SECONDS|VIDEO_KEYWORD_SCORE|HTTP_[0-9]+_BODY|prompt_tokens" | head -4; done
grep -a -oE "Tried to allocate [0-9.]+ [KMGT]iB|OutOfMemoryError" "$LOG" | tail -2
kill $S 2>/dev/null; sleep 8; kill -9 $S 2>/dev/null
bash /work/kill_serve.sh >/dev/null 2>&1
echo C3_GRAPH_VIDEO2_DONE
