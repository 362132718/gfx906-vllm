#!/usr/bin/env bash
# p05_serve_probe.sh -- run ON THE HOST, detached. The serve-path tok/s at
# DELIVERY CLOCKS that p05_cold_serve.sh missed: its probe hardcoded a served
# model name, but gfx906-serve only honours SERVED_NAME if it is exported, so
# the default is basename($MODEL) and every request came back HTTP 404.
#
# Reuses the existing p05-cold container, whose writable layer now carries the
# 552 MB compile cache -> this cycle is the WARM arm, and it resolves the model
# id from /v1/models instead of assuming it.
set -uo pipefail
IMAGE=vllm-gfx906:latest
CNAME=p05-cold
MODEL=/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
OUT=/home/x/vllm-work/p0
PORT=8000

boot0=$(uptime -s)
echo "=== P05 SERVE PROBE START $(date -Is) boot_since=$boot0"
if ! docker ps --format '{{.Names}}' | grep -qx "$CNAME"; then
  echo "--- container not running, recreating (that makes this a COLD arm again)"
  docker rm -f "$CNAME" >/dev/null 2>&1
  docker run -d --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --shm-size 16g --ipc=host \
    -v /home/x/models:/home/x/models:ro \
    -v "$OUT":"$OUT" \
    -p $PORT:$PORT \
    "$IMAGE" sleep infinity >/dev/null || { echo "FATAL: docker run failed"; exit 1; }
fi

docker exec -i "$CNAME" bash -s -- "$MODEL" "$PORT" "$OUT" <<'INSIDE'
set -uo pipefail
MODEL=$1; PORT=$2; OUT=$3
LOG=$OUT/p05_serve_probe.log

# A straggler worker holding VRAM is what made the previous warm arm's KV come in
# 3.6% low, so insist on an idle pair before timing anything.
pgrep -f 'vllm serve|VLLM::Wor' | xargs -r kill -9 2>/dev/null
for i in $(seq 1 12); do
  free=$(python3 -c 'import torch;print(min(torch.cuda.mem_get_info(i)[0] for i in (0,1))/1e9)' 2>/dev/null || echo 0)
  awk -v f="$free" 'BEGIN{exit !(f>16.0)}' && { echo "preflight_vram_free_minGB=$free"; break; }
  echo "waiting for VRAM to be released: ${free} GB"
  sleep 10
done
echo "cache in this container: $(du -sh /root/.cache/vllm-w4a16-* 2>/dev/null | awk '{printf "%s(%s) ", $1,$2}' || echo none)"

gfx906-serve "$MODEL" > "$LOG" 2>&1 &
S=$!
T0=$(date +%s)
READY=0
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
  kill -0 $S 2>/dev/null || { echo "PROBE_DIED after $(( $(date +%s) - T0 ))s"; tail -40 "$LOG"; exit 1; }
  sleep 10
done
[ "$READY" = 1 ] || { echo "PROBE_TIMEOUT_1800s"; tail -40 "$LOG"; exit 1; }
echo "PROBE_READY_AFTER=$(( $(date +%s) - T0 ))s"
grep -a -oE "torch.compile took [0-9.]+ s" "$LOG" | head -1
grep -a -oE "GPU KV cache size: [0-9,]+ tokens.*" "$LOG" | head -1

MODELS=$(curl -sf "http://127.0.0.1:$PORT/v1/models")
echo "v1/models: $(echo "$MODELS" | tr -d '\n' | cut -c1-160)"
NAME=$(echo "$MODELS" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "resolved_model_id=$NAME"

python3 - "$PORT" "$NAME" <<'PY'
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]

def ask(mt, prompt):
    body = json.dumps({"model": name, "prompt": prompt,
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1200))
    return d, time.time() - t0

d, el = ask(24, "The capital of France is")
print("REQ1: %.1f tok/s %r" % (d["usage"]["completion_tokens"] / el, d["choices"][0]["text"][:40]))
d, el = ask(64, "The capital of France is")
print("REQ2_WARM: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
rates = []
for i in range(3):
    d, el = ask(256, "请用一句话解释为什么显卡的显存带宽会限制大模型推理速度。")
    r = d["usage"]["completion_tokens"] / el
    rates.append(r)
    print("DECODE256 rep%d: %.1f tok/s  (%d tok / %.1fs) %r" % (
        i + 1, r, d["usage"]["completion_tokens"], el, d["choices"][0]["text"][:24]))
print("DECODE256 best=%.1f mean=%.1f" % (max(rates), sum(rates) / len(rates)))
PY

echo "--- batch8 sanity (max_num_seqs=48, same as the 1425MHz A/B arm):"
python3 - "$PORT" "$NAME" <<'PY'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, name = sys.argv[1], sys.argv[2]

def one(_):
    body = json.dumps({"model": name, "prompt": "Write one sentence about GPUs.",
                       "max_tokens": 64, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1200))
    return d["usage"]["completion_tokens"], time.time() - t0

with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(one, range(8)))
tok = sum(t for t, _ in res)
el = max(e for _, e in res)
print("BATCH8: %d tok total over %.1fs = %.1f tok/s aggregate, per-stream %.1f" % (
    tok, el, tok / el, tok / 8 / el))
PY

kill $S 2>/dev/null; sleep 10; kill -9 $S 2>/dev/null
pgrep -f 'VLLM::Wor' | xargs -r kill -9 2>/dev/null
sleep 25
echo "remaining vllm procs after teardown: $(pgrep -cf 'vllm serve|VLLM::Wor' || echo 0)"
INSIDE
echo "=== P05 SERVE PROBE DONE $(date -Is) boot_since=$(uptime -s)"
