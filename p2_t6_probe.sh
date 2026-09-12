#!/usr/bin/env bash
# p2_t6_probe.sh -- measure ONE arm against an already-running serve, writing the
# same probe_$ARM.txt the main script's gate reads. Exists because run 1 measured
# nothing: `python3 - ... | tee "$PROBE" <<'PY'` attaches the heredoc to TEE, so
# the file got the python source and python got EOF. No pipeline here.
set -uo pipefail
OUT=${1:-/home/x/vllm-work/p2t6}; ARM=${2:-on}; CNAME=${3:-p2-t6}
PROBE=$OUT/probe_$ARM.txt
docker exec -i "$CNAME" bash -s -- "$PROBE" <<'INSIDE'
set -uo pipefail
PROBE=$1
PORT=8000
NAME=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
[ -n "$NAME" ] || { echo "FATAL: could not resolve served model id"; exit 1; }
echo "resolved_model_id=$NAME  p2_env=${VLLM_WNA16_P2_GEMV:-unset(=> impl default 1)}"
python3 - "$PORT" "$NAME" <<'PY' > "$PROBE" 2>&1
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
print("REQ1: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
d, el = ask(64, "The capital of France is")
print("REQ2_WARM: %.1f tok/s" % (d["usage"]["completion_tokens"] / el))
rates = []
for i in range(3):
    d, el = ask(256, "请用一句话解释为什么显卡的显存带宽会限制大模型推理速度。")
    r = d["usage"]["completion_tokens"] / el
    rates.append(r)
    print("DECODE256 rep%d: %.1f tok/s" % (i + 1, r))
print("DECODE256 best=%.1f mean=%.1f" % (max(rates), sum(rates) / len(rates)))
PY
rc=$?
grep -q "DECODE256 best=" "$PROBE" || { echo "PROBE_EMPTY rc=$rc"; cat "$PROBE"; exit 1; }
cat "$PROBE"
INSIDE
echo "probe rc=$?"
