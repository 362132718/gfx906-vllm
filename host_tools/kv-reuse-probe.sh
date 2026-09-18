#!/bin/bash
# kv-reuse-probe.sh -- 量化「KV 复用」到底省下多少 prefill
#
# 原理：同一个超长 prompt 连发两次。
#   ① 冷跑：带唯一 nonce 的全新 prompt，必然重算 prefill
#   ② 重发：同 nonce 原样重发，命中前缀缓存（GPU 池，或 CPU/盘 offload 层）
# 两次 TTFT 之差 = 复用真实省下的时间。
#
# 用法（在宿主 /home/x/ 下执行）：
#   bash kv-reuse-probe.sh                      # 读 /home/x/.vllm-api-key，约 8k prompt
#   bash kv-reuse-probe.sh <API_KEY>
#   PROBE_TOKENS=32000 bash kv-reuse-probe.sh   # 换 prompt 长度
#
# 环境变量: CONTAINER(vllm) PORT(8000) MODEL PROBE_TOKENS(8000) PROBE_MAXTOK(32)
#
# ⚠ 注意：本项目已知坑 —— vLLM 前缀缓存默认开，测「纯 prefill 速度」必须关缓存 + 用唯一
#   nonce；本脚本反过来利用它来测「复用收益」，两次请求**必须**同 nonce。
set -euo pipefail

CONTAINER="${CONTAINER:-vllm}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen3.8-27B-OBLITERATED-AWQ-W4A16}"
PROBE_TOKENS="${PROBE_TOKENS:-8000}"
PROBE_MAXTOK="${PROBE_MAXTOK:-32}"

KEY="${1:-${API_KEY:-}}"
if [ -z "$KEY" ] && [ -s /home/x/.vllm-api-key ]; then KEY="$(cat /home/x/.vllm-api-key)"; fi
if [ -z "$KEY" ]; then
  echo "❌ 没有 API key：传参、设 \$API_KEY、或确保 /home/x/.vllm-api-key 存在"
  exit 1
fi

if [ "$(docker exec "$CONTAINER" curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
        -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)" != "200" ]; then
  echo "❌ $CONTAINER:$PORT 上没有可用服务（先 ./start-vllm.sh）"
  exit 1
fi

echo "── 当前服务的 KV 复用配置 ────────────────────────────────────"
# 用 "vllm [s]erve" 而不是 "vllm serve"：本行自身的命令行里含有这个 pattern 字面量，
# 直接写 "vllm serve" 会让 pgrep 匹配到自己 ⇒ 把 pattern 文本当成结果打出来。
docker exec "$CONTAINER" bash -lc \
  'for p in $(pgrep -f "vllm [s]erve"); do tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -oE -- "--kv-transfer-config [^ ]*|--kv-cache-dtype [^ ]*"; done | sort -u' 2>/dev/null \
  || true
echo "   （上面为空 = 只有 GPU 显存里的前缀缓存，没有跨实例常驻层）"
echo

PY=/tmp/kv_reuse_probe.py
cat > "$PY" <<'PYEOF'
import json
import os
import time
import urllib.request
import uuid

PORT = os.environ.get("PROBE_PORT", "8000")
URL = "http://127.0.0.1:%s/v1/chat/completions" % PORT
KEY = os.environ["PROBE_KEY"]
MODEL = os.environ["PROBE_MODEL"]
NT = int(os.environ.get("PROBE_TOKENS", "8000"))
MAXTOK = int(os.environ.get("PROBE_MAXTOK", "32"))

SENT = "The quick brown fox jumps over the lazy dog. "
need = int(NT * 4 / 1.2) + 128
filler = (SENT * (need // len(SENT) + 1))[:need]
nonce = uuid.uuid4().hex[:12]
prompt = "[nonce %s]\n%s\n\nReply with the single word: ok" % (nonce, filler)
MSG = [{"role": "user", "content": prompt}]


def send():
    body = json.dumps({
        "model": MODEL,
        "messages": MSG,
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + KEY,
    })
    t0 = time.time()
    ttft = None
    usage = {}
    chars = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except ValueError:
                continue
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    chars += len(delta["content"])
    return ttft, time.time() - t0, usage, chars


def fmt(x, unit="s"):
    return "n/a" if x is None else "%.2f %s" % (x, unit)


print("nonce = %s   target ≈ %d prompt tokens   max_tokens = %d" % (nonce, NT, MAXTOK))
print()
res = []
for tag in ("\u2460 \u51b7\u8dd1\uff08\u65e0\u53ef\u590d\u7528\u7f13\u5b58\uff09",
            "\u2461 \u91cd\u53d1\uff08\u5e94\u547d\u4e2d\u590d\u7528\uff09"):
    ttft, total, usage, chars = send()
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    res.append((pt, ttft, total))
    print(tag)
    print("    prompt_tokens = %s" % (pt or "n/a"))
    print("    TTFT          = %s" % fmt(ttft))
    print("    \u603b\u8017\u65f6        = %s   (\u8f93\u51fa %d tokens / %d \u5b57\u7b26)" % (fmt(total), ct, chars))
    print()

pt, t1, tot1 = res[0]
_, t2, tot2 = res[1]
print("\u2500\u2500 \u590d\u7528\u6536\u76ca \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
if not t1 or not t2:
    print("    \u26a0\ufe0f \u672a\u53d6\u5230 TTFT\uff0c\u68c0\u67e5\u670d\u52a1\u65e5\u5fd7")
else:
    saved = t1 - t2
    print("    \u51b7 prefill TTFT \u2248 %s" % fmt(t1))
    print("    \u590d\u7528\u540e TTFT   \u2248 %s" % fmt(t2))
    print("    \u7701\u4e0b         \u2248 %.2f s  (%.1f%%)" % (saved, saved / t1 * 100.0))
    if pt:
        print("    \u51b7\u8dd1\u6709\u6548 prefill \u901f\u5ea6 \u2248 %d tok/s" % (pt / t1))
PYEOF

docker cp "$PY" "$CONTAINER:/tmp/kv_reuse_probe.py" >/dev/null

docker exec \
  -e PROBE_KEY="$KEY" -e PROBE_MODEL="$MODEL" -e PROBE_PORT="$PORT" \
  -e PROBE_TOKENS="$PROBE_TOKENS" -e PROBE_MAXTOK="$PROBE_MAXTOK" \
  "$CONTAINER" python3 /tmp/kv_reuse_probe.py

rm -f "$PY"
