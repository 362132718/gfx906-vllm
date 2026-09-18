#!/bin/bash
# 长上下文冷跑 v2 —— 修掉 v1 的三个缺陷：
#   ① 每请求注入唯一 nonce ⇒ 保证真冷跑（不吃前缀缓存，见项目纪律）
#   ② 用 stream_options.include_usage 从最终的 usage 块读**精确** prompt_tokens
#      （v1 走 /tokenize 两次都 404，token 数只能靠 filler 段数估算）
#   ③ max_tokens 提到 256（v1 的 48 会截断模型的推理段 ⇒ 误判「未检索到针」）
# 目的：① 补齐 64K 档的检索正确性验证；② 给 prefill 单价曲线加一个 32K 数据点。
K=$(cat /home/x/.vllm-api-key)
URL=http://localhost:8000/v1
AUTH="Authorization: Bearer $K"
CT="Content-Type: application/json"

build() {  # $1=目标约 token 数  $2=名字
python3 - "$1" "$2" <<'PY'
import json, sys, uuid
target, name = int(sys.argv[1]), sys.argv[2]
F = "这是用于测试长上下文能力的中性填充文本，内容本身没有任何含义，只用于凑长度。"
CODE = "QX-8842"
NONCE = uuid.uuid4().hex[:12]
per = 25.35                      # v1 实测每段约 25.35 token（仅供参考，真值由 usage 给出）
n = max(2, int(target / per))
half = n // 2
body = (f"【会话ID {NONCE}】\n"
        + F * half
        + f"\n【关键信息】本次的取件码是 {CODE}，请务必记住。\n"
        + F * (n - half)
        + "\n\n取件码是多少？只回答取件码本身，不要任何其他文字。")
obj = {"model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
       "messages": [{"role": "user", "content": body}],
       "max_tokens": 256, "temperature": 0, "stream": True,
       "stream_options": {"include_usage": True}}
json.dump(obj, open(f"/home/x/lc2_{name}.json", "w"), ensure_ascii=False)
print(f"  built lc2_{name}.json (nonce={NONCE}, filler x{n})")
PY
}

send() {  # $1=json  $2=标签
  echo "--- $2 ---"
  curl -N -s -o /tmp/lc2_stream.txt \
    -w '  HTTP=%{http_code} TOTAL=%{time_total}s\n' \
    --max-time 2400 -H "$CT" -H "$AUTH" "$URL/chat/completions" -d @"$1"
  python3 - <<'PY'
import json
text, ptok, ctok = [], None, None
for line in open('/tmp/lc2_stream.txt', errors='ignore'):
    line = line.strip()
    if not line.startswith('data: '):
        continue
    d = line[6:]
    if d == '[DONE]':
        break
    try:
        j = json.loads(d)
    except Exception:
        continue
    u = j.get('usage')
    if u:
        ptok = u.get('prompt_tokens', ptok)
        ctok = u.get('completion_tokens', ctok)
    c = ((j.get('choices') or [{}])[0].get('delta') or {}).get('content')
    if c:
        text.append(c)
a = ''.join(text).strip().replace('\n', ' ')
print(f"  精确 prompt_tokens = {ptok}   completion_tokens = {ctok}")
print("  回答:", a[:300] if a else "(空)")
print("  含取件码:", "QX-8842" in a)
PY
}

check_oom() {
  echo "  --- 日志 OOM/异常检查（累计值，基线为 0）---"
  docker exec vllm grep -icE "out of memory|OutOfMemory|HIP error|CUDA error|Traceback" /logs/serve.log 2>/dev/null \
    | sed 's/^/  匹配行数: /'
}

echo "=== 长上下文冷跑 v2（唯一 nonce + include_usage + max_tokens=256）==="
for spec in "32000:32k" "64000:64k"; do
  t="${spec%%:*}"; name="${spec##*:}"
  build "$t" "$name"
  send "/home/x/lc2_$name.json" "${name} 冷跑"
  check_oom
  echo
done
echo "=== 服务存活 ==="
curl -s -o /dev/null -w "  /v1/models HTTP=%{http_code}\n" -H "$AUTH" http://localhost:8000/v1/models
