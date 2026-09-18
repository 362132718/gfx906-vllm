#!/bin/bash
# 长上下文实跑：64K / 128K 各发一次**冷跑**（唯一针 ⇒ 必不命中缓存），
# 测真实 TTFT（流式 time_starttransfer）、验证针能被检索到、并检查是否 OOM。
K=$(cat /home/x/.vllm-api-key)
URL=http://localhost:8000/v1
AUTH="Authorization: Bearer $K"
CT="Content-Type: application/json"

build() {  # $1=目标大约 token 数  $2=名字
python3 - "$1" "$2" <<'PY'
import json, sys
target, name = int(sys.argv[1]), sys.argv[2]
F = "这是用于测试长上下文能力的中性填充文本，内容本身没有任何含义，只用于凑长度。"
CODE = "QX-8842"
per = 25.35                      # 实测每段约 25.35 token
n = int(target / per)
half = n // 2
body = (F * half
        + f"\n【关键信息】本次的取件码是 {CODE}，请务必记住。\n"
        + F * (n - half)
        + "\n\n取件码是多少？只回答取件码本身，不要任何其他文字。")
obj = {"model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
       "messages": [{"role": "user", "content": body}],
       "max_tokens": 48, "temperature": 0, "stream": True}
json.dump(obj, open(f"/home/x/lc_{name}.json", "w"), ensure_ascii=False)
print(f"  built lc_{name}.json (filler x{n})")
PY
}

actual_tokens() {  # 用 /tokenize 精确测
python3 - "$1" <<'PY'
import json, sys, urllib.request
body = json.load(open(sys.argv[1]))
txt = body["messages"][0]["content"]
req = urllib.request.Request("http://localhost:8000/v1/tokenize",
        data=json.dumps({"model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16", "prompt": txt}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + open("/home/x/.vllm-api-key").read().strip()})
print(json.loads(urllib.request.urlopen(req).read())["count"])
PY
}

send() {  # $1=json  $2=标签
  echo "--- $2 ---"
  code=$(curl -N -s -o /tmp/lc_stream.txt -w 'HTTP=%{http_code} TTFB=%{time_starttransfer}s TOTAL=%{time_total}s' \
          --max-time 1800 -H "$CT" -H "$AUTH" "$URL/chat/completions" -d @"$1")
  echo "  $code"
  python3 - <<'PY'
import json
out = []
for line in open('/tmp/lc_stream.txt', errors='ignore'):
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
    ch = j.get('choices') or [{}]
    c = (ch[0].get('delta') or {}).get('content')
    if c:
        out.append(c)
a = ''.join(out).strip().replace('\n', ' ')
print("  回答:", a[:220] if a else "(空)")
print("  含取件码:", "QX-8842" in a)
PY
}

check_oom() {
  echo "  --- 日志 OOM/异常检查 ---"
  docker exec vllm grep -icE "out of memory|OutOfMemory|HIP error|CUDA error|Traceback" /logs/serve.log 2>/dev/null \
    | sed 's/^/  匹配行数: /'
}

echo "=== 长上下文实跑 ==="
build 64000  "64k"
echo "  实际 tokens = $(actual_tokens /home/x/lc_64k.json)"
send /home/x/lc_64k.json "64K 冷跑"
check_oom
echo
build 125000 "125k"
echo "  实际 tokens = $(actual_tokens /home/x/lc_125k.json)"
send /home/x/lc_125k.json "125K 冷跑"
check_oom
echo
echo "=== 服务存活 ==="
curl -s -o /dev/null -w "  /v1/models HTTP=%{http_code}\n" -H "$AUTH" http://localhost:8000/v1/models
