#!/bin/bash
# 前缀缓存「命中后输出是否正确」验证
#   针（needle）分别放在**中段**和**末段**，看冷跑与命中跑的答案是否一致且正确。
#   若 mid-block 复用的 mamba 状态有问题，答案会退化成流利但错误的内容。
K=$(cat /home/x/.vllm-api-key)
URL=http://localhost:8000/v1
AUTH="Authorization: Bearer $K"
CT="Content-Type: application/json"
TAG="${1:-run}"

python3 - <<'PY'
import json
F = "这是用于测试前缀缓存命中率的中性填充文本，内容本身没有任何含义，只用于凑长度。"
NEEDLE_MID = "\n【重要】本次的取件码是 K7-3312，请务必记住。\n"
NEEDLE_END = "\n【重要】本次的取件码是 K7-3312，请务必记住。\n"
Q = "\n\n取件码是多少？只回答取件码本身，不要任何其他文字。"

cases = {
    # (填充段数, 针位置)
    "mid_1121": (48, "mid"),
    "end_1121": (48, "end"),
    "mid_4465": (200, "mid"),
    "end_4465": (200, "end"),
}
for name, (n, pos) in cases.items():
    half = n // 2
    if pos == "mid":
        body = F * half + NEEDLE_MID + F * (n - half)
    else:
        body = F * n + NEEDLE_END
    body += Q
    obj = {"model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
           "messages": [{"role": "user", "content": body}],
           "max_tokens": 24, "temperature": 0}
    for i in (1, 2):
        json.dump(obj, open(f"/home/x/pv_{name}_{i}.json", "w"), ensure_ascii=False)
    print(f"  built {name}: filler x{n}, needle={pos}")
PY

answer() {  # $1=json, $2=label
  local t0 t1
  t0=$(date +%s%N)
  curl -s -o /tmp/pv.json -w '' --max-time 900 -H "$CT" -H "$AUTH" \
       "$URL/chat/completions" -d @"$1"
  t1=$(date +%s%N)
  local a
  a=$(python3 -c 'import json;print(json.load(open("/tmp/pv.json"))["choices"][0]["message"]["content"].strip().replace("\n"," "))')
  printf "    %-22s %5.2fs  → %s\n" "$2" \
     "$(awk -v x="$t0" -v y="$t1" 'BEGIN{print (y-x)/1e9}')" "$a"
}

echo "=== 冷/热一致性验证 [$TAG] ==="
for name in mid_1121 end_1121 mid_4465 end_4465; do
  echo "[$name]"
  answer /home/x/pv_${name}_1.json "第1次(冷)"
  answer /home/x/pv_${name}_2.json "第2次(命中缓存)"
done
