#!/bin/bash
# 前缀缓存命中率探针 v2
#   用法: bash prefix_hit_probe.sh [标签]
#   对每个长度发两次**完全相同**的 prompt，比较第二次的命中率。
#   关键一档是「刚好卡在块大小下方」的长度 —— 它最能暴露 1552 粒度悬崖。
K=$(cat /home/x/.vllm-api-key)
URL=http://localhost:8000/v1
AUTH="Authorization: Bearer $K"
CT="Content-Type: application/json"
MURL=http://localhost:8000/metrics
LABEL="${1:-默认}"

snap() {
  curl -s -H "$AUTH" "$MURL" | awk '
    /^vllm:prefix_cache_queries_total/ {q=$2}
    /^vllm:prefix_cache_hits_total/    {h=$2}
    END{printf "%d %d", q+0, h+0}'
}

ask() {  # $1=json 文件  $2=标签
  local q0 h0 q1 h1 dq dh pct t0 t1 code
  read -r q0 h0 <<<"$(snap)"
  t0=$(date +%s%N)
  code=$(curl -s -o /tmp/r_$$.json -w '%{http_code}' --max-time 900 \
         -H "$CT" -H "$AUTH" "$URL/chat/completions" -d @"$1")
  t1=$(date +%s%N)
  read -r q1 h1 <<<"$(snap)"
  dq=$((q1 - q0)); dh=$((h1 - h0))
  pct=$(awk -v a="$dh" -v b="$dq" 'BEGIN{printf "%.1f", (b>0 ? 100*a/b : 0)}')
  printf "  %-26s http=%s  %5.2fs  queries=+%-6d hits=+%-6d 命中率=%s%%\n" \
     "$2" "$code" "$(awk -v x="$t0" -v y="$t1" 'BEGIN{print (y-x)/1e9}')" "$dq" "$dh" "$pct"
}

tok() { curl -s -H "$CT" -H "$AUTH" "$URL/tokenize" \
        -d "{\"model\":\"Qwen3.8-27B-OBLITERATED-AWQ-W4A16\",\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()))' "$1")}" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("count"))'; }

python3 - <<'PY'
import json
F = "这是用于测试前缀缓存命中率的中性填充文本，内容本身没有任何含义，只用于凑长度。"
SPEC = [("A_tiny", 8,  "A"), ("B_just_under_blk", 48, "B"), ("C_multi_blk", 200, "C")]
for name, n, tag in SPEC:
    body = F * n + f"\n（标记 {tag}）\n请只回答一个词：收到"
    obj = {"model": "Qwen3.8-27B-OBLITERATED-AWQ-W4A16",
           "messages": [{"role": "user", "content": body}],
           "max_tokens": 8, "temperature": 0}
    json.dump(obj, open(f"/home/x/ph_{name}_1.json", "w"), ensure_ascii=False)
    json.dump(obj, open(f"/home/x/ph_{name}_2.json", "w"), ensure_ascii=False)
print("[probe] prompts built")
PY

echo "=== 配置: $LABEL ==="
grep -oE 'kv_cache_dtype=[a-z0-9_]+|max_seq_len=[0-9]+|prefix_match_unit=[0-9]+' /home/x/logs/serve.log | head -3
echo
for name in A_tiny B_just_under_blk C_multi_blk; do
  echo "[$name] prompt tokens = $(tok /home/x/ph_${name}_1.json)"
  ask /home/x/ph_${name}_1.json "${name} 第1次(冷)"
  ask /home/x/ph_${name}_2.json "${name} 第2次(同前缀)"
  echo
done
