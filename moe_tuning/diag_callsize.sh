#!/usr/bin/env bash
echo "=== torch/_dynamo/utils.py call_size 实现 ==="
grep -n "def call_size" -A 18 /usr/local/lib/python3.12/dist-packages/torch/_dynamo/utils.py
echo
echo "=== torch/_compile.py 40-60 ==="
sed -n '40,60p' /usr/local/lib/python3.12/dist-packages/torch/_compile.py
echo
echo "=== qwen3_next.py 615-625（精确行号）==="
sed -n '615,625p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py
echo
echo "=== qwen3_5.py 565-580 ==="
sed -n '565,580p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py
