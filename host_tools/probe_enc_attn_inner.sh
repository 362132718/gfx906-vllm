#!/bin/bash
# 在容器内运行：核实 encoder attention 的调度路径
SP=/usr/local/lib/python3.12/dist-packages
TA=$SP/vllm/v1/attention/backends/triton_attn.py
MM=$SP/vllm/model_executor/layers/attention/mm_encoder_attention.py

echo "=== [1] triton_attn.py 中 forward_encoder_attention 出现位置 ==="
grep -n forward_encoder_attention "$TA" || echo "(none)"

echo
echo "=== [2] 各出现点的上下文 ==="
for L in $(grep -n forward_encoder_attention "$TA" | cut -d: -f1); do
  sed -n "$((L-4)),$((L+14))p" "$TA"
  echo "--------"
done

echo
echo "=== [3] triton_attn.py 中 quantized 相关行 ==="
grep -n "is_quantized_kv_cache\|quantized KV\|NotImplementedError" "$TA" || echo "(none)"

echo
echo "=== [4] mm_encoder_attention.py 类与入口 ==="
grep -n "^class \|^    def \|^def " "$MM" || echo "(none)"
