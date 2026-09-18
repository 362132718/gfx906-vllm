#!/bin/bash
# 在容器内运行：补全 triton_attn.py 其余门禁 + 确认 MM 编码器后端
SP=/usr/local/lib/python3.12/dist-packages
TA=$SP/vllm/v1/attention/backends/triton_attn.py
MM=$SP/vllm/model_executor/layers/attention/mm_encoder_attention.py

echo "=== [A] triton_attn.py L624-640（另一处 NotImplementedError）==="
sed -n '624,640p' "$TA"

echo
echo "=== [B] triton_attn.py L640-700（attn_type 分派 + per-token-head 分支）==="
sed -n '640,700p' "$TA"

echo
echo "=== [C] triton_attn.py L840,860 ==="
sed -n '840,860p' "$TA"

echo
echo "=== [D] MMEncoderAttention.forward_cuda / _forward_triton 选路 ==="
sed -n '786,870p' "$MM"

echo
echo "=== [E] 日志里与编码器后端有关的行 ==="
grep -nE "encoder|Encoder|mm_encoder|MMEncoder" /logs/serve.log | tail -20
