#!/bin/bash
SP=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [A] config/vllm.py::_verify_kv_transfer_compat（标题就是「reject silently corrupt KV transfers」）==="
awk '/def _verify_kv_transfer_compat/,/^    def [a-z_]+\(self.*\) -> None:$/' "$SP/config/vllm.py" | head -60

echo
echo "=== [B] 该函数的全部 raise 文案 ==="
grep -n -A6 "raise ValueError" "$SP/config/vllm.py" | sed -n '1,80p' | grep -n -iE "quant|offload|scale|prefix" 

echo
echo "=== [C] prefix_match_unit 的语义文档（cache.py） ==="
sed -n '50,70p' "$SP/config/cache.py"

echo
echo "=== [D] kv_cache_utils.py 里 prefix_match_unit 的解析逻辑 ==="
sed -n '612,670p' "$SP/v1/core/kv_cache_utils.py"
