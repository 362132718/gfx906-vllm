set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [has_device_capability(89) 全包分布 —— 这就是所有 fp8 门禁] ==="
grep -rn --include=*.py "has_device_capability(89)" "$V"

echo
echo "=== [has_device_capability(80) 全包分布 —— bf16 同族门禁（只记录，本次不改）] ==="
grep -rn --include=*.py "has_device_capability(80)" "$V"

echo
echo "=== [_is_supported_kv_cache_dtype 调用点（确认它在热路径上）] ==="
grep -rn --include=*.py "_is_supported_kv_cache_dtype" "$V" -A6 | head -40

echo
echo "=== [triton_unified_attention.py 里的 fp8 / capability 判定（decode 路径）] ==="
grep -n -E "has_device_capability|supports_fp8|_is_supported|kv_cache_dtype" "$V/v1/attention/ops/triton_unified_attention.py" | head -30

echo
echo "=== [CacheDType 字面量全集（确认 startswith('fp8') 的覆盖面）] ==="
grep -n -A22 "^CacheDType = " "$V/config/cache.py" | head -30

echo
echo "=== [has_device_capability 在 rocm.py 的实现 —— 看 (9,0)>=(8,9) 是怎么成立的] ==="
grep -rn -B4 -A18 "def has_device_capability" "$V/platforms/rocm.py" "$V/platforms/interface.py" | head -50
