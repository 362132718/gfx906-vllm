set -u
P=/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [rocm.py 420-440: on_cdna() @429 上下文] ==="
sed -n '415,445p' "$P" | cat -n

echo
echo "=== [rocm.py 770-800: on_cdna() @779/784 上下文] ==="
sed -n '765,802p' "$P" | cat -n

echo
echo "=== [平台层 supported_kv_cache_dtypes 是否存在] ==="
grep -rn --include=*.py "supported_kv_cache_dtypes" "$V" | head -30

echo
echo "=== [谁在校验 kv_cache_dtype / fp8 与平台的组合] ==="
grep -rn --include=*.py -E "kv_cache_dtype.*(fp8|not supported|unsupported)|(fp8).*kv_cache_dtype" "$V" | grep -v "tests/" | head -40

echo
echo "=== [config/cache.py 里的 dtype 校验] ==="
grep -n -E "def |kv_cache_dtype|fp8|raise" "$V/config/cache.py" | head -60

echo
echo "=== [supports_fp8 定义在基类/其他平台] ==="
grep -rn --include=*.py -B2 -A4 "def supports_fp8" "$V/platforms" | head -60

echo
echo "=== [TritonAttentionBackend.supported_kv_cache_dtypes] ==="
grep -rn -B3 -A8 "supported_kv_cache_dtypes" "$V/v1/attention/backends/triton_attn.py" | head -40

echo
echo "=== [triton_attn.py 里那段 SM89 校验] ==="
grep -n -B6 -A10 "SM89\|sm89\|is_cuda()" "$V/v1/attention/backends/triton_attn.py" | head -60
