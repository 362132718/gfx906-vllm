set -u
echo "=== 容器信息 ==="
python3 -c "import sys; print('python', sys.version.split()[0])"
P=$(ls /usr/local/lib/python3*/dist-packages/vllm/platforms/rocm.py 2>/dev/null | head -1)
[ -z "$P" ] && P=$(ls /usr/lib/python3*/dist-packages/vllm/platforms/rocm.py 2>/dev/null | head -1)
echo "ROCM_PY=$P"
[ -z "$P" ] && { echo "!! rocm.py 未找到"; exit 1; }
echo "md5=$(md5sum "$P" | cut -d' ' -f1)"

echo
echo "=== [_ON_CDNA 定义与上下文] ==="
grep -n -B6 -A4 "_ON_CDNA" "$P"

echo
echo "=== [def on_cdna / on_gfx906 / supports_fp8 / is_fp8_fnuz] ==="
grep -n -B2 -A4 -E "def (on_cdna|on_gfx906|supports_fp8|is_fp8_fnuz|fp8_dtype|use_custom_allreduce|supports_mx)\b" "$P"

echo
echo "=== [on_cdna() 全引用点（rocm.py）] ==="
grep -n "on_cdna()" "$P"

echo
echo "=== [整个 vllm 包内的 on_cdna 引用] ==="
V=$(dirname "$P")/../..
grep -rn --include=*.py "on_cdna" "$V" | head -40

echo
echo "=== [supports_fp8 在包内被谁调用] ==="
grep -rn --include=*.py "supports_fp8()" "$V" | head -30

echo
echo "=== [gfx906 相关剔除写法（范本）] ==="
grep -n -B2 -A6 "flash_attn_triton_available" "$P"
