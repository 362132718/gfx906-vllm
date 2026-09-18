set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [A] is_fp4_marlin_supported() 的实现（marlin 是唯一非 SM100 的 fp4 矩阵路径）==="
sed -n '25,45p' "$V/model_executor/layers/quantization/utils/marlin_utils_fp4.py"

echo
echo "=== [B] fused_moe/utils.py 那条'blanket capability(100)'门 ==="
sed -n '585,625p' "$V/model_executor/layers/fused_moe/utils.py"

echo
echo "=== [C] ModelOptNvFp4Config 全文（看它怎么挑 kernel / 有没有非 CUDA 分支）==="
grep -n -A70 "^class ModelOptNvFp4Config" "$V/model_executor/layers/quantization/modelopt.py" | head -85

echo
echo "=== [D] nvfp4 emulation 路径（没有原生 fp4 时的软件模拟）要求什么 ==="
sed -n '1,40p' "$V/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
echo "--- emulation 的调用点与门 ---"
grep -rn --include=*.py "nvfp4_emulation\|emulate_nvfp4\|is_nvfp4_emulation" "$V" | head -15

echo
echo "=== [E] nvfp4 线性/MoE kernel 目录里有没有 ROCm 分支 ==="
ls "$V/model_executor/kernels/linear/nvfp4/" 2>/dev/null
grep -rn --include=*.py "is_rocm\|is_cuda" "$V/model_executor/kernels/linear/nvfp4/" 2>/dev/null | head -20
