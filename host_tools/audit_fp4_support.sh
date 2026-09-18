set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [1] ROCm 平台 supported_quantization 列表 ==="
grep -n -A45 "supported_quantization" "$V/platforms/rocm.py" | head -55

echo
echo "=== [2] 量化目录里有哪些 nvfp4/fp4 相关实现 ==="
grep -rln --include=*.py "nvfp4\|fp4" "$V/model_executor/layers/quantization/" | sort

echo
echo "=== [3] modelopt nvfp4 的支持门（is_supported / capability） ==="
grep -n -B4 -A18 "class ModelOptNvFp4Config\|def is_supported" "$V/model_executor/layers/quantization/modelopt.py" | head -80

echo
echo "=== [4] 全包对 FP4 矩阵内核的硬件要求（SM100/Blackwell/capability 100） ==="
grep -rn --include=*.py "has_device_capability(100)\|is_blackwell\|device_capability(120)\|SM100" "$V/model_executor" "$V/platforms" | head -25

echo
echo "=== [5] fp4 的 kernel 选择表（看有没有非 CUDA 分支） ==="
grep -rn -B3 -A15 "def _get_fp4_kernel\|get_fp4_gemm\|fp4_gemm" "$V/model_executor/layers/quantization/utils/" 2>/dev/null | head -50

echo
echo "=== [6] 有没有 marlin fp4 / triton fp4 的可用性判断 ==="
grep -rn --include=*.py "marlin.*fp4\|fp4.*marlin\|nvfp4" "$V/model_executor/layers/quantization/utils/marlin_utils_fp4.py" 2>/dev/null | head -20
grep -rn --include=*.py "def is_fp4_marlin_supported\|def is_nvfp4" "$V" | head -10

echo
echo "=== [7] 平台是否声明支持 nvfp4（platforms 层） ==="
grep -rn --include=*.py "nvfp4" "$V/platforms/" | head -20
