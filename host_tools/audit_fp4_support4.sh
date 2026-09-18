set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [J] ModelOptNvFp4W4A16LinearMethod（W4A16_NVFP4 走的就是它）用哪个 kernel ==="
grep -n -A60 "^class ModelOptNvFp4W4A16LinearMethod" "$V/model_executor/layers/quantization/modelopt.py" | head -75

echo
echo "=== [K] 它的 kernel 选择 / is_supported 门 ==="
grep -n -B5 -A30 "NvFp4LinearKernel\|get_kernel\|MarlinNvFp4LinearKernel" "$V/model_executor/layers/quantization/modelopt.py" | head -80

echo
echo "=== [L] marlin nvfp4 kernel 的支持声明 ==="
grep -n -B3 -A20 "class MarlinNvFp4LinearKernel" "$V/model_executor/kernels/linear/nvfp4/marlin.py" | head -40

echo
echo "=== [M] select_nvfp4_moe_backend：没有任何 CUDA 后端时会不会挑 EMULATION ==="
grep -n -A45 "^def select_nvfp4_moe_backend" "$V/model_executor/layers/fused_moe/oracle/nvfp4.py" | head -60
