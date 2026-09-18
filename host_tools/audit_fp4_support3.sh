set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [F] MoE fp4 后端 oracle：emulation 在什么条件下被选中 ==="
grep -n -B6 -A10 "nvfp4_emulation_moe\|EmulationNvFp4\|emulation" "$V/model_executor/layers/fused_moe/oracle/nvfp4.py" | head -70

echo
echo "=== [G] 线性 emulation 内核的平台门 ==="
sed -n '1,60p' "$V/model_executor/kernels/linear/nvfp4/emulation.py"

echo
echo "=== [H] emulation MoE 内核的平台门 ==="
grep -n -B4 -A12 "def can_implement\|is_cuda\|is_rocm\|current_platform" "$V/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py" | head -50

echo
echo "=== [I] 线性 nvfp4 后端选择的完整优先级表 ==="
grep -n -B4 -A30 "def _get_backends\|BACKENDS\|priority" "$V/model_executor/kernels/linear/nvfp4/__init__.py" 2>/dev/null | head -60
ls "$V/model_executor/kernels/linear/nvfp4/"
