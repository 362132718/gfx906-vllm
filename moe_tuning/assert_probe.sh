#!/usr/bin/env bash
# assert_probe.sh -- replicate the Dockerfile Path B assertion commands one by
# one inside a container, to find which single one fails. The build reports only
# "exit code: 1" for the whole chained RUN, which is useless for diagnosis.
#
# Run inside the container that supplied the tarball.
SP=/usr/local/lib/python3.12/dist-packages
FAILS=0
t() {  # t <label> <shell command>
  if eval "$2" >/dev/null 2>&1; then
    printf '  PASS  %s\n' "$1"
  else
    printf '  FAIL  %s\n        cmd: %s\n' "$1" "$2"
    FAILS=$((FAILS + 1))
  fi
}

echo "=== Dockerfile Path B 断言逐条复现 ==="
t "librccl.so.1.0 exists"        "test -f $SP/torch/lib/librccl.so.1.0"
t "librccl.so is a symlink"      "test -L $SP/torch/lib/librccl.so"
t "tensile has gfx906 kernels"   "ls $SP/torch/lib/rocblas/library/ | grep -q gfx906"
t "wna16_exllama.py exists"      "test -f $SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py"
t "WNA16ExllamaKernel registered" "grep -q WNA16ExllamaKernel $SP/vllm/model_executor/kernels/linear/__init__.py"
t "gptq_gemm_out in bridge"      "grep -q gptq_gemm_out $SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py"
t "on_gfx906 present"            "grep -q 'on_gfx906' $SP/vllm/v1/attention/ops/triton_decode_attention.py"
t "is_gfx906 absent"             "! grep -q 'is_gfx906' $SP/vllm/v1/attention/ops/triton_decode_attention.py"
t "use_gfx906_gemv present"      "grep -q use_gfx906_gemv $SP/vllm/model_executor/layers/utils.py"
t "NEW: m==1 broadcast branch"   "grep -q 'acc = (x_view \* weight).sum' $SP/vllm/model_executor/layers/utils.py"
t "NEW: MoE tuned config file"   "test -f '${SP}/vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json'"
t "libp2gemv.so non-empty"       "test -s $SP/gfx906/libp2gemv.so"
t "p2_gemv symbol exported"      "nm -D --defined-only $SP/gfx906/libp2gemv.so | grep -qw p2_gemv"
t "_P2_SHAPES present"           "grep -q _P2_SHAPES $SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py"
t "gemv_fix module"              "test -f $SP/gfx906_gemv_fix.py"
t "gemv_fix .pth"                "test -f $SP/gfx906_gemv_fix.pth"
t "gptq_gemm_out in .so"         "grep -aq gptq_gemm_out $SP/vllm/_C_stable_libtorch.abi3.so"
t "serve.sh executable"          "test -x /opt/gfx906/serve.sh"
t "gfx906-serve symlink"         "test -L /usr/local/bin/gfx906-serve"
t "QUICKSTART.md present"        "test -f /opt/gfx906/QUICKSTART.md"
t "version import"               "python3 -c \"from importlib.metadata import version as v; print(v('torch'), v('vllm'))\""

echo
echo "FAILS=$FAILS"
echo "--- 直接看两个可疑对象 ---"
ls -la "$SP/vllm/model_executor/layers/fused_moe/configs/" 2>/dev/null | grep -i radeon || echo "  (无 Radeon_Pro_VII 配置)"
grep -n "acc = (x_view" "$SP/vllm/model_executor/layers/utils.py" 2>/dev/null | head -3
echo "--- 用单引号模式再试一次 grep ---"
grep -q 'acc = (x_view \* weight).sum' "$SP/vllm/model_executor/layers/utils.py" && echo "  single-quote pattern: MATCH" || echo "  single-quote pattern: NO MATCH"
grep -qF 'acc = (x_view * weight).sum' "$SP/vllm/model_executor/layers/utils.py" && echo "  grep -F literal: MATCH" || echo "  grep -F literal: NO MATCH"
