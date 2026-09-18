#!/usr/bin/env bash
# 只读：能否仅凭「镜像」重建（prepare_build_context.sh 的容器回退路径是否可用）
IMG=vllm-gfx906:latest
SP=/usr/local/lib/python3.12/dist-packages

echo "=== 1) 镜像里 /work/ 有没有 prepare_build_context.sh 依赖的东西 ==="
docker run --rm --entrypoint bash "$IMG" -lc '
  echo "--- /work ---"; ls -la /work/ 2>/dev/null || echo "(无 /work 内容)"
  echo "--- /work/gfx906-supplements 文件数（[2/7] 的容器回退）---"
  if [ -d /work/gfx906-supplements ]; then find /work/gfx906-supplements -type f | wc -l; du -sh /work/gfx906-supplements;
  else echo "MISS /work/gfx906-supplements"; fi
  echo "--- /work/triton_kernels_pkg（[6/7]）---"
  if [ -d /work/triton_kernels_pkg ]; then find /work/triton_kernels_pkg -type f | wc -l;
  else echo "MISS /work/triton_kernels_pkg"; fi
'

echo
echo "=== 2) 三个「只在容器/镜像里」的补丁文件在镜像中的状态 ==="
docker run --rm --entrypoint bash "$IMG" -lc "
  for f in \
    $SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py \
    $SP/vllm/model_executor/kernels/linear/__init__.py \
    $SP/vllm/v1/attention/ops/triton_decode_attention.py \
    $SP/vllm/model_executor/layers/utils.py \
    $SP/gfx906_gemv_fix.py $SP/gfx906_gemv_fix.pth \
    $SP/gfx906/libp2gemv.so ; do
    if [ -f \"\$f\" ]; then printf '  OK   %8s  %s\n' \"\$(stat -c%s \$f)\" \"\$f\"; else echo \"  MISS \$f\"; fi
  done
  echo '--- 校验标记 ---'
  echo -n '  WNA16ExllamaKernel in linear/__init__.py: '; grep -c WNA16ExllamaKernel $SP/vllm/model_executor/kernels/linear/__init__.py
  echo -n '  gptq_gemm_out in wna16_exllama.py       : '; grep -c gptq_gemm_out $SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py
  echo -n '  on_gfx906 in triton_decode_attention.py : '; grep -c on_gfx906 $SP/vllm/v1/attention/ops/triton_decode_attention.py
  echo -n '  use_gfx906_gemv in layers/utils.py      : '; grep -c use_gfx906_gemv $SP/vllm/model_executor/layers/utils.py
  echo -n '  Tensile gfx906 files in torch rocblas   : '; ls $SP/torch/lib/rocblas/library/ | grep -c gfx906
  echo -n '  librccl.so.1.0 size                     : '; stat -c%s $SP/torch/lib/librccl.so.1.0
"

echo
echo "=== 3) 基座镜像是否公开可得（Dockerfile 的 FROM）==="
docker run --rm --entrypoint bash "$IMG" -lc "cat /etc/os-release | head -2"
echo "FROM 行："; grep -m1 '^ARG BASE_IMAGE' /dev/null 2>/dev/null || echo "  (见 Dockerfile: ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4-complete)"

echo
echo "=== 4) 镜像体积与可导出的 tar 预估 ==="
docker image inspect "$IMG" --format 'Image size: {{.Size}} bytes'
df -h /home | tail -1

echo
echo "=== 5) 结论所需的最后一项：/opt/gfx906 三个交付件 ==="
docker run --rm --entrypoint bash "$IMG" -lc 'ls -l /opt/gfx906/; echo "--- QUICKSTART 头部 ---"; head -12 /opt/gfx906/QUICKSTART.md'
