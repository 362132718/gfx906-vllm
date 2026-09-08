#!/bin/bash
# prepare_build_context.sh
# Gathers everything needed to rebuild the gfx906 vLLM environment into
# ./build-context/, extracting the custom pieces from the RUNNING working
# container (vllm-gfx906) plus host files. Run on the GPU host (192.168.10.11).
#
# Usage:  bash prepare_build_context.sh [container_name]
set -euo pipefail

CONTAINER="${1:-vllm-gfx906}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="$HERE/build-context"
SP="/usr/local/lib/python3.12/dist-packages"

# Host-side sources (adjust if your layout differs)
RCCL_SRC="${RCCL_SRC:-/home/x/rccl-gfx906-2.27.7-rocm7.2.4/librccl.so.1.0}"
SUPPLEMENT_SRC="${SUPPLEMENT_SRC:-/home/x/backup-rocm-tensors}"

echo "== container: $CONTAINER"
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || {
  echo "container '$CONTAINER' is not running; start it first"; exit 1; }

mkdir -p "$CTX/rccl" "$CTX/supplements" "$CTX/patches" "$CTX/wheels"

echo "== [1/7] gfx906 RCCL (torch/lib replacement)"
if [ -f "$RCCL_SRC" ]; then
  cp -f "$RCCL_SRC" "$CTX/rccl/librccl.so.1.0"
else
  docker cp "$CONTAINER:$SP/torch/lib/librccl.so.1.0" "$CTX/rccl/librccl.so.1.0"
fi
sha256sum "$CTX/rccl/librccl.so.1.0"

echo "== [2/7] Tensile gfx906 supplement files"
if [ -d "$SUPPLEMENT_SRC" ]; then
  cp -rf "$SUPPLEMENT_SRC/." "$CTX/supplements/"
else
  docker cp "$CONTAINER:/work/gfx906-supplements/." "$CTX/supplements/"
fi
echo "   $(ls "$CTX/supplements" | wc -l) files"

echo "== [3/7] patched vLLM python files"
docker cp "$CONTAINER:$SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py" \
  "$CTX/patches/wna16_exllama.py"
docker cp "$CONTAINER:$SP/vllm/model_executor/kernels/linear/__init__.py" \
  "$CTX/patches/linear__init__.py"
docker cp "$CONTAINER:$SP/vllm/v1/attention/ops/triton_decode_attention.py" \
  "$CTX/patches/triton_decode_attention.py"
# P0.5: gfx906 M=1 dense GEMV -> ROCm skinny LLGemm1_kernel dispatch
docker cp "$CONTAINER:$SP/vllm/model_executor/layers/utils.py" \
  "$CTX/patches/layers_utils.py"
docker cp "$CONTAINER:$SP/gfx906_gemv_fix.py" "$CTX/patches/gfx906_gemv_fix.py"
docker cp "$CONTAINER:$SP/gfx906_gemv_fix.pth" "$CTX/patches/gfx906_gemv_fix.pth"

echo "== [4/7] delivery entry points (gfx906-serve / gfx906-selfcheck / QUICKSTART)"
rm -rf "$CTX/entrypoint"
docker cp "$CONTAINER:/opt/gfx906/." "$CTX/entrypoint/"
ls "$CTX/entrypoint"

echo "== [5/7] record installed versions"
docker exec "$CONTAINER" bash -lc '
  for p in torch torchvision triton transformers vllm; do
    ls -d '$SP'/${p}-*.dist-info 2>/dev/null | head -1
  done' > "$CTX/versions.txt"
cat "$CTX/versions.txt"

echo "== [6/7] triton_kernels source dir (for TRITON_KERNELS_SRC_DIR)"
if docker exec "$CONTAINER" test -d /work/triton_kernels_pkg; then
  docker cp "$CONTAINER:/work/triton_kernels_pkg" "$CTX/triton_kernels_pkg"
  echo "   copied triton_kernels_pkg"
else
  echo "   (none found; TRITON_KERNELS_SRC_DIR will be unset)"
fi

echo "== [7/7] full python environment tarball (for Dockerfile 'rebuild' target)"
# Contains dist-packages (torch w/ gfx906 RCCL + Tensile supplements, triton,
# vllm w/ patches, transformers, ...) plus pip entrypoints in /usr/local/bin.
# ~18G uncompressed, several minutes. Skipped if already present (pass FORCE=1).
if [ -f "$CTX/dist-packages.tar.gz" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "   already present ($(du -h "$CTX/dist-packages.tar.gz" | cut -f1)); use FORCE=1 to redo"
else
  docker exec "$CONTAINER" tar -czf /tmp/dist-packages.tar.gz \
      -C / usr/local/lib/python3.12/dist-packages usr/local/bin
  docker cp "$CONTAINER:/tmp/dist-packages.tar.gz" "$CTX/dist-packages.tar.gz"
  docker exec "$CONTAINER" rm -f /tmp/dist-packages.tar.gz
  echo "   $(du -h "$CTX/dist-packages.tar.gz" | cut -f1)"
fi

echo
echo "Build context ready at: $CTX"
du -sh "$CTX"
