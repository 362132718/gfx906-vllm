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

echo "== [2/7] Tensile gfx906 supplement files (expect 156)"
# Two sources, both verified equivalent on 2026-09-18 for vllm-gfx906:latest:
#   A. a host directory holding the 156 files (SUPPLEMENT_SRC, default
#      /home/x/backup-rocm-tensors).
#   B. the container's own torch rocblas library. The image installs the same
#      156 files there (Dockerfile: COPY supplements/ -> torch/lib/rocblas/
#      library/); compared byte-for-byte -- 156/156 names, sizes and md5 match.
# B used to point at /work/gfx906-supplements, which only ever existed in the
# dev container and is NOT in the delivery image, so the fallback silently
# produced an empty supplements/ and Path B then built a rocBLAS without any
# gfx906 kernels. Keep B pointed at the rocblas library instead.
if [ -d "$SUPPLEMENT_SRC" ]; then
  cp -rf "$SUPPLEMENT_SRC/." "$CTX/supplements/"
else
  echo "   (no host dir at $SUPPLEMENT_SRC; extracting from the container's rocblas library)"
  rm -rf "$CTX/supplements"; mkdir -p "$CTX/supplements"
  docker exec "$CONTAINER" bash -lc \
    "cd $SP/torch/lib/rocblas/library && ls | grep gfx906 | tar -cf /tmp/gfx906_supp.tar -T -"
  docker cp "$CONTAINER:/tmp/gfx906_supp.tar" "$CTX/_supp.tar"
  tar -xf "$CTX/_supp.tar" -C "$CTX/supplements" && rm -f "$CTX/_supp.tar"
  docker exec "$CONTAINER" rm -f /tmp/gfx906_supp.tar
fi
n_supp=$(ls "$CTX/supplements" | wc -l)
echo "   $n_supp files"
if [ "$n_supp" -ne 156 ]; then
  echo "ERROR: expected 156 Tensile gfx906 supplements, got $n_supp." >&2
  echo "       Path B would produce a rocBLAS with no gfx906 kernels." >&2
  exit 1
fi
ls "$CTX/supplements" | grep -q 'TensileLibrary_lazy_gfx906.dat' || {
  echo "ERROR: supplements/ lacks TensileLibrary_lazy_gfx906.dat" >&2; exit 1; }

echo "== [3/7] patched vLLM python files"
# Rule for this whole group (same one P0.6 and the P2 .hip already follow):
# a file that the Dockerfile COPYs into the delivery image must come from the
# REPO, never from whatever the dev container happens to carry. $CTX/patches is
# where the Dockerfile reads from, so a container-sourced copy would let an
# unreviewed build shadow the audited one. The container is only used for files
# whose repo copy does not exist.
#
# Pinned in docker/patches/ (md5 recorded 2026-09-18, extracted from
# vllm-gfx906:latest and verified byte-identical to container_src/):
#   wna16_exllama.py            0a146c02eaf3e696  (W4A16 -> gptq_gemm ExLlama bridge + P2 M=1 route)
#   linear__init__.py           42e243693b55fa2b  (kernel registration; the ONLY copy that
#                                                  ever existed was inside the image)
#   triton_decode_attention.py  a5a0ac3f2a64fd55  (BLOCK=32 on gfx906)
#   gfx906_gemv_fix.py / .pth   ef07381791dd48a2 / 722a18dab978dbc5  (rocBLAS GEMV segfault workaround)
cp -f "$HERE/patches/wna16_exllama.py"           "$CTX/patches/wna16_exllama.py"
cp -f "$HERE/patches/linear__init__.py"          "$CTX/patches/linear__init__.py"
cp -f "$HERE/patches/triton_decode_attention.py" "$CTX/patches/triton_decode_attention.py"
cp -f "$HERE/patches/gfx906_gemv_fix.py"         "$CTX/patches/gfx906_gemv_fix.py"
cp -f "$HERE/patches/gfx906_gemv_fix.pth"        "$CTX/patches/gfx906_gemv_fix.pth"
# layers_utils.py and the two repo-only payloads below.
# P0.5: gfx906 M=1 dense GEMV -> ROCm skinny LLGemm1_kernel dispatch
docker cp "$CONTAINER:$SP/vllm/model_executor/layers/utils.py" \
  "$CTX/patches/layers_utils.py"
# P0.6: gfx906 must not be classified as CDNA. The substring match
# ("gfx9" in "gfx906") made supports_fp8() return True and let
# --kv-cache-dtype fp8_e4m3 through on a GPU with no FP8 arithmetic.
# Keep it in sync with docker/patches/platforms_rocm.py (md5 221a7013...).
cp -f "$HERE/patches/platforms_rocm.py" "$CTX/patches/platforms_rocm.py"
# P2: the GEMV kernel source comes from the repo too -- the container's
# libp2gemv.so is a dev build the Dockerfile deliberately deletes before
# compiling this .hip (and [7/7]'s tarball would otherwise carry it in).
cp -f "$HERE/patches/p2_w4a16_gemv.hip" "$CTX/patches/p2_w4a16_gemv.hip"

# Everything the Dockerfile COPYs must now be the repo's copy. Fingerprint the
# group so a stale $CTX/patches cannot ride into a build unnoticed ([3/7]
# overwrites in place; without this the failure is silent).
echo "   build-context/patches fingerprint:"
( cd "$CTX/patches" && md5sum *.py *.hip *.pth 2>/dev/null | sort -k2 )

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
