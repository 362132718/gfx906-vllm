#!/bin/bash
# Inventory of manual state inside vllm-gfx906 that Dockerfile.rocm_gfx906 does not create.
SP=/usr/local/lib/python3.12/dist-packages

echo "=== python packages (from dist-info, no GPU import) ==="
for p in vllm triton transformers torch torchvision flash_attn compressed-tensors xformers; do
  d=$(ls -d "$SP"/${p}-*.dist-info 2>/dev/null | head -1)
  [ -n "$d" ] && echo "$(basename "$d")"
done

echo
echo "=== custom / patched vllm files (mtime newer than package install) ==="
BASE=$SP/vllm
REF=$(ls -d "$SP"/vllm-*.dist-info/RECORD 2>/dev/null | head -1)
[ -z "$REF" ] && REF="$BASE/__init__.py"
echo "reference timestamp: $REF ($(stat -c %y "$REF" 2>/dev/null))"
find "$BASE" -name '*.py' -newer "$REF" -printf '%TY-%Tm-%Td %p\n' 2>/dev/null | sort

echo
echo "=== RCCL in torch/lib ==="
TL=$SP/torch/lib
ls -la "$TL" | grep -i rccl
sha256sum "$TL/librccl.so.1.0" 2>/dev/null | cut -c1-16

echo
echo "=== /opt/rocm RCCL ==="
ls -la /opt/rocm/lib/ | grep -i rccl | head -5

echo
echo "=== rocBLAS/Tensile gfx906 files ==="
for d in /opt/rocm/lib/rocblas/library /opt/rocm/rocblas/library; do
  [ -d "$d" ] && echo "$d: $(ls "$d" 2>/dev/null | grep -ci gfx906) gfx906 files, $(ls "$d" 2>/dev/null | wc -l) total"
done
ls /opt/rocm/lib/gfx906-supplements 2>/dev/null | wc -l

echo
echo "=== Triton kernels src dir env / triton_kernels pkg ==="
echo "TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR:-<unset>}"
ls -d "$SP"/triton_kernels 2>/dev/null && echo triton_kernels_present
ls -d /app/triton_kernels_pkg /work/triton_kernels_pkg 2>/dev/null

echo
echo "=== env files / history ==="
ls -la /root/.bashrc /home/*/.bashrc 2>/dev/null
grep -rn "VLLM_\|HSA_\|TRITON" /etc/environment 2>/dev/null | head
docker_marker=$(ls /work/gfx906-supplements 2>/dev/null | wc -l)
echo "/work/gfx906-supplements entries: $docker_marker"
