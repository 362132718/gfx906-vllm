#!/bin/bash
# Re-verify the P2 plan fences after they were rewritten to the landed text:
# extract -> drift-check against the shipped module -> compile the extracted .hip
# -> run the real ctypes/fp64 verification against the .so already installed.
# Never touches dist-packages (Task 1's installed .so stays where it is).
set -uo pipefail
cd /work/p2check2
INST=/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so

echo "== installed .so"; md5sum "$INST" 2>&1
echo "== shipped module md5 (want 37325e2aa3da0dcdcf3317111e098883)"
md5sum wna16_exllama.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py

echo "== [1] fence extraction + drift check vs landed"
python3 p2_plan_selfcheck.py x.md /work/p2check2 wna16_exllama.py \
  || { echo "SELFCHECK-FAIL"; exit 1; }

echo "== [2] hipcc build of the extracted .hip"
echo "   source vs Task 1 shipped .hip:"
md5sum p2_kernels_prod.hip /work/p2/p2_kernels_prod.hip 2>&1
cmp -s p2_kernels_prod.hip /work/p2/p2_kernels_prod.hip \
  && echo "   HIP-SOURCE-IDENTICAL" || echo "   HIP-SOURCE-DIFFERS"
hipcc -shared -fPIC --offload-arch=gfx906 -O3 p2_kernels_prod.hip \
  -o libp2gemv_fence.so || { echo "HIPCC-FAIL"; exit 1; }
nm -D --defined-only libp2gemv_fence.so | grep -qw p2_gemv \
  && echo "   symbol p2_gemv: ok" || { echo "SYMBOL-MISSING"; exit 1; }

echo "== [3] fence-built vs installed .so -- identical means Task 1 shipped the fence"
md5sum libp2gemv_fence.so "$INST"
cmp -s libp2gemv_fence.so "$INST" && echo "SO-IDENTICAL" || echo "SO-DIFFERS"

echo "== [4] real ctypes + fp64 reference (against the installed .so)"
HIP_VISIBLE_DEVICES=0 python3 verify_p2_blocks.py 2>&1 | tail -8

echo "== [5] gate file in sync with the repo copy"
md5sum p2_route_gate.py /work/p2/p2_route_gate.py 2>&1
echo FENCE-RECHECK-DONE
