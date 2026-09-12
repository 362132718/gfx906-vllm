#!/bin/bash
# Task 5 pre-flight. Two questions, both CPU-only, both about provenance:
#   (1) is docker/patches/p2_w4a16_gemv.hip (what the image will compile) the same
#       bytes as the fence in the reviewed plan? A "verified kernel" whose shipped
#       source drifted from the reviewed one is the failure this checks for.
#   (2) is the .so the gates actually ran on a build of that same source? Compare
#       strings with the __hip_cuid_* / __hip_gpubin_handle_* lines filtered -- the
#       stamp is per translation unit AND path-dependent, so .so md5 is not a
#       same-source test (pitfall #25).
set -uo pipefail
cd /work/p2t5 || exit 1
PROD=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py
INST=/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so

echo "== [0] prod python file: container vs shipped container_src copy"
md5sum "$PROD" wna16_exllama.py

echo "== [1] fence extraction + fragment compile"
# The optional 3rd argument (drift check against the landed file) is left off HERE,
# not because drift is fine -- see p2_fence_diff.py, which currently reports three
# stale fences from Task 3's rework. The python file's provenance is nailed by md5
# in [0] (prepare_build_context.sh extracts exactly that installed file), and this
# pre-flight is only about the .so this task adds to the image.
python3 p2_plan_selfcheck.py plan.md . || { echo "SELFCHECK FAILED"; exit 1; }

echo "== [2] shipped .hip vs fence .hip"
md5sum p2_kernels_prod.hip p2_w4a16_gemv.hip
if cmp -s p2_kernels_prod.hip p2_w4a16_gemv.hip; then echo "SOURCE IDENTICAL"; else
  echo "SOURCE DIFFERS"; diff p2_kernels_prod.hip p2_w4a16_gemv.hip | head -20; exit 1; fi

echo "== [3] compile the shipped source"
hipcc -shared -fPIC --offload-arch=gfx906 -O3 p2_w4a16_gemv.hip -o libfresh.so || exit 1
nm -D --defined-only libfresh.so | grep -qw p2_gemv || { echo "NO p2_gemv"; exit 1; }
ls -l libfresh.so "$INST"

echo "== [4] fresh build vs the gate-tested .so (cuid aside)"
strip_cuid() { strings "$1" | grep -v __hip_cuid | grep -v __hip_gpubin_handle | sort; }
if diff <(strip_cuid libfresh.so) <(strip_cuid "$INST") > /tmp/p2_so_diff.txt 2>&1; then
  echo "SAME-EXCEPT-CUID"
else
  echo "DIFFERS:"; head -30 /tmp/p2_so_diff.txt
fi
echo "PREFLIGHT DONE"
