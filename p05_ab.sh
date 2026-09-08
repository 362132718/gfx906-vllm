#!/usr/bin/env bash
# p05_ab.sh -- run the P0 attribution harness twice inside one detached job:
#   ab_off: VLLM_ROCM_USE_SKINNY_GEMM=0 -> must reproduce 53.7 ms. That proves
#           the env lever reaches the workers AND that nothing else drifted
#           since the P0 trace, so ab_on is a single-variable comparison.
#   ab_on : patched default
# Sequential because each arm needs both cards; the 180 s gap is because the
# pair thermally trips if it goes straight back into load.
set -uo pipefail
for arm in off on; do
  out=/work/p0/ab_$arm
  mkdir -p "$out"
  if [ "$arm" = off ]; then export VLLM_ROCM_USE_SKINNY_GEMM=0
  else unset VLLM_ROCM_USE_SKINNY_GEMM; fi
  export OUT="$out" CLOCK_MHZ=1425
  if [ "$arm" = on ]; then export EXPECTED_MS=33.1; else unset EXPECTED_MS; fi
  echo "=== arm=$arm skinny=${VLLM_ROCM_USE_SKINNY_GEMM:-unset(=True)} OUT=$out EXPECTED_MS=${EXPECTED_MS:-auto} $(date -Is)"
  python3 /work/p0/p0_step.py 2>&1 | tee "/work/p0/ab_$arm.step.log"
  echo "=== arm=$arm rc=${PIPESTATUS[0]} $(date -Is)"
  sleep 180
done
echo "=== AB DONE $(date -Is)"
