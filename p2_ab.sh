#!/usr/bin/env bash
# p2_ab.sh -- G-P3: engine A/B with ONE variable, VLLM_WNA16_P2_GEMV.
#   arm off: =0        -> must reproduce the anchor 33.008 ms/step @1425MHz.
#                        If that drifts, the comparison is void (check
#                        conditions before blaming the kernel).
#   arm on : default 1 -> step <= 31.9 +/- 0.3 ms, tok/s >= 30.6.
#
# Each arm gets its OWN VLLM_CACHE_ROOT. After the G-P3 fix the M test and this
# flag both live INSIDE the custom-op impl, which dynamo does not trace, so the
# two arms now compile to the same artifact. The isolation stays for a
# different reason: an artifact traced by the pre-fix code (shape test in the
# traced region) would otherwise be reused, and the cache key does not cover
# it. The plan said `rm -rf /root/.cache/vllm`; that is the
# wrong directory (p0_step.py keys the root by W4A16 mode: the live one is
# /root/.cache/vllm-w4a16-exllama) and destructive, so we point at fresh
# scratch instead. Only /tmp/p2cache_ab_* is ever removed.
#
# Clocks must be pinned from the HOST: inside the container /sys is read-only,
# so p0_clock.sh lock fails there while status still reads fine.
set -uo pipefail
for arm in off on; do
  out=/work/p2/ab_$arm
  mkdir -p "$out"
  export VLLM_CACHE_ROOT="/tmp/p2cache_ab_$arm"
  rm -rf "$VLLM_CACHE_ROOT"
  if [ "$arm" = off ]; then
    export VLLM_WNA16_P2_GEMV=0 EXPECTED_MS=33.008
  else
    unset VLLM_WNA16_P2_GEMV; export EXPECTED_MS=31.9
  fi
  export OUT="$out" CLOCK_MHZ=1425
  echo "=== arm=$arm p2=${VLLM_WNA16_P2_GEMV:-unset(=1)} cache=$VLLM_CACHE_ROOT" \
       "EXPECTED_MS=$EXPECTED_MS $(date -Is)"
  python3 /work/p0/p0_step.py 2>&1 | tee "/work/p2/ab_$arm.step.log"
  rc=${PIPESTATUS[0]}
  echo "=== arm=$arm p0_step rc=$rc $(date -Is)"
  if [ "$rc" = 0 ]; then
    PROF_DIR="$out/prof" OUT="$out" python3 /work/p0/p0_parse_trace.py \
      > "/work/p2/ab_$arm.trace.log" 2>&1
    echo "=== arm=$arm parse_trace rc=$? $(date -Is)"
    ls -l "$out"/p0_step_shapes.csv.* 2>&1 | sed 's/^/    shapes: /'
  fi
  sleep 180                      # the pair thermally trips if it goes straight back in
done
echo "=== G-P3 GATE $(date -Is)"
python3 /work/p2/p2_ab_gate.py
echo "=== AB DONE rc=$? $(date -Is)"
