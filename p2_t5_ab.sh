#!/usr/bin/env bash
# p2_t5_ab.sh -- G-P5's clean-container half: does the delivery image really honor
# the kill switch?
#
# Same two arms as G-P3's p2_ab.sh (one variable, VLLM_WNA16_P2_GEMV), but run in a
# FRESH container from vllm-gfx906:latest whose /root/.cache is empty by build
# guard -- so it also proves the image comes up from scratch and that the .so it
# compiled itself is the one the engine dispatches to.
#
# BASE=/work/p2t5 so Task 3's evidence under /work/p2/ab_* is not touched.
# p2_ab_gate.py reads BASE, so the same judge runs on this directory.
set -uo pipefail
export BASE=/work/p2t5
for arm in off on; do
  out=$BASE/ab_$arm
  mkdir -p "$out"
  export VLLM_CACHE_ROOT="/tmp/p2t5cache_ab_$arm"
  rm -rf "$VLLM_CACHE_ROOT"
  if [ "$arm" = off ]; then
    export VLLM_WNA16_P2_GEMV=0 EXPECTED_MS=33.008
  else
    unset VLLM_WNA16_P2_GEMV; export EXPECTED_MS=31.9
  fi
  export OUT="$out" CLOCK_MHZ=1425
  echo "=== arm=$arm p2=${VLLM_WNA16_P2_GEMV:-unset(=1)} cache=$VLLM_CACHE_ROOT EXPECTED_MS=$EXPECTED_MS $(date -Is)"
  python3 /work/p0/p0_step.py 2>&1 | tee "$BASE/ab_$arm.step.log"
  rc=${PIPESTATUS[0]}
  echo "=== arm=$arm p0_step rc=$rc $(date -Is)"
  if [ "$rc" = 0 ]; then
    PROF_DIR="$out/prof" OUT="$out" python3 /work/p0/p0_parse_trace.py \
      > "$BASE/ab_$arm.trace.log" 2>&1
    echo "=== arm=$arm parse_trace rc=$? $(date -Is)"
    ls -l "$out"/p0_step_shapes.csv.* 2>&1 | sed 's/^/    shapes: /'
  fi
  sleep 180                      # the pair thermally trips if it goes straight back in
done
echo "=== G-P5 ROUTING GATE $(date -Is)"
BASE=$BASE python3 /work/p2/p2_ab_gate.py
echo "=== T5 AB DONE rc=$? $(date -Is)"
