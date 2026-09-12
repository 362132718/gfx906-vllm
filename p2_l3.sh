#!/usr/bin/env bash
# p2_l3.sh -- G-P4 L3: teacher-forced margin A/B on the delivery (compiled) engine.
#
# Order is load-bearing, not cosmetic: the off arm writes l3_seqs.json (the
# sequence BOTH arms are asked to score) and the on arm reads it. Running on
# first aborts with a named reason instead of comparing two different texts.
#
# Fresh VLLM_CACHE_ROOT per arm for the same hygiene reason as p2_ab.sh: the key
# does not cover the W4A16 kernel choice, and a leftover artifact would be reused
# silently. Only /tmp/p2cache_l3_* is ever removed.
#
# Clocks: pinned from the HOST (container /sys is read-only) at the same 1425/1000
# the G-P3 A/B used, so L3's numerics and G-P3's tokens describe one machine state.
set -uo pipefail

wait_idle() {                      # both cards must be free before a start
  # Two traps in the obvious implementation:
  #   * `pgrep -cf 'VLLM::Wor'` is useless here -- vLLM hard-kills its TP workers
  #     and nobody reaps them, so the host keeps dozens of <defunct> VLLM::Worker_TP
  #     entries and the count never returns to 0 (also self-matches the shell).
  #     Hence: bracketed pattern, and drop state-Z lines.
  #   * a '[1-9][0-9]' busy regex misses single-digit utilization; compare numerically.
  for _ in $(seq 1 60); do
    live=$(ps -eo stat=,args= | grep '[V]LLM::Wor' | grep -cv '^Z' || true)
    busy_ok=1
    for b in $(cat /sys/class/drm/card0/device/gpu_busy_percent \
                   /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null); do
      [ "$b" -gt 0 ] 2>/dev/null && busy_ok=0
    done
    if [ "${live:-0}" = "0" ] && [ "$busy_ok" = 1 ]; then
      echo "    idle: live vllm procs=$live cards busy=0% $(date -Is)"; return 0
    fi
    echo "    waiting: live vllm procs=$live busy_ok=$busy_ok $(date -Is)"
    sleep 10
  done
  echo "    WARN still busy after 600 s: live=$live busy_ok=$busy_ok -- starting anyway"
}

for arm in off on; do
  out=/work/p2/l3
  mkdir -p "$out"
  export VLLM_CACHE_ROOT="/tmp/p2cache_l3_$arm"
  rm -rf "$VLLM_CACHE_ROOT"
  if [ "$arm" = off ]; then export VLLM_WNA16_P2_GEMV=0; else unset VLLM_WNA16_P2_GEMV; fi
  export ARM="$arm" OUT="$out" CLOCK_MHZ=1425
  echo "=== arm=$arm p2=${VLLM_WNA16_P2_GEMV:-unset(=1)} cache=$VLLM_CACHE_ROOT $(date -Is)"
  wait_idle
  python3 /work/p2/p2_l3_run.py 2>&1 | tee "/work/p2/p2_l3_$arm.log"
  rc=${PIPESTATUS[0]}
  echo "=== arm=$arm p2_l3_run rc=$rc $(date -Is)"
  [ "$rc" != 0 ] && { echo "=== L3 ABORTED: arm $arm failed, gate needs both"; exit "$rc"; }
  sleep 180                        # the pair thermally trips if it goes straight back in
done
echo "=== G-P4 L3 GATE $(date -Is)"
OUT=/work/p2/l3 python3 /work/p2/p2_l3_gate.py
echo "=== L3 DONE rc=$? $(date -Is)"
