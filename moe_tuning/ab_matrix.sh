#!/usr/bin/env bash
# ab_matrix.sh -- same-methodology 3-point A/B for the two MoE fixes.
#
#   mx_base : m==1 patch OFF, tuned MoE config OFF
#   mx_m1   : m==1 patch ON,  tuned MoE config OFF
#   mx_both : m==1 patch ON,  tuned MoE config ON
#
# All three at the DELIVERY config (MM=1 = vision on, MAX_LEN=16384,
# PREFIX_CACHE=1, GMU=0.92) except MAX_NUM_SEQS=32: the un-fixed baseline only
# has 33 Mamba cache blocks at MM=1, so 48 refuses to start. 32 is the largest
# value all three arms can run, which is what makes the comparison valid.
#
# Measurement uses the warmup + REPS=3 path in p5_decode.py; the pre-warmup
# numbers from earlier today are NOT comparable (the first request absorbed
# one-time init cost and inflated the reported step).
#
# NOTE: never restore the patch with toggle_m1.py `on` -- its backup predates the
# fix and would reinstate the buggy weight.reshape(-1) form. ON is always an
# explicit docker cp of the repo copy, verified by md5.
set -uo pipefail

DIR=/home/x/vllm-work/p5
MT=/home/x/vllm-work/moe_tuning
CTR=p5-moe
SP=/usr/local/lib/python3.12/dist-packages
UTILS="$SP/vllm/model_executor/layers/utils.py"
CFGDIR="$SP/vllm/model_executor/layers/fused_moe/configs"
CFGNAME="E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json"
SRC_PATCH="$MT/layers_utils_fixed.py"
SRC_CFG="$MT/configs/$CFGNAME"
LOG="$MT/abmatrix.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

set_patch() {   # $1 = on|off
  if [ "$1" = on ]; then
    docker cp "$SRC_PATCH" "$CTR:$UTILS" >/dev/null
    local a b
    a=$(md5sum "$SRC_PATCH" | awk '{print $1}')
    b=$(docker exec "$CTR" md5sum "$UTILS" | awk '{print $1}')
    [ "$a" = "$b" ] || { say "PATCH MD5 MISMATCH $a != $b"; exit 1; }
    say "  patch ON  (md5 $a)"
  else
    docker exec "$CTR" rm -f "$UTILS.m1bak"
    docker exec "$CTR" python3 /tmp/t.py off >/dev/null
    docker exec "$CTR" grep -q "acc = (x_view \* weight).sum" "$UTILS" \
      && { say "  patch OFF FAILED (marker still present)"; exit 1; }
    say "  patch OFF"
  fi
}

set_cfg() {     # $1 = on|off
  if [ "$1" = on ]; then
    docker exec "$CTR" cp "/tmp/moecfg.json" "$CFGDIR/$CFGNAME" 2>/dev/null || \
      docker cp "$SRC_CFG" "$CTR:$CFGDIR/$CFGNAME" >/dev/null
    say "  config ON"
  else
    docker exec "$CTR" rm -f "$CFGDIR/$CFGNAME"
    docker exec "$CTR" sh -c "ls '$CFGDIR' | grep -q Radeon_Pro_VII" \
      && { say "  config OFF FAILED"; exit 1; }
    say "  config OFF"
  fi
}

run_arm() {     # $1 = label
  local label="$1"
  say "=== arm $label ==="
  for p in $(ps -eo pid,args | grep -E "[E]ngineCore|[p]5_decode\.py" | awk '{print $1}'); do
    sudo kill -9 "$p" 2>/dev/null
  done
  sleep 2
  docker restart "$CTR" >/dev/null 2>&1       # also releases the GPU context
  sleep 6
  docker cp "$DIR/p5_decode.py" "$CTR:/work/p5_decode.py" >/dev/null 2>&1
  cd "$DIR" || exit 1
  setsid nohup bash p5_decode.sh \
    "$label:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=32;MM=1;GMU=0.92;N_GEN=64;CTX=1600;PREFIX_CACHE=1;REPS=3" \
    >/dev/null 2>&1 < /dev/null &
  for i in $(seq 1 60); do                 # up to 20 min per arm
    sleep 20
    grep -qE "DECODE_DONE_OK|DECODE_FAIL" "$DIR/dec-$label/decode.log" 2>/dev/null && break
  done
  grep -aE "^DECODE_STEP|DECODE_FAIL" "$DIR/dec-$label/decode.log" 2>/dev/null \
    | tee -a "$LOG"
}

say "=== AB MATRIX start $(date -Iseconds) ==="
[ -f "$SRC_PATCH" ] || { say "missing $SRC_PATCH"; exit 1; }
[ -f "$SRC_CFG" ]   || { say "missing $SRC_CFG"; exit 1; }

set_patch off; set_cfg off; run_arm mx_base
set_patch on;  set_cfg off; run_arm mx_m1
set_patch on;  set_cfg on;  run_arm mx_both

say "=== restore both fixes ON ==="
set_patch on; set_cfg on
say "=== AB MATRIX done $(date -Iseconds) ==="
