#!/usr/bin/env bash
# p2p_ssm_dtype.sh -- P2': does --mamba-ssm-cache-dtype bfloat16 buy Mamba capacity?
#
# Why bf16 and not fp16: upstream's Qwen3.5 production recipe
# ("vLLM Reaches 25K Total TPS/GPU on Qwen3.5", GB200 NVL72) uses
#   --mamba-ssm-cache-dtype bfloat16
# and says it "significantly increases effective KV cache capacity on the decode
# endpoint". bf16 keeps fp32's 8 exponent bits, so the fp16 65504 overflow our
# notes worried about CANNOT HAPPEN. Same 2-vs-4-byte saving.
#
# Measurement trick: the Mamba block count is only ever printed in the ERROR
# message ("max_num_seqs (N) exceeds available Mamba cache blocks (M)"); a
# successful start never reports M. So force it with --max-num-seqs 512 -- each
# probe returns at KV-init (~6 min) instead of running to completion.
#
# Resilience (added after the 2026-09-12 10:50 host reboot killed a run in
# flight): logs are read from the HOST bind-mount path rather than via
# `docker exec`, so a container dying does not blind the reader; and a probe
# whose log already carries a result is SKIPPED, so re-running after an
# interruption does not redo finished work.
set -uo pipefail

IMG=vllm-gfx906:latest
CTR=p2p
MODEL=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
PORT=8180
WORK=/home/x/vllm-work/p2p          # bind-mounted to /work inside the container
LOG="$WORK/p2p.log"
mkdir -p "$WORK"
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

have_result() { grep -qaE 'available Mamba cache blocks \([0-9]+\)' "$WORK/$1.log" 2>/dev/null; }

probe() {   # $1 = label, rest = extra args to gfx906-serve
  local label=$1; shift
  if have_result "$label"; then
    say "=== probe $label : result already present, SKIPPED ==="
    return 0
  fi
  say "=== probe $label   extra='$*' ==="
  docker rm -f $CTR >/dev/null 2>&1
  docker run -d --name $CTR \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
    --shm-size 16g --ipc=host \
    -v /home/x/models:/models:ro -v "$WORK":/work \
    -p $PORT:$PORT "$IMG" sleep infinity >/dev/null || { say "run FAILED"; return 1; }
  sleep 6
  docker exec -d $CTR bash -lc \
    "cd /work && PORT=$PORT SERVED_NAME=qwen38 gfx906-serve $MODEL --max-num-seqs 512 $* > /work/$label.log 2>&1"

  local i=0
  for i in $(seq 1 90); do
    have_result "$label" && break
    docker ps --format '{{.Names}}' | grep -qx "$CTR" || { say "    container gone at ~$((i*10))s"; break; }
    sleep 10
  done
  say "    waited ~$((i*10))s"

  {
    echo "--- $label ---"
    printf '  blocks   : '; grep -aoE 'available Mamba cache blocks \([0-9]+\)' "$WORK/$label.log" 2>/dev/null | tail -1; echo
    printf '  kv size  : '; grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$WORK/$label.log" 2>/dev/null | tail -1; echo
    printf '  attn blk : '; grep -aoE 'Setting attention block size to [0-9]+ tokens' "$WORK/$label.log" 2>/dev/null | tail -1; echo
    printf '  mamba pad: '; grep -aoE 'Padding mamba page size by [0-9.]+%' "$WORK/$label.log" 2>/dev/null | tail -1; echo
    printf '  dtype    : '; grep -aioE 'mamba_ssm_cache_dtype[^ ,"]{0,40}|ssm_?cache[a-z_]*[= ][a-z0-9._]{3,12}' "$WORK/$label.log" 2>/dev/null | sort -u | head -3 | tr '\n' ' '; echo
  } | tee -a "$LOG"

  docker exec $CTR bash -lc 'pkill -f "vllm serve"; sleep 5; pkill -9 -f "vllm serve"' >/dev/null 2>&1
  docker rm -f $CTR >/dev/null 2>&1
  sleep 3
}

say "########## P2' start  $(date -Iseconds)"
probe probe_A
probe probe_B --mamba-ssm-cache-dtype bfloat16
say "P2P_DONE"
