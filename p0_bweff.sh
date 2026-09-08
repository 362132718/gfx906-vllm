#!/usr/bin/env bash
# One idle card, clocks pinned, probe three working-set sizes.
set -euo pipefail
P0=/home/x/vllm-work/p0
bash $P0/p0_clock.sh lock
docker exec -e DEV=0 vllm-gfx906 python /work/p0/p0_bweff.py
docker exec -e DEV=1 vllm-gfx906 python /work/p0/p0_bweff.py
bash $P0/p0_clock.sh unlock   # restores auto; never hand-write the sudo restore
