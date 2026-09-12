#!/usr/bin/env bash
# Detached launcher for Task 6 run 2. p2_t6_serve.sh is re-entrant: it skips any
# arm whose probe file already has a DECODE256 line, so this run only measures off.
L=${1:-/home/x/vllm-work/p2_t6_serve_run2.log}
setsid nohup bash /home/x/vllm-work/p2_t6_serve.sh > "$L" 2>&1 < /dev/null &
echo "launched -> $L pid=$!"
