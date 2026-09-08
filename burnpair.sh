#!/usr/bin/env bash
# burnpair.sh -- A: only GPU0 busy, B: only GPU1 busy, C: both busy.
# 90s per step, 30s gap so the black box records cool-down between phases.
set -uo pipefail
step() {
  local label=$1 vis=$2
  echo "########## $label HIP_VISIBLE_DEVICES=$vis"
  ( docker exec -d vllm-gfx906 bash -c "HIP_VISIBLE_DEVICES=$vis python3 /tmp/burn.py 90 > /tmp/burn_$label.log 2>&1; echo BURN_EXIT=\$? >> /tmp/burn_$label.log" )
  sleep 30
  docker exec vllm-gfx906 bash -c "tail -2 /tmp/burn_$label.log" 2>/dev/null
  sleep 60
  echo "$label phase window closed"
  sleep 30
}
step A 0
step B 1
step C 0,1
echo BURNPART_DONE
