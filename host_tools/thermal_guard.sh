#!/bin/bash
# gfx906 sustained-load thermal guard + logger (v4).
# Samples amdgpu hwmon temps every 5s to /tmp/thermal_run.log.
# Safety: aborts the active workload when EITHER
#   - any die junction >= 104 C (crit 105 C), or
#   - any HBM temp   >= 94 C  (crit 94 / emerg 99 C)
# then keeps logging the cooldown curve. Re-arms once all junctions < 80 C
# and all HBM temps < 78 C.
LOG=/tmp/thermal_run.log
LIMIT_J=104000
LIMIT_M=94000
REARM_J=80000
REARM_M=78000
: > "$LOG"
echo "start $(date -Is)" >> "$LOG"

mapfile -t AMD < <(for d in /sys/class/hwmon/hwmon*; do
  [ "$(cat "$d/name" 2>/dev/null)" = "amdgpu" ] && echo "${d##*/}"
done)

for h in "${AMD[@]}"; do
  bdf=$(readlink -f "/sys/class/hwmon/$h/device" 2>/dev/null | awk -F/ '{print $NF}')
  echo "map $h -> $bdf" >> "$LOG"
done

kill_workload() {
  docker exec thermtest pkill -f burn.py 2>/dev/null
  docker exec vllm_fantest pkill -f Qwen3.8-27B 2>/dev/null
  docker exec vllm_fantest pkill -f gfx906-serve 2>/dev/null
  sleep 2
  docker exec thermtest pkill -9 -f burn.py 2>/dev/null
  docker exec vllm_fantest pkill -9 -f Qwen3.8-27B 2>/dev/null
  docker stop thermtest >/dev/null 2>&1
}

armed=1
for i in $(seq 1 2880); do
  ts=$(date +%H:%M:%S)
  line=""
  hotj=0
  hotm=0
  maxj=0
  maxm=0
  for h in "${AMD[@]}"; do
    e=$(cat "/sys/class/hwmon/$h/temp1_input" 2>/dev/null)
    j=$(cat "/sys/class/hwmon/$h/temp2_input" 2>/dev/null)
    m=$(cat "/sys/class/hwmon/$h/temp3_input" 2>/dev/null)
    p=$(cat "/sys/class/hwmon/$h/power1_input" 2>/dev/null)
    f=$(cat "/sys/class/hwmon/$h/freq1_input" 2>/dev/null)
    line="$line $h[e=$((e/1000)) j=$((j/1000)) m=$((m/1000)) p=$((p/1000000))W clk=$((f/1000000))]"
    [ -n "$j" ] && [ "$j" -gt "$maxj" ] && maxj=$j
    [ -n "$m" ] && [ "$m" -gt "$maxm" ] && maxm=$m
    [ -n "$j" ] && [ "$j" -ge "$LIMIT_J" ] && hotj=1
    [ -n "$m" ] && [ "$m" -ge "$LIMIT_M" ] && hotm=1
  done

  reason=""
  [ "$hotj" = "1" ] && reason="junction>=104C"
  [ "$hotm" = "1" ] && reason="$reason mem>=94C"

  if [ -n "$reason" ] && [ "$armed" = "1" ]; then
    echo "$ts$line  <<< ABORT $reason" >> "$LOG"
    kill_workload
    echo "$ts ABORTED by thermal guard ($reason); logging cooldown" >> "$LOG"
    armed=0
    sleep 5
    continue
  fi

  if [ "$armed" = "0" ] && [ "$maxj" -lt "$REARM_J" ] && [ "$maxm" -lt "$REARM_M" ]; then
    echo "$ts re-armed (junction<80C, mem<78C)" >> "$LOG"
    armed=1
  fi

  echo "$ts$line" >> "$LOG"
  sleep 5
done
echo "done $(date -Is)" >> "$LOG"
