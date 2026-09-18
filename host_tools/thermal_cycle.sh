#!/bin/bash
# Accelerated thermal cycling for TIM break-in + per-cycle interface metric.
#
# Drives GPU1 (0000:10:00.0) only: heat to ~98 C, cut the load, cool to ~62 C,
# repeat. Each cycle logs an interface metric measured at a FIXED junction
# temperature (98 C): delta = junction - edge. A better die->coldplate path
# means MORE heat reaches the sink for the same hotspot, i.e. a HIGHER edge
# reading and a SMALLER delta. "rise" (seconds to climb 40 C -> 98 C) should
# also get longer as the path improves.
#
# Env: CYCLES (default 10)
CYCLES=${CYCLES:-10}
HOT=98000
COOL=62000
LOG=/tmp/thermal_cycle.log

h4=""
for d in /sys/class/hwmon/hwmon*; do
  bdf=$(readlink -f "$d/device" 2>/dev/null | awk -F/ '{print $NF}')
  [ "$bdf" = "0000:10:00.0" ] && h4="$d"
done
if [ -z "$h4" ]; then
  echo "GPU1 hwmon (0000:10:00.0) not found" >&2
  exit 1
fi

rd() { cat "$h4/$1" 2>/dev/null; }

: > "$LOG"
echo "start $(date '+%F %T') gpu=$h4 cycles=$CYCLES" >> "$LOG"

for c in $(seq 1 "$CYCLES"); do
  docker start thermtest >/dev/null 2>&1
  sleep 1
  docker exec -d thermtest bash -c "DEV=1 DUR=900 python3 /tmp/burn.py > /tmp/cycle_burn.log 2>&1"

  t0=$(date +%s)
  edge_at_hot=""
  for _ in $(seq 1 150); do
    j=$(rd temp2_input)
    if [ -n "$j" ] && [ "$j" -ge "$HOT" ]; then
      edge_at_hot=$(rd temp1_input)
      break
    fi
    sleep 2
  done
  t_rise=$(( $(date +%s) - t0 ))
  jpk=$(rd temp2_input)
  mpk=$(rd temp3_input)

  docker exec thermtest pkill -f burn.py 2>/dev/null
  sleep 2
  docker exec thermtest pkill -9 -f burn.py 2>/dev/null

  t1=$(date +%s)
  for _ in $(seq 1 180); do
    j=$(rd temp2_input)
    if [ -n "$j" ] && [ "$j" -le "$COOL" ]; then
      break
    fi
    sleep 2
  done
  t_cool=$(( $(date +%s) - t1 ))

  if [ -n "$edge_at_hot" ]; then
    dlt=$(( (jpk - edge_at_hot) / 1000 ))
  else
    dlt="n/a"
  fi
  echo "cycle $c: rise=${t_rise}s j=$((jpk / 1000))C e_at_hot=$((edge_at_hot / 1000))C mem=$((mpk / 1000))C delta=$dlt C cool=${t_cool}s" >> "$LOG"
  sleep 5
done

echo "done $(date '+%F %T')" >> "$LOG"
