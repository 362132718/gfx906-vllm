#!/usr/bin/env bash
# 5s sysfs sampler (edge/junction temp, board power, sclk) for the whole T6 window,
# so a second silent power loss leaves evidence of what the pair was drawing at the
# trip. Named differently from the killed helper on purpose.
L=$1; n=0
while [ $n -lt 4200 ]; do
  line="$(date -Is)"
  for h in /sys/class/drm/card0/device/hwmon/hwmon* /sys/class/drm/card1/device/hwmon/hwmon*; do
    [ -d "$h" ] || continue
    line="$line ${h##*/hwmon}:$(cat "$h/name" 2>/dev/null)"
    for f in temp1_input temp2_input power1_input freq2_input; do
      v=$(cat "$h/$f" 2>/dev/null); line="$line ${f%%_input}=${v:-NA}"
    done
  done
  echo "$line" >> "$L"
  n=$((n+1)); sleep 5
done
