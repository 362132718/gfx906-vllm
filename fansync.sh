#!/usr/bin/env bash
# fansync.sh -- MI50-with-VII-BIOS cards often report no tach because the fan is
# not wired to the card header. Check what PWM control exists and optionally
# force it, so we can tell "cooling inadequate" from "power rail".
for h in /sys/class/hwmon/hwmon3 /sys/class/hwmon/hwmon4; do
  echo "$h: pwm files: $(ls $h | grep -E '^pwm|^fan' | tr '\n' ' ')"
  for f in pwm1 pwm1_input pwm1_max pwm1_auto_channels_temp fan1_input; do
    [ -e "$h/$f" ] && echo "   $f = $(cat $h/$f)"
  done
done
