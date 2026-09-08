#!/usr/bin/env bash
# fanmax.sh -- force both cards' fans to full (manual PWM) and report. Used to
# separate "cooling control is broken" from "power rail is broken".
for h in /sys/class/hwmon/hwmon3 /sys/class/hwmon/hwmon4; do
  echo 1 | sudo tee $h/pwm1_enable >/dev/null 2>&1
  echo 255 | sudo tee $h/pwm1 >/dev/null 2>&1
  echo "$h pwm1_enable=$(cat $h/pwm1_enable 2>/dev/null) pwm1=$(cat $h/pwm1) fan=$(cat $h/fan1_input)"
done
