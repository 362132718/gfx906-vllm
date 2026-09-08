#!/usr/bin/env bash
# clockcap.sh -- pin both cards to a low DPM level. The flashed VII power table
# does not clamp (we measured 224W against a "190W cap"), so clock is the only
# reliable way to cut current. Usage: clockcap.sh <level|auto> [show]
lvl=${1:-auto}
for d in /sys/class/drm/card0/device /sys/class/drm/card1/device; do
  echo "== $d"
  if [ "$lvl" = auto ]; then
    echo default | sudo tee $d/power_dpm_force_performance_level >/dev/null
    echo "default" | sudo tee $d/pp_dpm_sclk >/dev/null 2>&1
  elif [ "$lvl" = show ]; then
    :
  else
    echo manual | sudo tee $d/power_dpm_force_performance_level >/dev/null
    echo "s $lvl" | sudo tee $d/pp_dpm_sclk >/dev/null
  fi
  [ "$lvl" = show ] || sleep 1
  sudo cat $d/pp_dpm_sclk 2>/dev/null | tr '\n' ' '; echo
done
echo "force level now: $(cat /sys/class/drm/card0/device/power_dpm_force_performance_level)"
