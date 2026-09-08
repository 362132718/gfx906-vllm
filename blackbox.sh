#!/usr/bin/env bash
# blackbox.sh -- 1Hz hardware sampler streamed over ssh to the workstation so the
# last lines survive a hard power loss. Columns carry edge/junction/mem temps per
# GPU: a dead fan shows up as junction or HBM running away under load.
g0=/sys/class/hwmon/hwmon3; g1=/sys/class/hwmon/hwmon4
rd() { cat "$1" 2>/dev/null || echo -; }
( sudo dmesg -w 2>/dev/null | grep --line-buffered -iE 'over-current|amdgpu.*(reset|error|timeout|therm|warn)|mce|aer|xid|hung task|soft lockup' \
   | while read -r l; do echo "KERNEL ${l##*kernel: }"; done ) &
KD=$!
trap 'kill $KD 2>/dev/null' EXIT
echo "ts cpu_C soc_C g0_edge g0_junc g0_mem g1_edge g1_junc g1_mem g0_W g1_W fan0 fan1 g0busy g1busy load1 mhz"
while :; do
  printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
    "$(date +%H:%M:%S)" \
    "$(awk '{printf "%.1f",$1/1000}' <<<"$(rd /sys/class/hwmon/hwmon2/temp1_input)")" \
    "$(awk '{printf "%.1f",$1/1000}' <<<"$(rd /sys/class/hwmon/hwmon2/temp3_input)")" \
    "$(awk '{printf "%.0f",$1/1000}' <<<"$(rd $g0/temp1_input)")" "$(awk '{printf "%.0f",$1/1000}' <<<"$(rd $g0/temp2_input)")" "$(awk '{printf "%.0f",$1/1000}' <<<"$(rd $g0/temp3_input)")" \
    "$(awk '{printf "%.0f",$1/1000}' <<<"$(rd $g1/temp1_input)")" "$(awk '{printf "%.0f",$1/1000}' <<<"$(rd $g1/temp2_input)")" "$(awk '{printf "%.0f",$1/1000}' <<<"$(rd $g1/temp3_input)")" \
    "$(awk '{printf "%.0f",$1/1000000}' <<<"$(rd $g0/power1_input)")" \
    "$(awk '{printf "%.0f",$1/1000000}' <<<"$(rd $g1/power1_input)")" \
    "$(rd $g0/fan1_input)" "$(rd $g1/fan1_input)" \
    "$(rd /sys/class/drm/card0/device/gpu_busy_percent)" "$(rd /sys/class/drm/card1/device/gpu_busy_percent)" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$(awk -F: '/cpu MHz/{printf "%d",$2; exit}' /proc/cpuinfo)"
  sleep 1
done
