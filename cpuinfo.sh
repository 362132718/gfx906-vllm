#!/usr/bin/env bash
h=/sys/class/hwmon/hwmon2   # k10temp
echo "k10temp files: $(ls $h | tr '\n' ' ')"
for f in temp1_input temp1_max temp1_crit temp1_label tctl tjmax fan1_input; do
  [ -e "$h/$f" ] && echo "  $f = $(cat $h/$f)"
done
echo "cpu cooler hints: $(lscpu 2>/dev/null | grep -m1 'Model name')"
grep -m1 MHz /proc/cpuinfo
echo "--- cpufreq governor / current freqs ---"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
paste <(ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | head -2 | xargs -I{} cat {}) 2>/dev/null | head -2
echo "--- thermal zones ---"
for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null) trip=$(cat $z/policy 2>/dev/null)"; done
echo "--- fans ---"
for f in /sys/class/hwmon/hwmon*/fan*_input; do [ -e "$f" ] && echo "$f = $(cat $f)"; done
echo "--- any previous thermal/trip messages in last 2 days ---"
sudo journalctl -k --since "2026-09-01" --no-pager 2>/dev/null | grep -iE 'therm|trip|prochot|overheat|cpu.*hot|shutdown|watchdog' | tail -10
