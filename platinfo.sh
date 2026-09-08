#!/usr/bin/env bash
echo "=== amdgpu hwmon power/limit files ==="
for h in /sys/class/hwmon/hwmon*; do
  n=$(cat "$h/name" 2>/dev/null)
  [ "$n" = amdgpu ] || continue
  echo "hwmon: $h"
  ls "$h" | grep -E 'power|cur|volt' | tr '\n' ' '; echo
  for f in power1_cap power1_input device/power1_cap; do
    [ -e "$h/$f" ] && echo "   $f = $(cat "$h/$f")"
  done
done
echo "=== platform ==="
sudo dmidecode -s system-manufacturer 2>/dev/null
sudo dmidecode -s system-product-name 2>/dev/null
sudo dmidecode -s chassis-type 2>/dev/null
grep -m1 'model name' /proc/cpuinfo
echo "nproc=$(nproc)"
echo "=== power supply info from dmidecode (type 39) ==="
sudo dmidecode -t 39 2>/dev/null | grep -iE 'Name|Max Power Capacity|Status' | head -12
echo "=== recent USB devices attached ==="
lsusb 2>/dev/null | head -12
