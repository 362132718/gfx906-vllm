#!/usr/bin/env bash
echo "=== PCIe identity (vendor:device + subsystem) ==="
lspci -nn | grep -iE 'vga|display' 
for d in 0000:0d:00.0 0000:10:00.0; do
  echo "--- $d"
  for f in device subsystem_device subsystem_vendor vendor; do
    printf '  %-16s %s\n' "$f" "$(cat /sys/bus/pci/devices/$d/$f 2>/dev/null)"
  done
  echo "  mmio_reloc? pm: $(cat /sys/bus/pci/devices/$d/power_state 2>/dev/null)"
done
echo "=== vbios / product names ==="
docker exec vllm-gfx906 bash -c 'rocm-smi --showvbios 2>/dev/null | grep -iE "GPU\[|VBIOS" | head -6; rocm-smi --showproductname 2>/dev/null | grep -iE "GPU\[|Card|Product|Vendor" | head -10'
echo "=== power cap reality check: cap/crit values ==="
for h in /sys/class/hwmon/hwmon3 /sys/class/hwmon/hwmon4; do
  echo "$h cap=$(cat $h/power1_cap 2>/dev/null) cap_min=$(cat $h/power1_cap_min 2>/dev/null) cap_max=$(cat $h/power1_cap_max 2>/dev/null) cap_def=$(cat $h/power1_cap_default 2>/dev/null)"
done
echo "=== sensor labels exposed by driver ==="
for h in /sys/class/hwmon/hwmon3 /sys/class/hwmon/hwmon4; do
  for l in $h/temp*_label $h/power*_label $h/in*_label; do
    [ -e "$l" ] && echo "  $(basename $l)=$(cat $l)"
  done
done
