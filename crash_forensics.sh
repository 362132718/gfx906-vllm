#!/usr/bin/env bash
# crash_forensics.sh -- what actually killed this box, per incident.
# Collects: boot timeline, unclean shutdowns, hardware/voltage/thermal/GPU
# kernel signatures (excluding the USB over-current spam that precedes every
# death), and what stage the container-side job had reached at each death.
sudo journalctl --list-boots --no-pager 2>/dev/null | tail -10
echo "=== last -x (reboot/shutdown history) ==="
last -x -F 2>/dev/null | head -14
echo
for b in -1 -2 -3 -4 -5; do
  echo "=== boot $b ==="
  sudo journalctl -b "$b" --no-pager -o short-iso 2>/dev/null | sed -n '1p;$p'
  echo "  -- final 6 kernel lines --"
  sudo journalctl -b "$b" -k --no-pager -o short-iso 2>/dev/null | tail -6 | cut -c1-140
done
echo "=== hardware-ish kernel signatures (2 days, over-current spam excluded) ==="
sudo journalctl -k --no-pager --since "2026-09-02" 2>/dev/null \
  | grep -iE 'machine check|mce|hardware error|fatal|thermal|throttl|voltage|power|psu|sur|aer|pcieport|amdgpu' \
  | grep -viE 'over-current|usb ' | awk '!seen[substr($0,index($0,"]")+1)]++' | tail -25
echo "=== distinct over-current device list ==="
sudo journalctl -k --no-pager --since "2026-09-02" 2>/dev/null | grep -oE 'usb [0-9a-z-]+: over-current' | sort | uniq -c | sort -rn | head -8
echo "=== current-boot GPU/driver bring-up ==="
sudo dmesg 2>/dev/null | grep -iE 'amdgpu|kfd|kfd' | tail -8 | cut -c1-140
echo "=== thermal / power sensors ==="
for z in /sys/class/thermal/thermal_zone*/temp; do [ -e "$z" ] && echo "$z = $(cat "$z")"; done
command -v sensors >/dev/null && sensors 2>/dev/null | head -25 || echo "(no lm-sensors)"
echo "=== nvme smart (temp/spare) ==="
command -v smartctl >/dev/null && sudo smartctl -A /dev/nvme0n1 2>/dev/null | grep -iE 'temperature|available_spare|percentage|media_errors|power_on' || echo "(no smartctl)"
echo "=== container-side job logs: last timestamp in each ==="
docker start vllm-gfx906 >/dev/null 2>&1
docker exec vllm-gfx906 bash -c 'for f in /tmp/c1_test3.log /tmp/c1_ck.log /tmp/c1_coda.log /tmp/vf3_out.log /tmp/vv_uncapped.log /tmp/c3gv.log /tmp/c3gv2.log /tmp/c3gv3_out.log; do [ -f "$f" ] || continue; printf "%-22s %s\n" "$(basename $f)" "$(grep -aoE "09-0[0-9] [0-9]{2}:[0-9]{2}:[0-9]{2}" $f | tail -1) $(tail -c 90 $f | tr -cd "\11\12\15\40-\176" | tr "\n" " " | tail -c 80)"; done'
