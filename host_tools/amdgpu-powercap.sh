#!/bin/bash
# amdgpu-powercap.sh -- persist per-GPU power caps across reboot.
#
# Why: power1_cap is a runtime-only sysfs knob; a reboot (or a crash-reboot)
# restores the VBIOS default (190 W here). Without this hook the machine
# returns to 190 W and the dual-card thermal problem comes right back.
#
# ORDERING MATTERS (2026-09-14, observed): amdgpu-pp-fan-patch.sh writes the
# soft PowerPlay table, which re-initialises power management and RESETS
# power1_cap back to the VBIOS default. Both scripts run as cron @reboot, so a
# naive "set and forget" gets clobbered. This script therefore
#   1) waits for the amdgpu hwmon nodes,
#   2) waits until the PP-fan patch has finished THIS boot,
#   3) sets the cap and verifies it HOLDS, retrying on any reset.
#
# Installed as /usr/local/bin/amdgpu-powercap.sh, driven by
# /etc/cron.d/amdgpu-powercap (@reboot).
CAP_W=130                        # watts per GPU -- tune here
LOG=/var/log/amdgpu-powercap.log
PP_LOG=/var/log/amdgpu-pp-fan-patch.log
TARGET=$((CAP_W * 1000000))

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

amdgpu_hwmons() {
  for d in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$d/name" 2>/dev/null)" = "amdgpu" ] && echo "$d"
  done
}

log "start (target ${CAP_W}W per GPU)"

# --- 1) wait for hwmon (up to 180s) -----------------------------------------
for _ in $(seq 1 180); do
  [ -n "$(amdgpu_hwmons)" ] && break
  sleep 1
done

# --- 2) wait for the PP-table patch to finish this boot (up to 240s) --------
boot_epoch=$(date -d "-$(cut -d. -f1 /proc/uptime) seconds" +%s)
for _ in $(seq 1 48); do
  if [ -f "$PP_LOG" ]; then
    last=$(tail -n 1 "$PP_LOG" 2>/dev/null)
    case "$last" in
      *"pp-fan-patch done"*)
        ts=$(echo "$last" | cut -d' ' -f1,2)
        ep=$(date -d "$ts" +%s 2>/dev/null || echo 0)
        if [ "$ep" -ge "$boot_epoch" ]; then
          log "pp-fan-patch finished this boot; proceeding"
          break
        fi
        ;;
    esac
  fi
  sleep 5
done
sleep 3

# --- 3) set the cap and confirm it holds ------------------------------------
for attempt in 1 2 3 4 5 6; do
  for d in $(amdgpu_hwmons); do
    cur=$(cat "$d/power1_cap" 2>/dev/null)
    [ "$cur" = "$TARGET" ] && continue
    echo "$TARGET" > "$d/power1_cap" 2>/dev/null
  done
  sleep 8

  bad=0
  report=""
  for d in $(amdgpu_hwmons); do
    bdf=$(readlink -f "$d/device" 2>/dev/null | awk -F/ '{print $NF}')
    cur=$(cat "$d/power1_cap" 2>/dev/null)
    report="$report ${d##*/}($bdf)=$((cur / 1000000))W"
    [ "$cur" = "$TARGET" ] || bad=$((bad + 1))
  done

  if [ "$bad" -eq 0 ]; then
    log "OK attempt $attempt:$report"
    log "done"
    exit 0
  fi
  log "attempt $attempt:$report -- $bad card(s) off target, retrying"
done

log "FAILED: caps would not hold after 6 attempts"
log "done"
exit 1
