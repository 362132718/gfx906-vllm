#!/usr/bin/env bash
# clockcap.sh -- pin both cards to a low DPM level, or hand the clock back.
#
# Why: the flashed VII power table does not clamp (224 W measured against a
# declared "190 W cap"), and this host has silently power-cycled more than once.
# A pinned level also makes numbers directly comparable with the existing
# pin=1425MHz bucket table (logs/p0/p05_buckets_ab_on.log).
#
# Syntax note (verified 2026-09-11, kernel/amdgpu in ROCm 7.2.4):
#   `pp_dpm_sclk` accepts the BARE level index. The "s <lvl>" form documented in
#   older trees returns EINVAL here ("无效的参数"), which silently leaves the
#   card at whatever level it was on -- so we try bare first and fall back.
#   `power_dpm_force_performance_level` wants the literal `auto`; the older
#   `default` spelling ALSO returns EINVAL, so the original "restore" path of
#   this script never worked either. Both are now read back and asserted.
#   Confirm with `clockcap.sh show`: the pinned level carries a `*`.
#
# Usage:
#   clockcap.sh 4        pin to level 4 (1425 MHz)
#   clockcap.sh auto     hand the clock back to the driver
#   clockcap.sh show     read-only: current levels + force level
#
# NOTE ON EXPECTATION (2026-09-11): the observed power loss happened at
# 44-48 C and ~857 MHz idle clock, i.e. it was NOT thermal and NOT correlated
# with sustained load. Pinning is still worth doing for reproducibility, but do
# not treat it as a crash mitigation -- instead make every run resumable
# (detached + logs on a bind mount), which is what actually saves the work.
set -u
lvl=${1:-auto}

for d in /sys/class/drm/card0/device /sys/class/drm/card1/device; do
  echo "== $d"
  case "$lvl" in
    show)
      ;;
    auto)
      echo auto | sudo tee "$d/power_dpm_force_performance_level" >/dev/null || \
        echo "   !! could not set force level to auto" >&2
      # Once force level is auto the driver owns the level again; touching
      # pp_dpm_sclk here is at best a no-op, so don't bother.
      ;;
    *)
      echo manual | sudo tee "$d/power_dpm_force_performance_level" >/dev/null
      if ! echo "$lvl" | sudo tee "$d/pp_dpm_sclk" >/dev/null 2>&1; then
        echo "s $lvl" | sudo tee "$d/pp_dpm_sclk" >/dev/null 2>&1 || \
          echo "   !! could not set level $lvl (tried bare and 's $lvl')" >&2
      fi
      ;;
  esac
  [ "$lvl" = show ] || sleep 1
  sudo cat "$d/pp_dpm_sclk" 2>/dev/null | tr '\n' ' '; echo
done
echo "force level card0: $(cat /sys/class/drm/card0/device/power_dpm_force_performance_level)"
echo "force level card1: $(cat /sys/class/drm/card1/device/power_dpm_force_performance_level)"
# Read-back guard: a silent EINVAL used to leave the card at the wrong level.
if [ "$lvl" != show ] && [ "$lvl" != auto ]; then
  if ! sudo cat /sys/class/drm/card0/device/pp_dpm_sclk 2>/dev/null | grep -q "^$lvl:.*\*"; then
    echo "WARNING: level $lvl is not marked active on card0 -- verify before measuring" >&2
  fi
fi
