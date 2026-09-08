#!/usr/bin/env bash
# p0_clock.sh lock|status|unlock [sclk_level] [mclk_level]
# Discrete DPM pinning only: these cards are MI50s flashed with a Radeon Pro VII
# table, so pp_od_clk_voltage (arbitrary clocks) does not exist, and power1_cap
# is not actually enforced (we have read 202-224W against a 190W cap).
# A perf number measured without the pin is only comparable to another one
# taken in the same thermal state.
#
# BOTH clocks must be pinned. `manual` without pp_dpm_mclk leaves memory at DPM
# level 0 = 350MHz, i.e. ~358GB/s against the 1024GB/s the card can do, and every
# bandwidth-bound number (GEMV, lm_head, the whole decode step) comes out 2-3x
# inflated with no error anywhere. auto ramps mclk under sustained load, which is
# why the pre-pin delivery numbers looked fine; level 2 = 1000MHz is that state.
set -u
act=${1:-status}
# Level 8 = 1700MHz is the delivery-boost state. Long jobs are pinned lower when
# the host's thermtrip keeps killing them (see memory gfx906-host-instability):
# 4 = 1425MHz survived init where 8 died twice within ~5 minutes.
lvl=${2:-8}
mlvl=${3:-2}
rc=0
pin() { # $1=sysfs node  $2=level  $3=label
  sudo -n sh -c "echo $2 > $1" || { echo "cannot set level $2 on $1" >&2; return 1; }
  mhz=$(awk -F'[:M]' -v l="$2" '$1+0==l{gsub(/ /,"");print $2}' "$1")
  if grep -qE "^$2: ${mhz}Mhz *\*" "$1"; then
    echo "  $3 $1 -> level $2 = ${mhz}MHz"; return 0
  fi
  echo "  FAILED to pin $3 $1: $(grep -E '^[0-9]+: [0-9]+Mhz' "$1" | tr '\n' ' ')" >&2
  return 1
}
for dev in /sys/class/drm/card0/device /sys/class/drm/card1/device; do
  [ -e "$dev/pp_dpm_sclk" ] || { echo "SKIP $dev (no pp_dpm_sclk)"; continue; }
  if [ "$act" = lock ]; then
    # manual first: writing pp_dpm_sclk while level=auto silently does nothing.
    sudo -n sh -c "echo manual > $dev/power_dpm_force_performance_level" || { echo "cannot set manual on $dev" >&2; rc=1; continue; }
    pin "$dev/pp_dpm_sclk" "$lvl" sclk || rc=1
    pin "$dev/pp_dpm_mclk" "$mlvl" mclk || rc=1
  else
    # status / unlock both stay inside the loop: hand-written restore commands
    # have already bitten us once (nested quoting through ssh ate the $d).
    if [ "$act" = unlock ]; then
      sudo -n sh -c "echo auto > $dev/power_dpm_force_performance_level" || { echo "cannot restore $dev" >&2; rc=1; continue; }
      echo "unlocked $dev"
    fi
    echo "$dev level=$(cat $dev/power_dpm_force_performance_level) sclk=$(awk '/\*/{printf "%s ", $2}' "$dev/pp_dpm_sclk") mclk=$(awk '/\*/{printf "%s ", $2}' "$dev/pp_dpm_mclk")"
  fi
done
exit $rc
