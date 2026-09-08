#!/usr/bin/env bash
# p0_preflight.sh -- read-only probes that decide spec details. No writes to sysfs.
set -u
sep() { echo; echo "===== $1"; }

sep "hosts: gpu state"
rocm-smi --showproducts 2>/dev/null | head -5 || true
for h in /sys/class/drm/card*/device; do
  [ -d "$h" ] || continue
  grep -q . "$h/mem_info_vram_used" 2>/dev/null || continue
  echo "-- $h"
  echo "  vram_used=$(cat $h/mem_info_vram_used 2>/dev/null) / $(cat $h/mem_info_vram_total 2>/dev/null)"
  echo "  power1_cap=$(cat $h/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)"
  for f in power_dpm_force_performance_level pp_od_clk_voltage pp_dpm_sclk pp_sclk_dts; do
    p="$h/$f"
    if [ -e "$p" ]; then
      echo "  $f mode=$(stat -c '%a %U:%G' "$p") cur=$(head -c 200 "$p" 2>/dev/null | tr '\n' '|')"
    else
      echo "  $f ABSENT"
    fi
  done
done
sudo -n true 2>/dev/null && echo "(sudo: available)" || echo "(sudo: needs password)"

sep "models: quantization_config"
for d in /models/*/ ; do
  [ -f "$d/config.json" ] || continue
  echo "-- $d"
  grep -o '"quant_method"[^,}]*\|"quant_original_dtype"[^,}]*\|"bits"[^,}]*\|"group_size"[^,}]*\|"sym"[^,}]*\|"zero_point"[^,}]*\|"actorder"[^,}]*\|"format"[^,}]*\|"weight_format"[^,}]*' "$d/config.json" | sort -u
  python3 - "$d/config.json" <<'PY' 2>/dev/null || true
import json,sys
c=json.load(open(sys.argv[1]))
qc=c.get("quantization_config",{})
print("  keys:", sorted(qc.keys()))
print("  config_group keys:", sorted((qc.get("config") or {}).keys())[:10])
cg=(qc.get("config") or {})
for k,v in list(cg.items())[:3]:
    print("   ",k,"->",v)
PY
done

sep "containers"
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | head -10

sep "profiler availability in image"
docker run --rm --entrypoint bash vllm-gfx906:latest -c '
for b in rocprofv3 rocprof ompijitcc; do printf "%-10s " "$b"; command -v $b || echo MISSING; done
ls /opt/rocm/bin 2>/dev/null | grep -i -E "prof|rocprofv3" | head
python -c "import torch;print(\"torch\",torch.__version__)"
python -c "from torch.profiler import profile, ProfilerActivity; print(\"torch.profiler import ok\")"
' 2>&1 | tail -20

sep "blackbox still running?"
ls -l /tmp/blackbox*.log 2>/dev/null | tail -3
pgrep -af blackbox | head -3 || echo "(no blackbox process on host)"
