#!/usr/bin/env bash
# p0_preflight3.sh -- read quantization_config from the real model path, and test
# whether DPM clock pinning is actually writable (restores auto immediately).
set -u
sep() { echo; echo "===== $1"; }

sep "model dirs"
ls -d /home/x/models/*/ 2>/dev/null

sep "quantization_config"
for d in /home/x/models/*/ ; do
  [ -f "$d/config.json" ] || continue
  echo "-- $d"
  grep -o '"quant_method"[^,}]*\|"bits"[^,}]*\|"group_size"[^,}]*\|"sym"[^,}]*\|"zero_point"[^,}]*\|"actorder"[^,}]*\|"format"[^,}]*\|"input_scale"[^,}]*' "$d/config.json" | sort -u
  python3 -c "
import json,sys
c=json.load(open('$d/config.json'))
qc=c.get('quantization_config',{})
print('  qc keys:', sorted(qc.keys()))
cfg=qc.get('config')
if isinstance(cfg,dict):
    for k,v in list(cfg.items())[:4]: print('   ',k,'->',v)
print('  num_hidden_layers=',c.get('num_hidden_layers'),' hidden=',c.get('hidden_size'))
print('  vision' , 'vision_config' in c)
" 2>/dev/null || echo "  (python3 parse failed)"
done

sep "quant linear weight tensors (from a safetensors index, if present)"
for d in /home/x/models/*/ ; do
  idx="$d/model.safetensors.index.json"
  [ -f "$idx" ] || continue
  echo "-- $d"
  grep -o '"[^"]*q_proj\.[a-z_0-9]*"' "$idx" | sort -u | head -12
  grep -o '"[^"]*mlp[^"]*projector[^"]*"' "$idx" | head -0
  python3 -c "
import json
m=json.load(open('$idx'))['metadata']
print('   metadata:',m)
" 2>/dev/null
  break
done

sep "clock pin test (pin sclk to top DPM level, then restore auto)"
DEV=/sys/class/drm/card0/device
echo "before: $(cat $DEV/power_dpm_force_performance_level) | $(cat $DEV/pp_dpm_sclk | tr '\n' ' ')"
sudo -n sh -c "echo manual > $DEV/power_dpm_force_performance_level" 2>&1 | head -2
echo "after manual: $(cat $DEV/power_dpm_force_performance_level)"
sudo -n sh -c "echo 8 > $DEV/pp_dpm_sclk" 2>&1 | head -2
echo "pin rc readback: $(cat $DEV/pp_dpm_sclk | tr '\n' ' ')"
sudo -n sh -c "echo auto > $DEV/power_dpm_force_performance_level" 2>&1 | head -2
echo "restored: $(cat $DEV/power_dpm_force_performance_level) | $(cat $DEV/pp_dpm_sclk | tr '\n' ' ')"
