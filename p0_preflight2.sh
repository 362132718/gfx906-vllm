#!/usr/bin/env bash
# p0_preflight2.sh -- locate the model, read quantization_config, and empirically
# test whether kernel-level GPU tracing works for gfx906 under ROCm 7.2.4.
set -u
sep() { echo; echo "===== $1"; }

sep "dev container mounts (find the models dir)"
docker inspect vllm-gfx906 --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'

sep "quantization_config from inside the container"
docker exec vllm-gfx906 bash -lc '
for d in /models/*/ /model/models/*/ ; do
  [ -f "$d/config.json" ] || continue
  echo "-- $d"
  grep -o "\"quant_method\"[^,}]*\|\"bits\"[^,}]*\|\"group_size\"[^,}]*\|\"sym\"[^,}]*\|\"zero_point\"[^,}]*\|\"actorder\"[^,}]*\|\"format\"[^,}]*\|\"weight_format\"[^,}]*" "$d/config.json" | sort -u
done' 2>&1 | head -40

sep "rocprofv3 kernel trace test (tiny fp16 matmul, 1 GPU)"
docker exec vllm-gfx906 bash -lc '
cd /tmp && rm -rf rp && mkdir rp && cd rp
cat > t.py <<PY
import torch
a=torch.randn(512,5120,dtype=torch.float16,device="cuda")
b=torch.randn(5120,1024,dtype=torch.float16,device="cuda")
for _ in range(5): c=a@b
torch.cuda.synchronize()
PY
timeout 240 rocprofv3 --kernel-trace --output-format csv -o rp -- python t.py 2>&1 | tail -12
echo "--- outputs:"; find . -maxdepth 4 \( -name "*.csv" -o -name "*.json" \) | head -8
f=$(find . -maxdepth 4 -name "*.csv" -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$f" ]; then echo "--- largest csv: $f"; wc -l "$f"; head -2 "$f" | cut -c1-200; fi
' 2>&1 | tail -30

sep "torch.profiler HIP/CPU breakdown smoke test"
docker exec vllm-gfx906 bash -lc '
cd /tmp && python - <<PY 2>&1 | tail -14
import torch
from torch.profiler import profile, ProfilerActivity
a=torch.randn(1,5120,dtype=torch.float16,device="cuda"); b=torch.randn(5120,1024,dtype=torch.float16,device="cuda")
for _ in range(3): c=a@b
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
    for _ in range(10): c=a@b
    torch.cuda.synchronize()
ev=[e for e in p.key_averages() if e.self_device_time_total>0]
print("device-side kernel rows:", len(ev))
for e in sorted(ev,key=lambda x:-x.self_device_time_total)[:4]:
    print("  %-40s %.1fus" % (e.key[:40], e.self_device_time_total/10))
PY' 2>&1 | tail -20
