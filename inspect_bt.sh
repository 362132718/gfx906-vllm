#!/bin/bash
echo "=== SIGNAL/SIGBUF LINES ==="
grep -n -i "signal\|SIGSEGV" /tmp/bt0.log | cut -c1-200 | head -10
echo "=== FIRST 14 FRAMES ==="
grep -n "^[[]*#[0-9]" /tmp/bt0.log | head -14 | cut -c1-200
echo "=== WHO DEFINES amd::Kernel::signature ==="
for lib in /usr/local/lib/python3.12/dist-packages/torch/lib/librccl.so /opt/rocm/lib/libamdhip64.so.7 /opt/rocm/lib/librocprofiler-sdk-tool.so /opt/rocm/lib/librocm-dbgapi.so.0; do
  [ -f "$lib" ] || continue
  n=$(nm -D --defined-only "$lib" 2>/dev/null | grep -c "amd.*Kernel.*signature")
  echo "$lib -> $n"
done
echo "=== SEARCH ALL ROCM LIBS ==="
grep -rl "amd::Kernel" /opt/rocm/lib/*.so* 2>/dev/null | head -5
echo "=== TORCH LIBRCCL vs ROCM LIBRCCL ==="
ls -la /usr/local/lib/python3.12/dist-packages/torch/lib/librccl* 2>&1
md5sum /usr/local/lib/python3.12/dist-packages/torch/lib/librccl.so 2>&1
