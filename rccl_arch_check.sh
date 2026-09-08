#!/bin/bash
T=/usr/local/lib/python3.12/dist-packages/torch/lib
R=/opt/rocm/lib
echo "=== torch/lib hip-related bundled .so ==="
ls -la $T | grep -E "amdhip|rccl|hsa|rocprofiler|hiprt" | awk '{print $5, $9}'
echo "=== arch strings: torch librccl ==="
strings -a $T/librccl.so | grep -oE "gfx9[0-9]{2}[a-z]?|gfx1[0-9]{3}[a-z]?" | sort | uniq -c | sort -rn | head
echo "=== arch strings: rocm librccl ==="
strings -a $R/librccl.so.1.0.70204 | grep -oE "gfx9[0-9]{2}[a-z]?|gfx1[0-9]{3}[a-z]?" | sort | uniq -c | sort -rn | head
echo "=== versions ==="
python3 - <<'PY'
import ctypes
for p in ["/usr/local/lib/python3.12/dist-packages/torch/lib/librccl.so",
          "/opt/rocm/lib/librccl.so.1"]:
    try:
        l = ctypes.CDLL(p)
        v = ctypes.c_int()
        l.ncclGetVersion(ctypes.byref(v))
        print(p, "-> raw", v.value)
    except Exception as e:
        print(p, "ERR", e)
PY
echo "=== does libtorch_hip NEED librccl? ==="
objdump -p $T/libtorch_hip.so 2>/dev/null | grep -i "rccl\|NEED.*hip" | head
echo "=== RPATH ==="
objdump -p $T/libtorch_hip.so 2>/dev/null | grep -i "rpath\|runpath" | head
