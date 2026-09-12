# p2_diag_import.py -- why is _P2_OK what it is, in a fresh process?
#
# The G-P3 on arm recorded ZERO p2_gemv launches, i.e. the route never fired.
# The route is gated by _p2_layer_ok(), whose first term is _P2_OK -- set by a
# try/except at import time that swallows the reason. Reproduce that decision
# here instead of paying 12 minutes of engine start.
import ctypes
import os
import sys
import traceback

SO = "/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so"
ORDER = os.environ.get("P2_DIAG_ORDER", "torch-first")

print("order=%s so_exists=%s size=%s"
      % (ORDER, os.path.exists(SO), os.path.getsize(SO) if os.path.exists(SO) else 0))

if ORDER == "torch-first":
    import torch
    print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))

try:
    lib = ctypes.CDLL(SO)
    print("CDLL ok; p2_gemv callable =", callable(getattr(lib, "p2_gemv", None)))
except Exception:
    print("CDLL RAISED:")
    traceback.print_exc()

import vllm.model_executor.kernels.linear.mixed_precision.wna16_exllama as m
print("module _P2_OK =", m._P2_OK, "| _P2_LIB =", m._P2_LIB)
print("_P2_SPLITK =", m._P2_SPLITK, "| _P2_SHAPES =", sorted(m._P2_SHAPES))
import torch
print("op registered =", hasattr(torch.ops.vllm, "p2_w4a16_gemm"))
sys.stdout.flush()
