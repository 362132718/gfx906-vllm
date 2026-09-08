# gemv_sweep_case.py -- run ONE matmul shape, exit 90 if it survives.
# A segfault (no "OK" print, nonzero/negative exit) means this shape crashes.
# Usage: python3 gemv_sweep_case.py <mode> <m> <n> <k> <dtype>
import sys

import torch

mode, m, n, k, dt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dtype = {"f32": torch.float32, "f16": torch.float16}[dt]
torch.manual_seed(0)
dev = "cuda"

if mode == "mm":          # [m,k] @ [k,n]
    a = torch.randn(m, k, device=dev, dtype=dtype)
    b = torch.randn(k, n, device=dev, dtype=dtype)
    c = a @ b
elif mode == "mv":        # [m,k] @ [k]   (matrix-vector -> GEMV path)
    a = torch.randn(m, k, device=dev, dtype=dtype)
    v = torch.randn(k, device=dev, dtype=dtype)
    c = a @ v
elif mode == "mm1n":      # [m,k] @ [k,1]  (n=1 GEMV-shaped)
    a = torch.randn(m, k, device=dev, dtype=dtype)
    b = torch.randn(k, 1, device=dev, dtype=dtype)
    c = a @ b
torch.cuda.synchronize()
print("OK", mode, m, n, k, dt, float(c.float().sum()))
sys.exit(90)
