#!/usr/bin/env python3
"""gfx906 sustained-load thermal probe.

Saturating fp16 HGEMM loop -> compute-bound -> worst-case power draw.
Env:
  DEV  : HIP device index (default 0)
  N    : matmul dimension (default 8192)
  DUR  : seconds to run (default 1800)
"""
import os
import time

import torch

dev = int(os.environ.get("DEV", "0"))
n = int(os.environ.get("N", "8192"))
dur = float(os.environ.get("DUR", "1800"))

torch.cuda.set_device(dev)
name = torch.cuda.get_device_name(dev)
print(f"[dev{dev}] {name} N={n} dur={dur:.0f}s", flush=True)

a = torch.randn(n, n, dtype=torch.float16, device="cuda")
b = torch.randn(n, n, dtype=torch.float16, device="cuda")

for _ in range(5):
    a @ b
torch.cuda.synchronize()

t0 = time.time()
it = 0
while time.time() - t0 < dur:
    c = a @ b
    torch.cuda.synchronize()
    it += 1
    if it % 20 == 0:
        el = time.time() - t0
        print(
            f"[dev{dev}] it={it} elapsed={el:.0f}s "
            f"tflops={2.0 * n ** 3 * it / el / 1e12:.2f}",
            flush=True,
        )

el = time.time() - t0
print(f"[dev{dev}] DONE it={it} elapsed={el:.0f}s", flush=True)
