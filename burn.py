# burn.py -- sustained fp16 matmul on the visible devices; used to load exactly
# one GPU (HIP_VISIBLE_DEVICES) or both, to bisect which side kills the box.
import os, sys, time
import torch

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
print("visible devices:", devs, os.environ.get("HIP_VISIBLE_DEVICES"), flush=True)
xs = [(torch.randn(4096, 4096, device=d, dtype=torch.float16),
       torch.randn(4096, 4096, device=d, dtype=torch.float16)) for d in devs]
t0 = time.time()
n = 0
while time.time() - t0 < secs:
    for a, b in xs:
        c = a @ b
    torch.cuda.synchronize()
    n += 1
    if n % 50 == 0:
        print(f"t={time.time()-t0:.0f}s iters={n} done", flush=True)
print("BURN_DONE", n, flush=True)
