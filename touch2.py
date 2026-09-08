# touch2.py -- initialise both Vega20 devices and keep both contexts alive with
# negligible work. size=0 -> allocation only; size=256 -> tiny matmuls (a few W).
# If the box dies here, the trigger is dual-device activity itself, not power.
import sys, time
import torch

size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
n = torch.cuda.device_count()
print("devices:", n, flush=True)
a = [torch.zeros(max(size, 16), device=f"cuda:{i}") for i in range(n)]
b = [torch.zeros(max(size, 16), device=f"cuda:{i}") for i in range(n)]
print("both devices initialised", flush=True)
t0, it = time.time(), 0
while time.time() - t0 < secs:
    for i in range(n):
        if size:
            x = torch.randn(size, size, device=f"cuda:{i}", dtype=torch.float16)
            y = x @ x
            del x, y
        else:
            a[i].add_(1.0)
    torch.cuda.synchronize()
    it += 1
    if it % 200 == 0:
        print(f"t={time.time()-t0:.0f}s it={it}", flush=True)
print("TOUCH2_DONE", it, flush=True)
