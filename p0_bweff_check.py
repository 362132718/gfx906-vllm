# p0_bweff_check.py -- independent cross-check of the Triton probe: same traffic,
# different implementation (torch reduction kernels) plus a read+write copy whose
# expected time is ~2x the pure-read time. If these disagree with p0_bweff.py,
# the number is not usable as #20's floor.
import os, statistics
import torch

def timeit(fn, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

torch.cuda.set_device(int(os.environ.get("DEV", "0")))
n = int(4 * 2**30 / 2)                       # 4 GiB of fp16
x = torch.zeros(n, dtype=torch.float16, device="cuda")
y = torch.empty_like(x)
gb = n * 2 / 2**30
r = timeit(lambda: torch.sum(x))
c = timeit(lambda: y.copy_(x))
print("buffer        = %.1f GiB" % gb)
print("torch.sum     = %7.1f GB/s  (%.1f ms)" % (n * 2 / (r / 1e3) / 1e9, r))
print("copy r+w      = %7.1f GB/s  (%.1f ms)  [traffic counted twice]"
      % (2 * n * 2 / (c / 1e3) / 1e9, c))
print("ratio copy/read = %.2f  (expect ~1.8-2.3 if the read number is real)" % (c / r))
del x, y
