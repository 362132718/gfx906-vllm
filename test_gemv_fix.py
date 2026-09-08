# test_gemv_fix.py -- import fix, then exercise the previously-crashing ops.
import torch

import gfx906_gemv_fix
print("APPLIED:", gfx906_gemv_fix.APPLIED, flush=True)
assert gfx906_gemv_fix.APPLIED, "fix did not engage"

v = torch.randn(256, device="cuda")
a = torch.randn(256, 256, device="cuda")
ref = a.cpu() @ v.cpu()
for name, fn in (
    ("aten.mv", lambda: torch.ops.aten.mv(a, v)),
    ("torch.mv", lambda: torch.mv(a, v)),
    ("matmul-1d", lambda: a @ v),
    ("addmv", lambda: torch.addmv(torch.zeros(256, device="cuda"), a, v)),
):
    r = fn()
    torch.cuda.synchronize()
    print("%s maxdiff=%.2e" % (name, (r.cpu() - ref).abs().max().item()),
          flush=True)
# fp16 path still works (was never broken)
b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
vf = torch.randn(256, device="cuda", dtype=torch.float16)
torch.mv(b, vf)
torch.cuda.synchronize()
print("f16 mv still OK", flush=True)
print("SURVIVED", flush=True)
