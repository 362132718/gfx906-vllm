# p1_diag5.py -- poison probe: fill scratch with -7777, run reduce=False, and see
# which chunks the kernel actually overwrote. Distinguishes "block never ran"
# (poison survives) from "ran and computed zero" (0 written) with zero rebuild
# ambiguity.
import os
os.environ["VLLM_WNA16_EXLLAMA"] = "1"
import torch
from torch.utils.cpp_extension import load

mod = load(name="p1_kernels", sources=["/work/p1/p1_kernels.cu"],
           extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
           build_directory="/tmp/p1_bench_build", verbose=False)
DEV, G = "cuda", 128
KBIT = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)],
                    device=DEV, dtype=torch.int64)


def unpack(qw):
    K8, N = qw.shape
    w = qw.to(torch.int64).unsqueeze(-1)
    c = (w >> KBIT.view(1, 1, 8)) & 0xF
    return c.permute(0, 2, 1).reshape(K8 * 8, N).to(torch.int8)


def unpack_zp(zr, N):
    K_G, N8 = zr.shape
    z = zr.to(torch.int64).unsqueeze(-1)
    sh = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    return ((z >> sh.view(1, 1, 8)) & 0xF).reshape(K_G, N8 * 8)[:, :N].to(torch.int32)


K, N, sk = 1024, 256, 4
g0 = torch.Generator(device=DEV); g0.manual_seed(0)
qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), device=DEV, dtype=torch.int32, generator=g0)
sc = (torch.rand(K // G, N, device=DEV, generator=g0) * 0.02 + 0.001).half()
zr = torch.randint(0, 2**31 - 1, (K // G, N // 8), device=DEV, dtype=torch.int32, generator=g0)
code = unpack(qw); zp = unpack_zp(zr, N)
k = 300
x = torch.zeros(K, device=DEV, dtype=torch.float16); x[k] = 1.0
POISON = -7777.0
scratch = torch.full((sk * N,), POISON, device=DEV, dtype=torch.float32)
parts = mod.p1_gemv(x, qw, sc, zr, scratch, 3, sk, 8, False)
print("poison probe (V3, sk=%d, x=e_%d):" % (sk, k))
for s in range(sk):
    p = parts[s]
    n_poison = int((p == POISON).sum())
    nz = p[p != POISON]
    print("chunk %d: poison=%3d/256  max|non-poison|=%.4e  first4=%s" % (
        s, n_poison, nz.abs().max().item() if nz.numel() else 0.0,
        [round(t, 3) for t in p[:4].tolist()]))
g = k // G
want = (code[k].float() - zp[g].float()) * sc[g].float()
print("expected chunk %d first4=%s (want[n=218]=%.4f)"
      % (g, [round(t, 4) for t in want[:4].tolist()], want[218].item()))
