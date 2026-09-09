# p1_diag4.py -- on the CURRENT binary: per-chunk partials for the failing case,
# with the exact expected contribution, plus the argmax column traced end to end.
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
scratch = torch.zeros(sk * N, device=DEV, dtype=torch.float32)
parts = mod.p1_gemv(x, qw, sc, zr, scratch, 3, sk, 8, False)
print("partials for x=e_%d (V3, sk=%d):" % (k, sk))
for s in range(sk):
    m = parts[s].abs().max().item()
    print("  chunk %d: max|p|=%.4e first4=%s" % (s, m, [round(t, 4) for t in parts[s][:4].tolist()]))
g = k // G
want = (code[k].float() - zp[g].float()) * sc[g].float()
print("expected chunk %d contribution: first4=%s" % (g, [round(t, 4) for t in want[:4].tolist()]))
out = mod.p1_gemv(x, qw, sc, zr, scratch, 3, sk, 8, True).float()
n = int((out - want).abs().argmax())
print("argmax n=%d: out=%.4f want=%.4f code=%d zp=%d scale=%.5f" % (
    n, out[n].item(), want[n].item(), code[k, n].item(), zp[g, n].item(), sc[g, n].item()))
# is out simply missing chunk g (i.e. equal to sum over chunks != g)?
others = parts[[s for s in range(sk) if s != g]].sum(0).float()
print("out == sum(other chunks)?", torch.allclose(out, others, atol=1e-4),
      "| out == partial[g]?", torch.allclose(out, parts[g].half().float(), atol=1e-4))
