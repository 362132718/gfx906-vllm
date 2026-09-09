# p1_diag6.py -- why does check() fail at REAL shapes when it passes at 1024x256?
# One-hot readout + random-x stats at (5120, 5120), V3 sk1: separates wiring-at-
# scale from reference-at-scale.
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


def tensors(K, N, seed=0):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), device=DEV, dtype=torch.int32, generator=g)
    sc = (torch.rand(K // G, N, device=DEV, generator=g) * 0.02 + 0.001).half()
    zr = torch.randint(0, 2**31 - 1, (K // G, N // 8), device=DEV, dtype=torch.int32, generator=g)
    x = torch.zeros(K, device=DEV, dtype=torch.float16)
    return x, qw, sc, zr


K, N = 5120, 5120
x, qw, sc, zr = tensors(K, N)
code = unpack(qw); zp = unpack_zp(zr, N)
print("code stats: min=%d max=%d shape=%s" % (code.min(), code.max(), tuple(code.shape)))
print("zp stats  : min=%d max=%d shape=%s" % (zp.min(), zp.max(), tuple(zp.shape)))
scratch = torch.zeros(N, device=DEV, dtype=torch.float32)

print("=== one-hot at real shape (V3, sk=1) ===")
for k in (0, 100, 2560, 5119):
    x.zero_(); x[k] = 1.0
    got = mod.p1_gemv(x, qw, sc, zr, scratch, 3, 1, 8, True).float()
    g = k // G
    want = (code[k].float() - zp[g].float()) * sc[g].float()
    d = (got - want).abs()
    print("k=%-5d maxabs=%.3e at n=%d (got %.4f want %.4f)" % (
        k, d.max().item(), int(d.argmax()), got[int(d.argmax())].item(),
        want[int(d.argmax())].item()))

print("=== random-x (V3, sk=1) vs reference ===")
g0 = torch.Generator(device=DEV); g0.manual_seed(3)
xr = torch.randn(K, device=DEV, generator=g0).half()
got = mod.p1_gemv(xr, qw, sc, zr, scratch, 3, 1, 8, True).float()
acc = torch.zeros(N, device=DEV, dtype=torch.float32)
for g in range(K // G):
    xg = xr[g * G:(g + 1) * G].to(torch.float32)
    acc += sc[g].float() * ((xg.view(-1, 1) * code[g * G:(g + 1) * G].float()).sum(0)
                            - zp[g].float() * xg.sum())
want = acc.half().float()
rel = ((got - want).abs() / want.abs().clamp(min=1e-3)).max()
print("random-x max_rel=%.3e ; got NaN=%d want NaN=%d" % (
    rel, int(torch.isnan(got).sum()), int(torch.isnan(want).sum())))
