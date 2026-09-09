# p1_diag2.py -- instrument with reduce=False so the per-chunk partials are
# directly observable from python: separates "kernel computes wrong" from
# "reduction sums wrong".
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


print("=== [B2] V3 K=1024 N=256 sk=4 reduce=False: per-chunk partials, x=e_300 ===")
K, N = 1024, 256
x, qw, sc, zr = tensors(K, N)
code = unpack(qw); zp = unpack_zp(zr, N)
k = 300
x.zero_(); x[k] = 1.0
scratch = torch.zeros(4 * N, device=DEV, dtype=torch.float32)
parts = mod.p1_gemv(x, qw, sc, zr, scratch, 3, 4, 8, False)   # [4, N] float
for s in range(4):
    nz = int(parts[s].abs().sum().item() != 0)
    print("chunk %d: |partial|sum=%.4f  first4=%s" % (
        s, parts[s].abs().sum().item(), [round(t, 4) for t in parts[s][:4].tolist()]))
g = k // G
want_contrib = (code[k].float() - zp[g].float()) * sc[g].float()
print("expected contribution of chunk %d (holds k=%d): first4=%s" % (
    k // 256, k, [round(t, 4) for t in want_contrib[:4].tolist()]))

print("=== [B3] V1 K=1024 N=256 sk=4 reduce=False: per-chunk partials ===")
parts = mod.p1_gemv(x, qw, sc, zr, scratch, 1, 4, 8, False)
for s in range(4):
    print("chunk %d: |partial|sum=%.4f  first4=%s" % (
        s, parts[s].abs().sum().item(), [round(t, 4) for t in parts[s][:4].tolist()]))

print("=== [A2] V2 K=256 N=64 sk=1: full got vs want for k=0 ===")
K, N = 256, 64
x, qw, sc, zr = tensors(K, N)
code = unpack(qw); zp = unpack_zp(zr, N)
scratch = torch.zeros(2 * N, device=DEV, dtype=torch.float32)
x.zero_(); x[0] = 1.0
got = mod.p1_gemv(x, qw, sc, zr, scratch, 2, 1, 8, True).float()
want = (code[0].float() - zp[0].float()) * sc[0].float()
d = (got - want)
print("n       :", list(range(16)))
print("code    :", code[0, :16].tolist())
print("got     :", [round(t, 3) for t in got[:16].tolist()])
print("want    :", [round(t, 3) for t in want[:16].tolist()])
print("num wrong n:", int((d.abs() > 1e-3).sum()), "of", N)
