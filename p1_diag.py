# p1_diag.py -- one-hot diagnostics for the L1 prototypes. L0's trick applied to
# each variant: x = e_k makes out[n] the dequantized weight at (k, n), so any
# wrong pairing/offset/reduction shows up as a readable pattern, not a norm.
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


print("=== [A] V2 one-hot over ALL k (K=256, N=64, sk=1): find the pattern ===")
K, N = 256, 64
x, qw, sc, zr = tensors(K, N)
code = unpack(qw); zp = unpack_zp(zr, N)
scratch = torch.zeros(8 * N, device=DEV, dtype=torch.float32)
x.zero_(); x[0] = 1.0
got = mod.p1_gemv(x, qw, sc, zr, scratch, 2, 1, 8, True).float()
want = (code[0].float() - zp[0].float()) * sc[0].float()
print("n:        ", list(range(8)))
print("code[0,:8]:", code[0, :8].tolist())
print("got[:8]:  ", [round(t, 4) for t in got[:8].tolist()])
print("want[:8]: ", [round(t, 4) for t in want[:8].tolist()])
# hypothesis tests on n=0..7: which permutation of codes would explain got?
# print got/ (sc*x) ratios where x[k0]=1 => got[n] = scale * (code'[k0,n] - zp') for
# whatever code' the variant effectively used.
eff = got / sc[0].float().clamp(min=1e-6) + zp[0].float()
print("effective code used by V2 at k=0, n=0..7:", [round(t, 2) for t in eff[:8].tolist()])

print("=== [B] V3 at check() params (K=1024, N=256, sk=4): locate bad chunks ===")
K, N = 1024, 256
x, qw, sc, zr = tensors(K, N)
code = unpack(qw); zp = unpack_zp(zr, N)
scratch = torch.zeros(4 * N, device=DEV, dtype=torch.float32)
for k in (0, 100, 300, 540, 900, 1023):
    x.zero_(); x[k] = 1.0
    got = mod.p1_gemv(x, qw, sc, zr, scratch, 3, 4, 8, True).float()
    g = k // G
    want = (code[k].float() - zp[g].float()) * sc[g].float()
    d = (got - want).abs()
    print("v3 sk4 k=%-5d chunk=%d maxabs=%.3e at n=%d" % (
        k, k // 256, d.max().item(), int(d.argmax())))
