# p1_diag3.py -- minimal smokes on the NEW binary, ordered to isolate:
#   (1) the config that was EXACT on the old binary (V3 sk1 K=256 N=64),
#   (2) multi-group-in-one-chunk without split (V3 sk1 K=1024),
#   (3) split with 2 groups per chunk (V3 sk2 K=1024),
#   (4) V2 sk1 (was nonzero-garbage),
# plus launch-error surfacing from the C++ (TORCH_CHECK on hipGetLastError).
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


def onehot_maxabs(v, K, N, sk, k, seed=0):
    x, qw, sc, zr = tensors(K, N, seed)
    x[k] = 1.0
    scratch = torch.zeros(sk * N, device=DEV, dtype=torch.float32)
    got = mod.p1_gemv(x, qw, sc, zr, scratch, v, sk, 8, True).float()
    g = k // G
    want = (unpack(qw)[k].float() - unpack_zp(zr, N)[g].float()) * sc[g].float()
    d = (got - want).abs().max().item()
    print("v%d K=%-5d N=%-4d sk=%d k=%-5d maxabs=%.3e" % (v, K, N, sk, k, d))
    return d


print("=== smokes on the new binary (one-hot; exact = fp16 rounding ~1e-4) ===")
onehot_maxabs(3, 256, 64, 1, 0)        # exact on the old binary
onehot_maxabs(3, 256, 64, 1, 100)
onehot_maxabs(3, 1024, 256, 1, 0)      # 8 groups in ONE chunk, no split
onehot_maxabs(3, 1024, 256, 1, 300)
onehot_maxabs(3, 1024, 256, 2, 300)    # 4 groups per chunk
onehot_maxabs(3, 1024, 256, 4, 300)    # 2 groups per chunk (was wrong)
onehot_maxabs(3, 1024, 256, 4, 0)
onehot_maxabs(1, 1024, 256, 4, 300)    # V1 with per-word debias
onehot_maxabs(1, 256, 64, 1, 0)
onehot_maxabs(2, 256, 64, 1, 0)        # V2 (was nonzero garbage)
