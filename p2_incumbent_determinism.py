# p2_incumbent_determinism.py -- probe: is the incumbent reproducible run to run?
#
# WHY THIS EXISTS. p2_route_gate rows C/D compare two calls that both take the
# incumbent branch (gptq_gemm_out) and assert "<=1 ULP". They failed with
# n_diff ~71% of positions and max bit-index distance ~1.3e4 -- an order of
# magnitude beyond the "1-ULP atomicAdd" note on file. Either
#   (a) the incumbent really is that non-reproducible (fp16 atomicAdd over the
#       split-k partials, non-associative), or
#   (b) the P2 patch changed something on the incumbent path.
# Only (a) can be tested here, because this probe never calls apply_weights:
# it drives ops.gptq_gemm_out directly, which is the unpatched code path. If the
# spread reproduces, the gate's INSTRUMENT is wrong (bit-index distance explodes
# on cancellation) and the assertion must move to an fp64-reference band, which
# is what the project's own kernel-swap acceptance rule already says.
#
# Three measurement kinds per case, all on the SAME inputs twice:
#   eq        torch.equal (bit reproducibility)
#   adiff     max |a-b|, and max |a-b| / max|ref|  (the honest physical unit)
#   ulpidx    the monotone-bit-index distance the gate used, to show how badly
#             that metric misreads cancellation
# plus, for calibration, the error of EACH run against an fp64 reference: if two
# runs are each within eps of the reference but 100x eps apart, the difference is
# reordering, not a different computation.
import os
import sys

os.environ["VLLM_WNA16_EXLLAMA"] = "1"
sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l0_mapping as l0
from vllm import _custom_ops as ops

DEV, G, BITS = "cuda", 128, 4
CASES = ((5120, 8192, 1), (5120, 8192, 2), (5120, 17408, 1), (5120, 3072, 1))
# L0-pinned: the bridge writes code i at bits (i//2)*BITS + (i%2)*16
# (shuffle_4bit_8), NOT at the sequential 4*i a checkpoint stores it in. Using
# 4*i here made the first version of this probe disagree with the kernel by
# mean 2.4 absolute -- a broken reference, not a broken kernel.
KBIT = ((torch.arange(8) // 2) * BITS + (torch.arange(8) % 2) * 16)


def unpack64(qw):
    K8, N = qw.shape
    w = qw.to(torch.int64).unsqueeze(-1)
    c = (w >> KBIT.to(DEV).view(1, 1, 8)) & 0xF
    return c.permute(0, 2, 1).reshape(K8 * 8, N)


def unpack_zp64(zr, N):
    K_G, N8 = zr.shape
    z = zr.to(torch.int64).unsqueeze(-1)
    sh = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    return ((z >> sh.view(1, 1, 8)) & 0xF).reshape(K_G, N8 * 8)[:, :N]


def ref64(x, w_q, w_s, w_zp):
    K, N = x.shape[1], w_q.shape[1]
    code = unpack64(w_q).to(torch.float64)
    zp = unpack_zp64(w_zp, N).to(torch.float64)
    x64 = x.to(torch.float64)
    out = torch.zeros(x.shape[0], N, device=DEV, dtype=torch.float64)
    for g in range(K // G):
        xg = x64[:, g * G:(g + 1) * G]
        cg = code[g * G:(g + 1) * G]
        out += ((xg.unsqueeze(2) * (cg.unsqueeze(0) - zp[g])).sum(1)
                * w_s[g].to(torch.float64))
    return out


def ulp_idx(a16, b16):
    ai = a16.contiguous().view(torch.int16).to(torch.int32)
    bi = b16.contiguous().view(torch.int16).to(torch.int32)
    am = torch.where(ai >= 0, ai, -(ai & 0x7FFF))
    bm = torch.where(bi >= 0, bi, -(bi & 0x7FFF))
    return (am - bm).abs()


def inc(x, w_q, w_s, w_zp, g_idx, out):
    tdq = torch.empty(0, dtype=torch.float16, device=DEV)
    out.zero_()
    ops.gptq_gemm_out(out, x, w_q, w_zp, w_s, g_idx, True, True, BITS, tdq)
    torch.cuda.synchronize()
    return out


def main():
    for K, N, M in CASES:
        k2, layer, codes, scales, zp = l0.make_layer_and_kernel(K, N)
        w_q, w_s, w_zp, g_idx = k2._get_weight_params(layer)
        torch.manual_seed(7)
        x = torch.randn(M, K, device=DEV, dtype=torch.float16)
        a = inc(x, w_q, w_s, w_zp, g_idx, torch.zeros(M, N, device=DEV,
                                                      dtype=torch.float16))
        b = inc(x, w_q, w_s, w_zp, g_idx, torch.zeros(M, N, device=DEV,
                                                      dtype=torch.float16))
        # and the reused-slab form apply_weights uses: _cbuf[:M] after a zero_
        slab = torch.zeros(64, N, device=DEV, dtype=torch.float16)
        c = inc(x, w_q, w_s, w_zp, g_idx, slab[:M])
        d = inc(x, w_q, w_s, w_zp, g_idx, slab[:M])
        r = ref64(x, w_q, w_s, w_zp)
        scale = float(r.abs().max())
        print("== K=%d N=%d M=%d  |ref|max=%.4f ==" % (K, N, M, scale))
        for name, (u, v) in (("fresh vs fresh", (a, b)),
                             ("slab vs slab", (c, d)),
                             ("fresh vs slab", (a, c))):
            du = (u.float() - v.float()).abs()
            di = ulp_idx(u, v)
            print("   %-15s eq=%-5s n_diff=%6d/%d  max|a-b|=%.4e (%.2e of ref)  "
                  "max_ulpidx=%d  mean_ulpidx=%.3f"
                  % (name, bool(torch.equal(u, v)), int((du > 0).sum()),
                     u.numel(), float(du.max()), float(du.max()) / scale,
                     int(di.max()), float(di.float().mean())))
        for name, u in (("run a", a), ("run b", b), ("slab c", c), ("slab d", d)):
            e = (u.to(torch.float64) - r).abs()
            print("   %-15s max|run-ref|=%.4e  mean|run-ref|=%.4e"
                  % (name, float(e.max()), float(e.mean())))
        del k2, layer, w_q, w_s, w_zp, a, b, c, d, slab
        torch.cuda.empty_cache()
    print("DET-DONE")


if __name__ == "__main__":
    main()
