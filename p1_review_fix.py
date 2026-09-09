# p1_review_fix.py -- review-v2 disposition items #1 (V3 ULP split, veto power),
# #2 (probe saturation: scalar vs mlp vs x4), #5 (VGPR via -Rpass-analysis).
#
# #1 measures ULP the way spec L2-c means it: fp64 reference, fp16 output, ULP =
# monotonic-bit distance. Reports RMS/MEAN/MAX ULP and %>0.5 both over ALL
# elements and over |ref|>0.01 (the old rel metric's clamp(min=1e-3) inflated
# near-zero outputs, so the 6.3e-4 reading mixed two components). The incumbent
# is measured identically: if it also fails 0.5 ULP, the gate as written is not
# achievable by anyone and must be recalibrated before it vetoes option A.
import os
os.environ["VLLM_WNA16_EXLLAMA"] = "1"
import json, subprocess, sys
import torch
from torch.utils.cpp_extension import load

mod = load(name="p1_kernels", sources=["/work/p1/p1_kernels.cu"],
           extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
           build_directory="/tmp/p1_bench_build", verbose=False)
sys.path.insert(0, "/work/p1")
import p1_l1_bench as b

DEV, G = b.DEV, 128
KBIT = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)],
                    device=DEV, dtype=torch.int64)
FLOORS = json.load(open("/work/p1/p1_floors.json"))


def unpack64(qw):
    K8, N = qw.shape
    w = qw.to(torch.int64).unsqueeze(-1)
    c = (w >> KBIT.view(1, 1, 8)) & 0xF
    return c.permute(0, 2, 1).reshape(K8 * 8, N)          # [K, N] int64


def unpack_zp64(zr, N):
    K_G, N8 = zr.shape
    z = zr.to(torch.int64).unsqueeze(-1)
    sh = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    return ((z >> sh.view(1, 1, 8)) & 0xF).reshape(K_G, N8 * 8)[:, :N]


def ref64_for(x, qw, sc, zr):
    """fp64 reference -- the zero-freedom ideal L2 wants."""
    K, N = qw.shape[0] * 8, qw.shape[1]
    code = unpack64(qw).to(torch.float64)
    zp = unpack_zp64(zr, N).to(torch.float64)
    acc = torch.zeros(N, device=DEV, dtype=torch.float64)
    x64 = x.to(torch.float64)
    for g in range(K // G):
        xg = x64[g * G:(g + 1) * G]
        acc += sc[g].to(torch.float64) * (
            (xg.view(-1, 1) * code[g * G:(g + 1) * G]).sum(0) - zp[g] * xg.sum())
    return acc


def fp16_ulp_dist(a16, b16):
    """Monotonic-bit ULP distance between two fp16 tensors (NaN excluded)."""
    ai = a16.view(torch.int16).to(torch.int32)
    bi = b16.view(torch.int16).to(torch.int32)
    am = torch.where(ai >= 0, ai, -(ai & 0x7fff))
    bm = torch.where(bi >= 0, bi, -(bi & 0x7fff))
    return (am - bm).abs()


def ulp_stats(got16, ref64, label):
    ref16 = ref64.half()
    ulp = fp16_ulp_dist(got16, ref16).to(torch.float64)
    big = ref64.abs() > 0.01                      # clamp-free subset
    res = dict(
        rms=float(torch.sqrt((ulp ** 2).mean())),
        mean=float(ulp.to(torch.float64).mean()),
        mx=int(ulp.max()),
        pct_gt_half=float(100.0 * (ulp > 0).double().mean()),  # any nonzero err
        rms_big=float(torch.sqrt((ulp[big] ** 2).mean())) if big.any() else -1,
        mx_big=int(ulp[big].max()) if big.any() else -1,
        n=int(ulp.numel()), n_big=int(big.sum()))
    print("%s RMS=%.3f MEAN=%.3f MAX=%d | big-only(|ref|>0.01, n=%d): RMS=%.3f MAX=%d"
          % (label, res["rms"], res["mean"], res["mx"], res["n_big"],
             res["rms_big"], res["mx_big"]), flush=True)
    return res


SHAPES = [(5120, 8192, 8, 8), (5120, 7168, 8, 8)]     # (K, N, sk, u) = V3 best
res = {"ulp": {}, "probe": {}}

print("=== [#1] ULP decomposition: V3 vs INCUMBENT, fp64 reference ===", flush=True)
from vllm import _custom_ops as ops
for K, N, sk, u in SHAPES:
    x, qw, sc, zr = b.tensors(K, N, 5)
    scratch = torch.zeros(sk * N, device=DEV, dtype=torch.float32)
    got_v3 = mod.p1_gemv(x, qw, sc, zr, scratch, 3, sk, u, True)
    c_out = torch.zeros(1, N, device=DEV, dtype=torch.float16)
    temp_dq = torch.empty(0, dtype=torch.float16, device=DEV)
    g_idx = torch.empty((0,), dtype=torch.int32, device=DEV)
    c_out.zero_()
    ops.gptq_gemm_out(c_out, x.view(1, K).contiguous(), qw, zr, sc, g_idx,
                      True, True, 4, temp_dq)
    ref64 = ref64_for(x, qw, sc, zr)
    res["ulp"]["%d,%d" % (K, N)] = dict(
        v3=ulp_stats(got_v3, ref64, "V3   K=%-5d N=%-5d" % (K, N)),
        inc=ulp_stats(c_out[0], ref64, "INC  K=%-5d N=%-5d" % (K, N)))

print("=== [#2] probe saturation: scalar vs mlp vs x4, 5 shapes ===", flush=True)
for key, meta in sorted(FLOORS["shapes"].items()):
    K, N = (int(v) for v in key.split(","))
    x, qw, sc, zr = b.tensors(K, N, 1)
    sink = torch.zeros(64, device=DEV, dtype=torch.int32)
    qw_bytes = (K // 8) * N * 4
    entry = {}
    for kind, name in ((None, "scalar"), (0, "mlp"), (1, "x4")):
        best = None
        for sk in (1, 2, 4, 8):
            if (K // 8) % sk:
                continue
            if kind is None:
                fn = lambda sk=sk: mod.p1_loadonly_launch(qw, sink, sk)
            else:
                fn = lambda sk=sk, kind=kind: mod.p1_loadonly_v2_launch(qw, sink, sk, kind)
            t = b.timeit(b.graphed(fn), warm=8, reps=32, R=8)
            best = t if best is None else min(best, t)
        entry[name] = dict(us=best, gbps=round(qw_bytes / best / 1e3, 1))
    res["probe"][key] = entry
    print("probe K=%-6d N=%-6d scalar=%.1fus (%s GB/s) mlp=%.1fus (%s) x4=%.1fus (%s)"
          % (K, N, entry["scalar"]["us"], entry["scalar"]["gbps"],
             entry["mlp"]["us"], entry["mlp"]["gbps"],
             entry["x4"]["us"], entry["x4"]["gbps"]), flush=True)

json.dump(res, open("/work/p1/p1_review_fix.json", "w"), indent=1)
print("REVIEW-FIX-DONE")
