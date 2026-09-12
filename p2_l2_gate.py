# p2_l2_gate.py -- G-P4 L2: judge P2 on the REAL weights and REAL activations.
#
# Run in the container after p2_l2_dump.py:
#   docker exec -d vllm-gfx906 bash -lc "cd /work/p2 && \
#     OUT=/work/p2/l2 python3 p2_l2_gate.py > /work/p2/p2_l2_gate.log 2>&1"
#
# Task 0/1 measured ULP on SYNTHETIC operands (p1_l1_bench.tensors: random
# int32 words, uniform scales). This is the same question with the engine's own
# tensors, which is what the L2 gate is actually about: real scales span orders
# of magnitude and a real hidden state is not unit-norm.
#
# Two conditions must BOTH hold, per family, per dump (plan Task 4 Step 1):
#   [a] regression gate   RMS <= 0.05
#   [b] production gate   RMS <= incumbent RMS / 10
# [b] is the one with meaning here: the incumbent ExLlama kernel measures RMS
# 69-219 ULP against the same fp64 reference, so "0.05" alone would be an
# absolute band nobody on this hardware occupies. MAX and the big-only subset
# (|ref| > 0.01, the same split P1 review #1 introduced) are recorded, not gated.
#
# The reference and the splitter are copied from p1_review_fix.py (which runs
# measurement code at import time, so it cannot be imported) -- same unpack
# positions (L0-pinned on real GPTQ->ExLlama weights), same monotonic-bit ULP,
# same subset rule. Do not "improve" them here: a different ruler here would be
# incomparable with Task 0/1.
#
# The P2 side is the DUMPED out -- what the engine actually produced -- and the
# kernel is also re-run on the same operands into a fresh buffer: the two must be
# bit-identical, or the dump was corrupted / the kernel is not deterministic.
# The incumbent arm is non-deterministic on gfx906 (atomicAdd into c_out, see
# [[exllama-fused-kernel-atomic-add]]), so it is run N times and its own spread
# is reported alongside its RMS.
import glob
import json
import os
import sys

os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
import torch
from vllm import _custom_ops as ops

sys.path.insert(0, "/work/p2")
sys.path.insert(0, "/work/p1")
import p2_shim

DEV = "cuda"
G, SK = 128, 10
KBIT = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)],
                    device=DEV, dtype=torch.int64)
OUT = os.environ.get("OUT", "/work/p2/l2")
ULP_GATE, INC_RATIO = 0.05, 10.0
INC_RUNS = 3


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
    K = qw.shape[0] * 8
    code = unpack64(qw).to(torch.float64)
    zp = unpack_zp64(zr, qw.shape[1]).to(torch.float64)
    acc = torch.zeros(qw.shape[1], device=DEV, dtype=torch.float64)
    x64 = x.to(torch.float64)
    for g in range(K // G):
        xg = x64[g * G:(g + 1) * G]
        acc += sc[g].to(torch.float64) * (
            (xg.view(-1, 1) * code[g * G:(g + 1) * G]).sum(0) - zp[g] * xg.sum())
    return acc


def fp16_ulp_dist(a16, b16):
    ai = a16.reshape(-1).view(torch.int16).to(torch.int32)
    bi = b16.reshape(-1).view(torch.int16).to(torch.int32)
    am = torch.where(ai >= 0, ai, -(ai & 0x7FFF))
    bm = torch.where(bi >= 0, bi, -(bi & 0x7FFF))
    return (am - bm).abs()


def ulp_stats(got16, ref64):
    """Same fields as p1_review_fix.ulp_stats (minus its printing)."""
    ref16 = ref64.half()
    ulp = fp16_ulp_dist(got16, ref16).to(torch.float64)
    big = ref64.abs() > 0.01
    return dict(
        rms=float(torch.sqrt((ulp ** 2).mean())),
        mean=float(ulp.mean()),
        mx=int(ulp.max()),
        pct_gt_half=float(100.0 * (ulp > 0).double().mean()),
        rms_big=float(torch.sqrt((ulp[big] ** 2).mean())) if big.any() else -1,
        mx_big=int(ulp[big].max()) if big.any() else -1,
        n=int(ulp.numel()), n_big=int(big.sum()),
        ref_absmax=float(ref64.abs().max()),
        ref_absmean=float(ref64.abs().mean()))


def incumbent(x2d, w_q, w_s, w_zp, N):
    c_out = torch.zeros(1, N, device=DEV, dtype=torch.float16)
    temp_dq = torch.empty(0, dtype=torch.float16, device=DEV)
    g_idx = torch.empty((0,), dtype=torch.int32, device=DEV)
    ops.gptq_gemm_out(c_out, x2d, w_q, w_zp, w_s, g_idx, True, True, 4, temp_dq)
    return c_out[0]


def rerun_p2(d, K, N):
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    p2_shim.p2_gemv(d["x"].view(1, K).contiguous(), d["w_q"], d["w_s"],
                    d["w_zp"], part, out, K, N, SK)
    torch.cuda.synchronize()
    return out


def gate_row(K, N, p2_rms, inc_rms):
    return p2_rms <= ULP_GATE and p2_rms <= inc_rms / INC_RATIO


def main():
    files = sorted(glob.glob(os.path.join(OUT, "l2_dump_*.pt")))
    if not files:
        print("FAIL no dumps under %s -- run p2_l2_dump.py first" % OUT, flush=True)
        return 1
    print("L2 gate on real engine operands: %d dump(s), gate RMS <=%.2f and"
          " <= incumbent/%g" % (len(files), ULP_GATE, INC_RATIO), flush=True)
    rows, by_fam = [], {}
    for path in files:
        # weights_only: these dumps are ours (tensors + a dict of scalars), and
        # refusing to unpickle arbitrary objects costs nothing.
        d = torch.load(path, map_location=DEV, weights_only=True)
        m = d["meta"]
        K, N = m["K"], m["N"]
        x2d = d["x"].to(DEV).view(1, K).contiguous()
        w_q, w_s, w_zp = d["w_q"].to(DEV), d["w_s"].to(DEV), d["w_zp"].to(DEV)
        assert list(w_q.shape) == m["w_q_shape"] and w_s.shape[1] == N, \
            "dump round-trip changed a shape: %s" % path
        # SK is passed to p2_gemv below; a mismatch with how the engine produced
        # this dump would otherwise surface as a confusing "not bit-equal" FAIL.
        assert m["splitk"] == SK, "dump used splitk=%s, this gate passes %d" % (
            m["splitk"], SK)
        ref = ref64_for(x2d.view(-1), w_q, w_s, w_zp)
        eng = d["out"].to(DEV).reshape(-1)
        rerun = rerun_p2(d, K, N)
        identical = bool(torch.equal(eng, rerun))
        p2 = ulp_stats(eng, ref)
        incs, inc_outs = [], []
        for _ in range(INC_RUNS):
            o = incumbent(x2d, w_q, w_s, w_zp, N)
            inc_outs.append(o.clone())
            incs.append(ulp_stats(o, ref))
        inc = incs[0]
        # distance between the two production paths, on identical operands
        dist = fp16_ulp_dist(eng, inc_outs[0]).to(torch.float64)
        # the incumbent's own run-to-run noise, same operands: this is what
        # "reproducible" means for the path we are replacing
        inc_noise = fp16_ulp_dist(inc_outs[0], inc_outs[-1]).to(torch.float64)
        ok = gate_row(K, N, p2["rms"], inc["rms"])
        r = dict(K=K, N=N, file=os.path.basename(path), pid=m["pid"], dev=m["dev"],
                 hit=m["hit"], p2=p2, inc=inc,
                 inc_rms_all=[round(i["rms"], 3) for i in incs],
                 inc_rerun_max_ulp=int(inc_noise.max()),
                 inc_rerun_pct_nonzero=float(100.0 * (inc_noise > 0).double().mean()),
                 rerun_bit_equal=identical,
                 p2_vs_inc_rms=float(torch.sqrt((dist ** 2).mean())),
                 p2_vs_inc_max=int(dist.max()),
                 p2_vs_inc_pct_nonzero=float(100.0 * (dist > 0).double().mean()),
                 gate=ok)
        rows.append(r)
        by_fam.setdefault((K, N), []).append(r)
        print("\n-- %dx%d dev%s pid%s hit#%s  (%s)"
              % (K, N, m["dev"], m["pid"], m["hit"], os.path.basename(path)),
              flush=True)
        print("   ref      |y| max=%.3f mean=%.4f n=%d" %
              (p2["ref_absmax"], p2["ref_absmean"], p2["n"]), flush=True)
        print("   P2 eng   RMS=%.4f MEAN=%.4f MAX=%d >0=%.2f%% | big(n=%d)"
              " RMS=%.4f MAX=%d"
              % (p2["rms"], p2["mean"], p2["mx"], p2["pct_gt_half"],
                 p2["n_big"], p2["rms_big"], p2["mx_big"]), flush=True)
        print("   INC      RMS=%.3f MEAN=%.3f MAX=%d >0=%.2f%% | big RMS=%.3f"
              " MAX=%d   (%d runs RMS=%s)"
              % (inc["rms"], inc["mean"], inc["mx"], inc["pct_gt_half"],
                 inc["rms_big"], inc["mx_big"], INC_RUNS,
                 r["inc_rms_all"]), flush=True)
        print("   ratio    INC/P2 = %.0fx  (need >=%gx)"
              % (inc["rms"] / max(p2["rms"], 1e-9), INC_RATIO), flush=True)
        print("   P2 vs INC same operands: RMS=%.2f MAX=%d >0=%.1f%%"
              " | rerun bit-equal=%s"
              % (r["p2_vs_inc_rms"], r["p2_vs_inc_max"],
                 r["p2_vs_inc_pct_nonzero"], identical), flush=True)
        print("   INC rerun vs itself: MAX=%d ULP, %.1f%% of positions move"
              " (the incumbent's own noise floor)"
              % (r["inc_rerun_max_ulp"], r["inc_rerun_pct_nonzero"]), flush=True)
        print("   [%s]" % ("PASS" if ok else "FAIL"), flush=True)

    print("\n=== per-family verdict ===", flush=True)
    all_ok = True
    for (K, N), rs in sorted(by_fam.items()):
        worst_p2 = max(r["p2"]["rms"] for r in rs)
        best_ratio = min(r["inc"]["rms"] / max(r["p2"]["rms"], 1e-9) for r in rs)
        ok = all(r["gate"] for r in rs) and all(r["rerun_bit_equal"] for r in rs)
        all_ok = all_ok and ok
        print("K=%d N=%d  %d dump(s): worst P2 RMS=%.4f (gate %.2f),"
              " min INC/P2=%.0fx (need %gx), rerun-bit-equal=%s -> %s"
              % (K, N, len(rs), worst_p2, ULP_GATE, best_ratio, INC_RATIO,
                 all(r["rerun_bit_equal"] for r in rs),
                 "PASS" if ok else "FAIL"), flush=True)
    json.dump(rows, open(os.path.join(OUT, "p2_l2_rows.json"), "w"), indent=1)
    print("G-P4 L2 %s  (%d dumps)" % ("PASS" if all_ok else "FAIL", len(rows)),
          flush=True)
    print("L2-GATE-DONE", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
