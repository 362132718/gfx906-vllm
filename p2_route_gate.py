# p2_route_gate.py -- #20-P2 Task 2 / gate G-P2: the routing matrix, asserted on
# the REAL kernel object before any engine is started.
#
# Rows (plan Task 2 Step 2 + Step 3):
#   [A] PROBE      _P2_OK / the registered op / _p2_part allocated on whitelisted
#                 shapes and skipped on the others. NOT a routing row: without it
#                 every negative row below passes VACUOUSLY. A silently missing
#                 .so would turn "P2 never fired" into a green test of nothing.
#   [B] OP PATH    M=1 whitelisted: apply_weights output bit-equal to a direct
#                 ctypes call on the same tensors, and bit-stable across repeats.
#                 What this row is FOR: the silent mutation class. Swapping w_s
#                 and w_zp (mutation M1) is NOT silent -- the kernel reads the
#                 163 KB qzeros as a 655 KB scales buffer and dies on a GPU
#                 memory access fault, so that bug cannot reach a numeric band at
#                 all. The dangerous ones stay in bounds, e.g. passing K//2 as K
#                 (mutation M1'): half the codes are then never read and the
#                 output still looks like a plausible dot product. M1' turns
#                 exactly this row red (n_diff 8190/8192, max|d| 1.4e+01).
#   [C] M=2        P2 must not fire, and the output must stay inside the
#                 incumbent's own noise band vs (i) the switch-off call and
#                 (ii) a direct gptq_gemm_out call.
#   [D] OFF-LIST   three non-whitelisted shapes x M=1: _p2_part is None AND zero
#                 hits AND output inside the incumbent band.
#   [E] KILL-SWITCH VLLM_WNA16_P2_GEMV unset/0/1: fires on the default and on
#                 "1", never on "0", and switching really changes kernel.
#   [F] PREDICATE  five shapes x M=1 in one sweep => hit count 1 for each
#                 whitelisted family, 0 for the other three.
#   [G] CAPTURE    one captured whitelisted call + two replays, bit-stable, equal
#                 to the eager result, AND a poisoned output must be rewritten by
#                 replay -- the falsifiable half, since a launch that leaked out
#                 of the graph also "replays bit-stably", on stale data.
#
# WHY THE INCUMBENT ROWS USE AN ABSOLUTE BAND AND NOT BIT EQUALITY
# (p2_repro_discriminator.py, p2_incumbent_determinism.py, both 2026-09-09):
# gptq_gemm_out is NOT reproducible run to run on gfx906. Six identical calls
# differ on ~71% of positions, max|d| = 3.1e-2..6.3e-2 absolute on outputs of
# |max| ~ 17-22, and that holds for the unpatched entry point called directly --
# it is the fused kernel's fp16 atomicAdd over the split-K partials, in an order
# that varies with scheduling. Its own error against a zero-freedom fp64
# reference is the same size (max 2.6e-2..5.5e-2), so the spread is the kernel's
# noise floor, not a discrepancy. Two consequences, both measured:
#   * a monotone bit-index "ULP" metric is USELESS here: it explodes in the
#     cancellation region (max 1.1e4-1.5e4 for a 4e-2 absolute difference).
#   * comparing a tensor to a VIEW of the buffer it was written into is vacuous
#     (that is what made an earlier probe claim "same slab => bit reproducible").
#     Everything compared below is a .clone().
# Band = max(6 fp16 ULP of the |out| binade, 2x the incumbent's own measured
# run-to-run spread in this process). A branch mis-routing changes the result by
# O(|out|) ~ 1-20, i.e. 2-3 orders above this band, so the test keeps its teeth.
import math
import os
import sys

os.environ["VLLM_WNA16_EXLLAMA"] = "1"          # can_implement() requires it
sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch
from vllm import _custom_ops as ops

import p1_l0_mapping as l0
import p2_shim
from vllm.model_executor.kernels.linear.mixed_precision import (
    wna16_exllama as W)

DEV, SK = "cuda", 10
ON = ((5120, 8192), (5120, 7168))                # whitelisted families
NON = ((5120, 17408), (5120, 8704), (5120, 3072))
THEORY_ULPS = 6.0          # fp16 reordering slack on the dominant partial
EMPIRICAL_X = 2.0          # x the incumbent's own measured spread
X = 7                       # seed for every x in this gate


def mk(K, N, seed=0):
    return l0.make_layer_and_kernel(K, N, seed)[:2]


def setenv(v):
    if v is None:
        os.environ.pop("VLLM_WNA16_P2_GEMV", None)
    else:
        os.environ["VLLM_WNA16_P2_GEMV"] = v


def run(kern, layer, x):
    return kern.apply_weights(layer, x).clone()


def actx(K, M, seed=X):
    torch.manual_seed(seed)
    return torch.randn(M, K, device=DEV, dtype=torch.float16)


def adiff(a, b):
    d = (a.float() - b.float()).abs()
    return int((d > 0).sum()), float(d.max())


def ulp_at(v):
    """fp16 ULP of the binade containing |v| (10 explicit mantissa bits)."""
    e = math.floor(math.log2(max(abs(float(v)), 1e-6)))
    return 2.0 ** (e - 10)


def noise(kern, layer, x, reps=3):
    """The incumbent's own run-to-run spread, measured with P2 off."""
    setenv("0")
    rs = [run(kern, layer, x) for _ in range(reps)]
    setenv(None)
    return max(adiff(rs[0], r)[1] for r in rs[1:])


def band(ref, sigma):
    scale = float(ref.float().abs().max())
    return max(THEORY_ULPS * ulp_at(scale), EMPIRICAL_X * sigma)


def row_A():
    ok = [("probe _P2_OK", W._P2_OK, "loader found p2_gemv"),
          ("ops registered", hasattr(torch.ops.vllm, "p2_w4a16_gemm")
           and hasattr(torch.ops.vllm, "p2_gemv_or_fused"),
           "p2_w4a16_gemm + p2_gemv_or_fused (the latter is what apply_weights"
           " calls since the G-P3 fix)")]
    for K, N in ON:
        kern, layer = mk(K, N)
        good = (kern._p2_part is not None
                and tuple(kern._p2_part.shape) == (SK, N)
                and kern._p2_part.dtype == torch.float32
                and kern._p2_route is True)   # the flag apply_weights now reads
        ok.append(("part allocated %d/%d" % (K, N), good,
                   "None" if kern._p2_part is None else
                   tuple(kern._p2_part.shape)))
    for K, N in NON:
        kern, layer = mk(K, N)
        ok.append(("part skipped %d/%d" % (K, N),
                   kern._p2_part is None and kern._p2_route is False,
                   "None" if kern._p2_part is None else
                   tuple(kern._p2_part.shape)))
    return ok


def row_B():
    ok = []
    for K, N in ON:
        kern, layer = mk(K, N)
        x = actx(K, 1)
        W._P2_HITS.clear()
        got = run(kern, layer, x)
        n_hits = len(W._P2_HITS)
        # independent path: raw ctypes, its own scratch and output buffers
        w_q, w_s, w_zp, _ = kern._get_weight_params(layer)
        part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
        ref = torch.zeros(1, N, device=DEV, dtype=torch.float16)
        p2_shim.p2_gemv(x, w_q, w_s, w_zp, part, ref, K, N, SK)
        torch.cuda.synchronize()
        eq = bool(torch.equal(got, ref))
        nd, mx = adiff(got, ref)
        ok.append(("op==ctypes bit-equal %d/%d" % (K, N),
                   eq and n_hits == 1 and W._P2_HITS[0] == (K, N),
                   "equal=%s n_diff=%d max=%.1e hits=%d last=%s"
                   % (eq, nd, mx, n_hits, W._P2_HITS[-1] if n_hits else None)))
        # P2 has no atomicAdd: unlike the incumbent it must be bit-stable.
        rs = [run(kern, layer, x) for _ in range(4)]
        st = all(bool(torch.equal(r, got)) for r in rs)
        ok.append(("P2 bit-stable x4 %d/%d" % (K, N), st,
                   "n_diff=%s" % [adiff(got, r)[0] for r in rs]))
    return ok


def row_C():
    ok = []
    K, N = ON[0]
    kern, layer = mk(K, N)
    x = actx(K, 2)
    sigma = noise(kern, layer, x)
    W._P2_HITS.clear()
    a = run(kern, layer, x)
    hits_on = len(W._P2_HITS)
    tol = band(a, sigma)
    setenv("0")
    b = run(kern, layer, x)
    setenv(None)
    nd, mx = adiff(a, b)
    ok.append(("M=2 no P2 hit", hits_on == 0, "hits=%d" % hits_on))
    ok.append(("M=2 in incumbent band", mx <= tol,
               "max|d|=%.3e <= tol=%.3e (sigma_inc=%.3e, tol=%.1f ULP@|max|=%.1f)"
               % (mx, tol, sigma, tol / ulp_at(float(a.float().abs().max())),
                  float(a.float().abs().max()))))
    # the incumbent really ran: gptq_gemm_out called directly, fresh buffer
    w_q, w_s, w_zp, g_idx = kern._get_weight_params(layer)
    bits = kern.config.weight_type.size_bits
    tdq = torch.empty(0, dtype=torch.float16, device=DEV)
    direct = torch.zeros(2, N, device=DEV, dtype=torch.float16)
    ops.gptq_gemm_out(direct, x, w_q, w_zp, w_s, g_idx, True, True, bits, tdq)
    torch.cuda.synchronize()
    nd2, mx2 = adiff(b, direct)
    ok.append(("M=2 == gptq_gemm_out", mx2 <= tol,
               "n_diff=%d max|d|=%.3e <= tol=%.3e" % (nd2, mx2, tol)))
    return ok


def row_D():
    ok = []
    for K, N in NON:
        kern, layer = mk(K, N)
        x = actx(K, 1)
        sigma = noise(kern, layer, x)
        W._P2_HITS.clear()
        a = run(kern, layer, x)
        hits = len(W._P2_HITS)
        setenv("0")
        b = run(kern, layer, x)
        setenv(None)
        nd, mx = adiff(a, b)
        tol = band(a, sigma)
        ok.append(("%d/%d off-list skipped" % (K, N),
                   kern._p2_part is None and hits == 0
                   and bool(torch.isfinite(a).all()),
                   "part=%s hits=%d" % (kern._p2_part is not None, hits)))
        ok.append(("%d/%d unchanged vs off" % (K, N), mx <= tol,
                   "n_diff=%d/%d max|d|=%.3e <= tol=%.3e (sigma_inc=%.3e)"
                   % (nd, a.numel(), mx, tol, sigma)))
    return ok


def row_E():
    ok = []
    K, N = ON[0]
    kern, layer = mk(K, N)
    x = actx(K, 1)
    W._P2_HITS.clear()
    setenv("0")
    off = run(kern, layer, x)
    hits_off = len(W._P2_HITS)
    setenv(None)
    on = run(kern, layer, x)
    hits_on = len(W._P2_HITS)
    setenv("1")
    on2 = run(kern, layer, x)
    hits_on2 = len(W._P2_HITS)
    setenv(None)
    ok.append(("kill-switch: no hits", hits_off == 0 and hits_on == 1
               and hits_on2 == 2, "off=%d default=%d on=%d"
               % (hits_off, hits_on, hits_on2)))
    # The switched-off call is the incumbent. P2 is bit-stable (row B), so
    # "not bit-equal" plus the hit log can only mean a different kernel ran --
    # no magnitude threshold needed, and one would be wrong anyway: P2's distance
    # from the incumbent is just the incumbent's own ~5e-2 fp16 error.
    nd, mx = adiff(off, on)
    ok.append(("switch changes kernel", nd > 0,
               "n_diff=%d/%d max|d|=%.3e" % (nd, on.numel(), mx)))
    ok.append(("switch output finite", bool(torch.isfinite(off).all())
               and bool(torch.isfinite(on).all()), ""))
    return ok


def row_F():
    W._P2_HITS.clear()
    for K, N in ON + NON:
        kern, layer = mk(K, N)
        run(kern, layer, actx(K, 1))
    want = {kn: (1 if kn in ON else 0) for kn in ON + NON}
    got = {kn: sum(1 for h in W._P2_HITS if h == kn) for kn in ON + NON}
    return [("five shapes hit-count exact",
             got == want and len(W._P2_HITS) == 2,
             "want=%s got=%s" % (want, got))]


def row_G():
    ok = []
    for K, N in ON:
        kern, layer = mk(K, N)
        x = actx(K, 1)
        W._P2_HITS.clear()
        eager = run(kern, layer, x)
        assert len(W._P2_HITS) == 1
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = kern.apply_weights(layer, x)
        hits_during_capture = len(W._P2_HITS)
        gr.replay()
        torch.cuda.synchronize()
        a = out.clone()
        gr.replay()
        torch.cuda.synchronize()
        b = out.clone()
        stable = bool(torch.equal(a, b))
        # Poison BOTH the slab and whatever the op handed back: if
        # functionalization returned a copy in the graph's private pool, zeroing
        # only the slab would leave the copy non-zero and the check would pass
        # vacuously.
        kern._cbuf.zero_()
        out.zero_()
        alias = out.data_ptr() == kern._cbuf.data_ptr()
        gr.replay()
        torch.cuda.synchronize()
        rewrote = not bool((out == 0).all())
        matches_eager = bool(torch.equal(a, eager))
        del gr
        ok.append(("capture %d/%d" % (K, N),
                   stable and rewrote and matches_eager
                   and hits_during_capture == 2,
                   "replay_stable=%s rewrote_poison=%s ==eager=%s "
                   "hits_in_capture=%d aliases_slab=%s"
                   % (stable, rewrote, matches_eager, hits_during_capture,
                      alias)))
    return ok


def main():
    print("P2 route gate (G-P2): _P2_OK=%s lib=%s splitk=%d shapes=%s seed=%d"
          % (W._P2_OK, W._P2_LIB is not None, W._P2_SPLITK,
             sorted(W._P2_SHAPES), X), flush=True)
    if not W._P2_OK:
        print("ABORT: the .so is not loaded; every negative row below would be "
              "vacuous. Fix the loader before reading this gate.", flush=True)
        return 1
    all_ok = True
    for name, fn in (("A probe", row_A), ("B op==ctypes", row_B),
                     ("C M=2", row_C), ("D off-list", row_D),
                     ("E kill-switch", row_E), ("F predicate", row_F),
                     ("G capture", row_G)):
        print("== row %s ==" % name, flush=True)
        for label, passed, detail in fn():
            all_ok = all_ok and passed
            print("   %-34s %-4s %s" % (label, "ok" if passed else "BAD",
                                         detail), flush=True)
        setenv(None)
    print("\nROUTE %s" % ("ok" if all_ok else "BAD"), flush=True)
    print("G-P2 %s" % ("PASS" if all_ok else "FAIL"), flush=True)
    print("ROUTE-GATE-DONE", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
