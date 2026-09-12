# p2_prod_gate.py -- #20-P2 Task 1 / gate G-P1: the production .so gated on its
# own, before any vLLM code is touched.
#
# Four arms (plan Task 1 Step 3):
#   [1] WIRING   16 one-hot probes, worst |got-want| <= 5e-4 absolute.
#   [2] ULP      fp64 zero-freedom reference, RMS <= 0.05 AND within 2x of Task 0.
#                "within 2x of Task 0" is the regression gate: a compiler flag
#                change must not move the numbers, so a bare <=0.05 pass would
#                also accept a kernel that silently drifted inside the band.
#   [3] TIMING   inside a CUDA graph, compared to Task 0's same-tier measurement
#                (+-20%). Capture VALIDITY is itself a test item: a ctypes launch
#                that never entered the graph still "times fast" on replay, so the
#                buffer is poisoned after capture and the replay must rewrite it.
#   [4] BIT-EQUAL production .so vs the p1_kernels.cu V3 prototype, same x, same
#                weights, same sk. ULP bands on both sides can hide a silent
#                systematic difference; torch.equal cannot.
#
# The timing instrument is copied from p2_gate0.py / p1_l1_bench.main() (two
# rotated weight buffers, one persistent scratch, timeit(graphed(fn)), warm16 /
# reps64 / R24, median). Re-inventing it would measure a different unit.
import json
import sys
import time

sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l1_bench as b
import p2_shim

DEV, G = b.DEV, b.G
SK, U = 10, 8                     # production tier; U only pins the prototype arm
FAMS = ((5120, 8192), (5120, 7168))
# Task 0, single accumulator, sk=10, u=8 -- logs/p2/p2_gate0_single_acc.log
TASK0 = {(5120, 8192): dict(graph_us=60.53, ulp_rms=0.0292),
         (5120, 7168): dict(graph_us=58.22, ulp_rms=0.0205)}
WIRE_GATE, ULP_GATE, TIME_TOL, ULP_REGRESS = 5e-4, 0.05, 0.20, 2.0
ROWS_PATH = "/work/p2/p2_prod_gate_rows.json"
KBIT = b.KBIT


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
    ai = a16.view(torch.int16).to(torch.int32)
    bi = b16.view(torch.int16).to(torch.int32)
    am = torch.where(ai >= 0, ai, -(ai & 0x7FFF))
    bm = torch.where(bi >= 0, bi, -(bi & 0x7FFF))
    return (am - bm).abs()


def prod(x, qw, sc, zr, part, out, K, N):
    p2_shim.p2_gemv(x, qw, sc, zr, part, out, K, N, SK)
    return out


def arm_wiring(K, N):
    """one-hot x = e_k: any (k,n) mis-pairing shows up at 100% scale."""
    x, qw, sc, zr = b.tensors(K, N, seed=5)
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    code, zp = unpack64(qw), unpack_zp64(zr, N)
    worst = 0.0
    for k in range(0, K, max(1, K // 16)):
        x.zero_()
        x[k] = 1.0
        prod(x, qw, sc, zr, part, out, K, N)
        want = (code[k].to(torch.float32) - zp[k // G].to(torch.float32)) \
            * sc[k // G].to(torch.float32)
        worst = max(worst, float((out.float() - want).abs().max()))
    return worst


def arm_ulp(K, N):
    x, qw, sc, zr = b.tensors(K, N, seed=5)
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    prod(x, qw, sc, zr, part, out, K, N)
    ref16 = ref64_for(x, qw, sc, zr).half()
    ulp = fp16_ulp_dist(out, ref16).to(torch.float64)
    return float(torch.sqrt((ulp ** 2).mean())), float(ulp.mean()), int(ulp.max())


def arm_timing_and_capture(K, N):
    bufs = [b.tensors(K, N, seed=1), b.tensors(K, N, seed=2)]
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    state = [0]

    def fn():
        t = bufs[state[0] % 2]
        state[0] += 1
        prod(t[0], t[1], t[2], t[3], part, out, K, N)

    eager = b.timeit(fn)
    fn()                                     # eager warm-up: code object loads here
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    gr.replay()
    torch.cuda.synchronize()
    a = out.clone()
    out.zero_()                              # poison: the falsifiable capture test
    gr.replay()
    torch.cuda.synchronize()
    rewrote = not bool((out == 0).all())
    replay_stable = bool(torch.equal(a, out.clone()))
    graph = b.timeit(lambda: gr.replay())
    del gr
    return eager, graph, rewrote, replay_stable


def arm_bit_compare(K, N):
    x, qw, sc, zr = b.tensors(K, N, seed=5)
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    prod(x, qw, sc, zr, part, out, K, N)
    proto = b.mod.p1_gemv(x, qw, sc, zr, part, 3, SK, U, True)
    eq = bool(torch.equal(out, proto))
    if eq:
        return eq, 0, 0.0
    d = fp16_ulp_dist(out, proto).to(torch.float64)
    return eq, int((d > 0).sum()), float(d.max())


def main():
    print("P2 production gate (G-P1): sk=%d lib=%s" % (SK, p2_shim._LIB or "lazy"),
          flush=True)
    rows = []
    for K, N in FAMS:
        t0 = TASK0[(K, N)]
        print("== K=%d N=%d  Task0 ref graph=%.2fus ulp_rms=%.4f =="
              % (K, N, t0["graph_us"], t0["ulp_rms"]), flush=True)
        wire = arm_wiring(K, N)
        rms, mean, mx = arm_ulp(K, N)
        eager, graph, rewrote, stable = arm_timing_and_capture(K, N)
        eq, ndiff, maxd = arm_bit_compare(K, N)
        drift = abs(graph - t0["graph_us"]) / t0["graph_us"]
        r = dict(K=K, N=N, splitk=SK, wire=round(wire, 7), ulp_rms=round(rms, 4),
                 ulp_mean=round(mean, 4), ulp_max=mx, eager_us=round(eager, 2),
                 graph_us=round(graph, 2), task0_us=t0["graph_us"],
                 drift=round(drift, 4), rewrote_poison=rewrote,
                 replay_stable=stable, bit_equal=eq, n_diff=ndiff,
                 max_diff_ulp=maxd)
        rows.append(r)
        json.dump(rows, open(ROWS_PATH, "w"), indent=1)
        print("   [1] wiring  worst=%.2e (gate <=%.0e) %s"
              % (wire, WIRE_GATE, "ok" if wire <= WIRE_GATE else "BAD"), flush=True)
        print("   [2] fp64    RMS=%.4f MEAN=%.4f MAX=%d (gate <=%.2f, Task0 %.4f, "
              "<=2x) %s" % (rms, mean, mx, ULP_GATE, t0["ulp_rms"],
                            "ok" if rms <= ULP_GATE and rms <= ULP_REGRESS * t0["ulp_rms"]
                            else "BAD"), flush=True)
        print("   [3] graph   eager=%.2f graph=%.2f us vs Task0 %.2f (%+.1f%%, tol %d%%)"
              " | replay rewrote poison=%s replay_bitstable=%s %s"
              % (eager, graph, t0["graph_us"], 100 * (graph / t0["graph_us"] - 1),
                 int(100 * TIME_TOL), rewrote, stable,
                 "ok" if drift <= TIME_TOL and rewrote and stable else "BAD"),
              flush=True)
        print("   [4] vs proto sk=%d u=%d bit-equal=%s n_diff=%d max_dist=%.3f ULP"
              % (SK, U, eq, ndiff, maxd), flush=True)
        print("   temps=%s clocks=%s" % (b.temps_c(), b.clock_status()), flush=True)
        time.sleep(10)

    print("\n=== G-P1 VERDICT ===", flush=True)
    all_ok = True
    for r in rows:
        t0 = TASK0[(r["K"], r["N"])]
        ok = (r["wire"] <= WIRE_GATE
              and r["ulp_rms"] <= ULP_GATE and r["ulp_rms"] <= ULP_REGRESS * t0["ulp_rms"]
              and r["drift"] <= TIME_TOL and r["rewrote_poison"]
              and r["replay_stable"])
        all_ok = all_ok and ok
        print("K=%d N=%d -> %s (bit_equal=%s)"
              % (r["K"], r["N"], "PASS" if ok else "FAIL", r["bit_equal"]), flush=True)
    print("G-P1 %s" % ("PASS" if all_ok else "FAIL"), flush=True)
    print("PROD-GATE-DONE", flush=True)


if __name__ == "__main__":
    main()
