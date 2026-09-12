# p2_gate0.py -- #20-P2 Task 0 go/kill gate (G-P0): V3 multi-acc, both whitelisted
# families, splitK x UNROLL rescan.
#
# The timing instrument is copied from p1_l1_bench.main() (two rotated weight
# buffers, one persistent scratch, same timeit/graphed/R), NOT re-invented: the
# gates 60.8/57.0 us are 78.99/74.14 divided by 1.3 and those two numbers were
# read with THAT ruler. Anything else measures a different unit.
#
# Only VARIANT==3 is checked or timed -- p1_kernels.cu's multi-acc commit removed
# V1's per-word debias, so V1/V2 are numerically dead by design (see the NOTE in
# the .cu and P1 report v3: V1 kill, V2 lose).
import json, sys, time

sys.path.insert(0, "/work/p1")
import torch
import p1_l1_bench as b

DEV, G = b.DEV, b.G
GATES = {(5120, 8192): 60.8, (5120, 7168): 57.0}
ROWS_PATH = "/work/p1/p2_gate0_rows.json"
KBIT = b.KBIT
# incumbent_rot = the two-rotated-buffer re-measure in p1_gap_probe.json. That is
# what 78.99 / 74.14 (and hence the 60.8 / 57.0 gates) came from. p1_incumbent.json
# holds the OLDER single-buffer numbers (80.05 / 74.07) -- using it as the
# denominator silently inflates every speedup by ~1.3%.
INCUMBENT = json.load(open("/work/p1/p1_gap_probe.json"))["incumbent_rot"]


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
    """fp64 zero-freedom reference (same code as p1_review_fix.ref64_for, copied
    so this script does not import a module that runs a whole sweep)."""
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


def ulp_rms_fp64(variant, sk, u, K, N, seed=5):
    x, qw, sc, zr = b.tensors(K, N, seed)
    scratch = torch.zeros(sk * N, device=DEV, dtype=torch.float32)
    got = b.mod.p1_gemv(x, qw, sc, zr, scratch, variant, sk, u, True)
    ref16 = ref64_for(x, qw, sc, zr).half()
    ulp = fp16_ulp_dist(got, ref16).to(torch.float64)
    return (float(torch.sqrt((ulp ** 2).mean())), float(ulp.mean()), int(ulp.max()))


rows = []
# The plan pins sk in {4,8,16}, but the launcher TORCH_CHECKs that splitk divides
# the K/GROUP = 40 groups, so 16 is STRUCTURALLY IMPOSSIBLE for both whitelisted
# families (40 % 16 = 8; P1 hit the same and silently skipped it). The rescan the
# plan wants ("multi-acc may move the optimum off 8") is served by every divisor
# of 40, so that is the grid; override with `python3 p2_gate0.py 4,8 4,8,16`.
SKS = tuple(int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "2,4,5,8,10,20").split(","))
US = tuple(int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "4,8,16").split(","))
print("grid sk=%s u=%s (sk must divide 40 = K/GROUP)" % (SKS, US), flush=True)
for K, N in ((5120, 8192), (5120, 7168)):
    inc = INCUMBENT["%d,%d" % (K, N)]["graph_us"]
    print("== K=%d N=%d incumbent-rot graph=%.2fus gate=%.1fus (=%.2fx) =="
          % (K, N, inc, GATES[(K, N)], inc / GATES[(K, N)]), flush=True)
    bufs = [b.tensors(K, N, seed=1), b.tensors(K, N, seed=2)]
    for sk in SKS:
        if (K // G) % sk:
            print("   skip sk=%d: must divide the %d groups of K=%d"
                  % (sk, K // G, K), flush=True)
            continue
        for u in US:
            scratch = torch.zeros(sk * N, device=DEV, dtype=torch.float32)
            try:
                ok = b.check(3, sk, u, K, N)
                rms, mean, mx = ulp_rms_fp64(3, sk, u, K, N)
                state = [0]

                def fn():
                    t = bufs[state[0] % 2]
                    state[0] += 1
                    return b.mod.p1_gemv(t[0], t[1], t[2], t[3], scratch, 3, sk, u, True)

                eager, graph = b.timeit(fn), b.timeit(b.graphed(fn))
            except Exception as ex:
                print("   skip sk=%d u=%d: %s" % (sk, u, str(ex)[:90]), flush=True)
                continue
            r = dict(K=K, N=N, variant=3, splitk=sk, unroll=u, wiring=bool(ok),
                     ulp_rms=round(rms, 4), ulp_mean=round(mean, 4), ulp_max=mx,
                     eager_us=round(eager, 2), graph_us=round(graph, 2),
                     speedup=round(inc / graph, 3))
            rows.append(r)
            print("   sk=%-2d u=%-2d wiring=%s ULP rms=%.4f max=%d | eager=%.2f "
                  "graph=%.2f us (%.3fx inc, gate %.1f) %s"
                  % (sk, u, "ok" if ok else "BAD", rms, mx, eager, graph,
                     inc / graph, GATES[(K, N)],
                     "GATE-OK" if ok and rms <= 0.05 and graph <= GATES[(K, N)]
                     else ""), flush=True)
            json.dump(rows, open(ROWS_PATH, "w"), indent=1)
    print("   temps=%s clocks=%s" % (b.temps_c(), b.clock_status()), flush=True)
    time.sleep(10)

json.dump(rows, open(ROWS_PATH, "w"), indent=1)
print("\n=== VERDICT ===", flush=True)
for (K, N), gate in GATES.items():
    ok_rows = [r for r in rows if r["K"] == K and r["N"] == N
               and r["wiring"] and r["ulp_rms"] <= 0.05]
    best = min(ok_rows, key=lambda r: r["graph_us"], default=None)
    print("GATE K=%d N=%d best=%s gate<=%.1f -> %s" % (
        K, N, best, gate, "PASS" if best and best["graph_us"] <= gate else "FAIL"),
        flush=True)
print("GATE0-DONE", flush=True)
