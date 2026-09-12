# p2_prod_ab.py -- is the production .so really 3.8% slower than the p1_kernels.cu
# prototype? G-P1 arm [4] proved the two are BIT-IDENTICAL at sk=10/u8, so any
# timing difference is codegen (standalone `hipcc -O3` vs torch `load_inline`) or
# run-to-run environment -- not algorithm. It matters: +3.8% on both families
# turns Task 0's 1 142.1 us aggregate gain into ~997 us, which changes the
# promised tok/s.
#
# Method: ONE process, ONE weight buffer shared by both arms (no rotation --
# rotation only decides which buffer the captured graph points at, and the two
# arms must point at the same one), capture each arm's own graph, then time them
# ALTERNATELY in several rounds so slow thermal/clock drift hits both arms
# equally and cancels in the per-round difference.
import sys

sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l1_bench as b
import p2_shim

DEV = b.DEV
SK, U = 10, 8
ROUNDS = 6


def capture(fn):
    fn()
    fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    torch.cuda.synchronize()
    return gr


for K, N in ((5120, 8192), (5120, 7168)):
    x, qw, sc, zr = b.tensors(K, N, seed=1)
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    scratch = torch.zeros(SK * N, device=DEV, dtype=torch.float32)

    gp = capture(lambda: b.mod.p1_gemv(x, qw, sc, zr, scratch, 3, SK, U, True))
    gq = capture(lambda: p2_shim.p2_gemv(x, qw, sc, zr, part, out, K, N, SK))
    print("== K=%d N=%d  prototype vs production, alternating %d rounds ==" % (
        K, N, ROUNDS), flush=True)
    dp, dq = [], []
    for r in range(ROUNDS):
        tp = b.timeit(gp.replay, warm=8, reps=32, R=24)
        tq = b.timeit(gq.replay, warm=8, reps=32, R=24)
        dp.append(tp)
        dq.append(tq)
        print("   round %d proto=%.2f prod=%.2f delta=%+.2f us (%+.2f%%) temps=%s"
              % (r, tp, tq, tq - tp, 100 * (tq / tp - 1), b.temps_c()), flush=True)
    mp = sorted(dp)[len(dp) // 2]
    mq = sorted(dq)[len(dq) // 2]
    print("   MEDIAN proto=%.2f prod=%.2f -> %+.2f us (%+.2f%%), prod wins %d/%d"
          % (mp, mq, mq - mp, 100 * (mq / mp - 1),
             sum(1 for a, c in zip(dp, dq) if c < a), ROUNDS), flush=True)
print("AB-DONE", flush=True)
