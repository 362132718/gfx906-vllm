# p2_gate_time_contamination.py -- why does G-P1's arm [3] report 7168 at 58.85 us
# when Task 0's sweep (p2_gate0.py) and the A/B scripts (p2_unroll_ab.py /
# p2_body_ab.py) both put the SAME production code at 58.06-58.22?
#
# The two candidate explanations are not the same finding, and the 99% cumulative
# adjudication line is 1148.7*0.99 = 1137.2 us while the gate's own numbers land
# at 1136.3 -- so this 0.7 us decides whether Task 1 reports a pass or a miss.
#
#   H1 process state: arm [1] (one-hot wiring) and arm [2] (fp64 reference) allocate
#       hundreds of MB of int64/fp64 before arm [3] times anything, so the weight
#       buffers land at different addresses / the allocator is fragmented.
#   H2 code path: the ctypes .so really is slower than the load_inline prototype and
#       one of the three harnesses is wrong.
#
# Discriminator: time BOTH code paths in the SAME process, alternating rounds (the
# order is rotated per round so a monotonic thermal drift cannot favour one arm),
# once clean (nothing but the timing arm has run) and once after the heavy arms.
#   H1 -> prod and proto both shift up together after the arms.
#   H2 -> prod stays above proto in both stages.
import statistics
import sys

sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l1_bench as b
import p2_prod_gate as pg

SK, U = pg.SK, pg.U
ROUNDS = 5


def pair(K, N):
    """One graph-timed measurement of each code path, same buffers, same ruler."""
    bufs = [b.tensors(K, N, seed=1), b.tensors(K, N, seed=2)]
    scratch = torch.zeros(SK * N, device=b.DEV, dtype=torch.float32)
    part = torch.zeros((SK, N), device=b.DEV, dtype=torch.float32)
    out = torch.zeros(N, device=b.DEV, dtype=torch.float16)
    graphs = {}
    for name in ("prod", "proto"):
        state = [0]

        def fn():
            t = bufs[state[0] % 2]
            state[0] += 1
            if name == "prod":
                pg.prod(t[0], t[1], t[2], t[3], part, out, K, N)
            else:
                b.mod.p1_gemv(t[0], t[1], t[2], t[3], scratch, 3, SK, U, True)

        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        gr.replay()
        torch.cuda.synchronize()
        graphs[name] = (gr, bufs)
    return graphs


def timed_pair(K, N, tag):
    graphs = pair(K, N)
    order = ["prod", "proto"]
    per = {k: [] for k in order}
    for r in range(ROUNDS):
        rot = order[r % 2:] + order[:r % 2]
        vals = {}
        for name in rot:
            gr = graphs[name][0]
            vals[name] = b.timeit(lambda gr=gr: gr.replay())
        for name in rot:
            per[name].append(vals[name])
        print("   %s r%d prod=%.2f proto=%.2f temps=%s"
              % (tag, r, vals["prod"], vals["proto"], b.temps_c()), flush=True)
    m = {k: statistics.median(v) for k, v in per.items()}
    print("   %s MEDIAN K=%d N=%d prod=%.2f proto=%.2f  prod-proto=%+.2f us"
          % (tag, K, N, m["prod"], m["proto"], m["prod"] - m["proto"]), flush=True)
    del graphs
    return m


for K, N in ((5120, 8192), (5120, 7168)):
    print("== K=%d N=%d ==" % (K, N), flush=True)
    clean = timed_pair(K, N, "CLEAN")
    wire = pg.arm_wiring(K, N)
    rms, _, _ = pg.arm_ulp(K, N)
    print("   (heavy arms done: wiring=%.2e ulp_rms=%.4f)" % (wire, rms), flush=True)
    post = timed_pair(K, N, "POSTARMS")
    print("   DELTA-after-arms prod=%+.2f proto=%+.2f us"
          % (post["prod"] - clean["prod"], post["proto"] - clean["proto"]), flush=True)
print("CONTAM-DONE", flush=True)
