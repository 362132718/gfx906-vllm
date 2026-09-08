# p0_launchcost.py -- marginal cost of one more kernel node inside a replayed
# CUDA graph. Most of the seven-ish candidate shapes need Split-K, so a
# two-phase reduction adds ~O(200) launches per token; this is the number that
# decides whether that is affordable or whether Split-K must stay persistent
# (last-block / single-wave reduction, no extra node).
#
# Slope method, not total/240: a graph replay has a fixed submission cost, and
# dividing a single replay by its node count hides it in the per-launch number.
# Capturing the same op at 1 / 60 / 240 nodes and fitting the slope removes it.
#
# Run inside the dev container, clocks pinned:
#   docker exec vllm-gfx906 python3 /work/p0/p0_launchcost.py
import os
import statistics

import torch

NODES = [int(x) for x in os.environ.get("NODES", "1,60,240").split(",")]


def timed(fn, iters=80):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1e3          # us


def graph_of(n, op):
    op()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            op()
    return timed(g.replay, iters=60) / n        # us per node (includes replay/n)


def main():
    one = torch.zeros(1, device="cuda")
    op = lambda: one.zero_()                    # noqa: E731  (tiny, real kernel)
    eager = timed(op)
    print("eager  single launch+sync: %.2f us -> %d/token = %.2f ms"
          % (eager, NODES[-1], eager * NODES[-1] / 1e3))
    pts = []
    for n in NODES:
        per = graph_of(n, op)
        pts.append((n, per))
        print("graph  n=%-4d total %7.2f us, %.3f us/node (replay cost included)"
              % (n, per * n, per), flush=True)
    (n1, p1), (n2, p2) = pts[0], pts[-1]
    fixed = p1 * n1                              # ~replay overhead
    slope = (p2 * n2 - p1 * n1) / (n2 - n1)      # us per extra node
    print("graph  replay overhead ~%.2f us ; marginal cost per node = %.3f us"
          % (fixed, slope))
    tot = slope * NODES[-1] / 1e3
    print("%d extra launches/token = %.2f ms/token -> verdict: %s"
          % (NODES[-1], tot,
             "two-phase reduce is affordable" if tot < 0.5
             else "persistent/last-block reduction is required for Split-K"))


if __name__ == "__main__":
    main()
