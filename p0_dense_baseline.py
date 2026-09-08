# p0_dense_baseline.py -- M=1 timing of the DENSE fp16 GEMMs that the Task 3 trace
# put at 49 launches/step (48 x 318 us = 15.26 ms/step, 1 x 6982 us = 6.98 ms/step)
# -- i.e. the matmul work in a decode step that is NOT the W4A16 GEMV.
#
# Why this exists: the only language-model fp16 weight replicated 48x is the GDN
# in_proj_ba (K=5120, N=2*num_v_heads=96), whose whole weight is <1 MB per rank.
# 318 us for <1 MB is 3 GB/s, two orders of magnitude under the read floor, which
# means either my identification is wrong or hipBLASLt picks a pathological kernel
# for M=1. And lm_head (248320 x 5120 vocab-parallel -> 1.27 GB/rank in fp16) at
# 6.98 ms is 182 GB/s = 22% of floor. Both would be wins that need no hand-written
# W4A16 kernel, so the P0 gate cannot be called without settling them.
#
# Three formulations of the same product, to separate "shape is hard" from
# "hipBLASLt chose badly" and from "storage layout matters on gfx906":
#   mm_KN    x[1,K] @ w[K,N]            what LinearBase does today
#   mm_NK    x[1,K] @ w'[N,K].t()       same math, transposed storage
#   mv_NK    mv(w'[N,K], x[K])          GEMV formulation, no BLAS at all
# each compared against the pure-read floor (weight bytes / BW_eff).
#
#   docker exec -e BW_EFF=821 vllm-gfx906 python3 /work/p0/p0_dense_baseline.py
import json
import os
import time

import torch

BW_EFF = float(os.environ.get("BW_EFF", "0"))       # GB/s, from Task 1
DTYPE = getattr(torch, os.environ.get("DTYPE", "float16"))
# (name, K, N, launches_per_step) -- N per rank after TP=2.
SHAPES = json.loads(os.environ.get("SHAPES", """[
  ["gdn_in_proj_ba", 5120, 96, 48],
  ["gdn_in_proj_a",  5120, 48, 48],
  ["lm_head",        5120, 124160, 1]]"""))
OUT = "/work/p0/p0_dense.json"


def timed(fn, iters=60):
    """(eager us, in-graph us) per call; the graph number is what the engine pays."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    eager = (time.perf_counter() - t0) / iters * 1e6
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize()
    return eager, (time.perf_counter() - t0) / 200 * 1e6


def main():
    if BW_EFF <= 0:
        raise SystemExit("FATAL: BW_EFF unset -- the efficiency column is the "
                         "whole point of this run")
    torch.cuda.set_device(0)
    print("BW_eff = %.0f GB/s, dtype=%s, device=%s"
          % (BW_EFF, DTYPE, torch.cuda.get_device_name(0)), flush=True)
    rows = []
    for name, K, N, nls in SHAPES:
        w = torch.randn(K, N, device="cuda", dtype=DTYPE) * 0.02
        wt = w.t().contiguous()                 # [N,K] -- what a GEMV wants
        x = torch.randn(1, K, device="cuda", dtype=DTYPE) * 0.05
        out = torch.empty(1, N, device="cuda", dtype=DTYPE)
        bytes_w = K * N * 2
        print("\n%-16s K=%-6d N=%-7d x%-3d/step  weight=%.2f MB/rank"
              % (name, K, N, nls, bytes_w / 1e6), flush=True)
        for form, fn in (
                ("mm_KN", lambda: torch.matmul(x, w, out=out)),
                ("mm_NK", lambda: torch.matmul(x, wt.t(), out=out)),
                ("mv_NK", lambda: torch.mv(wt, x.view(K), out=out.view(N))),
                ("mulsum", (lambda: torch.sum(wt * x, dim=1, out=out.view(N)))
                           if K * N * 4 < 2e8 else None)):
            if fn is None:
                print("   %-7s skipped (product temp too large)" % form, flush=True)
                continue
            try:
                eager, graph = timed(fn)
            except Exception as e:                                        # noqa: BLE001
                print("   %-7s FAILED %s" % (form, str(e)[:90]), flush=True)
                continue
            gbps = bytes_w / 1e9 / (graph * 1e-6)
            floor = bytes_w / (BW_EFF * 1e9) * 1e6
            print("   %-7s eager=%8.1f graph=%8.1f us  %7.1f GB/s = %5.1f%% of"
                  " floor (%.1f us)  -> %7.2f ms/step"
                  % (form, eager, graph, gbps, floor / graph * 100, floor,
                     graph * nls / 1e3), flush=True)
            rows.append(dict(name=name, form=form, K=K, N=N, launches=nls,
                             eager_us=eager, graph_us=graph, gbps=gbps,
                             floor_us=floor, pct_of_floor=floor / graph * 100,
                             ms_per_step=graph * nls / 1e3))
        del w, wt, x, out
        torch.cuda.empty_cache()
    json.dump({"bw_eff_gbps": BW_EFF, "dtype": str(DTYPE), "rows": rows},
              open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT)
    print("trace said 48/step @ 318 us and 1/step @ 6982 us. A form that reaches"
          " the floor on the same operands is a drop-in; the gap between mm_KN and"
          " the best form is time that can be reclaimed without touching W4A16.")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
