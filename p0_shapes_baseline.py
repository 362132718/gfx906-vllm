# p0_shapes_baseline.py -- the deployed ExLlama fused kernel at M=1, per real
# per-rank shape, timed eager AND inside a captured graph, compared against the
# measured BW_eff floor (bytes the kernel actually pulls: 4-bit weight + fp16
# scales + packed zero points -- not K*N/2 alone).
#
# Shapes and per-step counts are the Task 2 measurement (p0_shapes.log eager
# census, confirmed by the p0_step_shapes.csv compiled-trace census), not the
# 7-shape guess in the plan draft.
#
# The timed unit is one full bridge call: c_out.zero_() + ops.gptq_gemm_out(...).
# The fused kernel accumulates with atomicAdd, so zero_() is not optional -- it
# is part of the cost a hand-written kernel must beat.
#
# Run inside the dev container with the clocks pinned (same state as Task 1):
#   ./p0_clock.sh lock
#   docker exec -e BW_EFF=821 vllm-gfx906 python3 /work/p0/p0_shapes_baseline.py
import json
import os
import statistics

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
import torch                                        # noqa: E402
from vllm import _custom_ops as ops                 # noqa: E402

BW_EFF = float(os.environ.get("BW_EFF", "0"))       # GB/s, from Task 1
GROUP, BITS = 128, 4
OUT = os.environ.get("OUT", "/work/p0")
# (K, N, calls-per-decode-step-per-rank)
SHAPES = [(3072, 5120, 64), (5120, 7168, 16), (5120, 8192, 48),
          (5120, 17408, 64), (8704, 5120, 64)]
CALLS = sum(c for _, _, c in SHAPES)


def make(K, N, dev):
    return dict(
        x=torch.randn(1, K, dtype=torch.float16, device=dev),
        wq=torch.randint(-2**30, 2**30, (K // 8, N), dtype=torch.int32,
                         device=dev),
        wz=torch.randint(0, 2**30, (K // GROUP, N // 8), dtype=torch.int32,
                         device=dev),
        ws=torch.randn(K // GROUP, N, dtype=torch.float16, device=dev),
        gi=torch.empty(0, dtype=torch.int32, device=dev),
        c=torch.empty(1, N, dtype=torch.float16, device=dev),
        td=torch.empty(0, dtype=torch.float16, device=dev),
    )


def call(t):
    # Exactly what WNA16ExllamaKernel.apply_weights does at M=1.
    t["c"].zero_()
    ops.gptq_gemm_out(t["c"], t["x"], t["wq"], t["wz"], t["ws"], t["gi"],
                      True, True, BITS, t["td"])


def timed(fn, iters=60):
    for _ in range(16):
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
    return statistics.median(ts)


def main():
    if BW_EFF <= 0:
        raise SystemExit("FATAL: BW_EFF unset -- the efficiency column is "
                         "meaningless without the Task 1 number")
    if CALLS != 256:
        raise SystemExit("FATAL: shape counts sum to %d, not the measured 256"
                         " calls/step" % CALLS)
    print("BW_eff = %.0f GB/s ; %d GEMV calls/step/rank" % (BW_EFF, CALLS),
          flush=True)
    dev = "cuda:0"
    rows = []
    for K, N, cnt in SHAPES:
        t = make(K, N, dev)
        eager = timed(lambda: call(t))
        call(t)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                call(t)
        graph = timed(lambda: g.replay(), iters=20) / 20
        # bytes this call must move: 4-bit weights + fp16 scales + packed zp
        cbytes = K * N / 2 * (1 + 1 / 32 + 1 / 128)
        floor_us = cbytes / (BW_EFF * 1e9) * 1e6
        rows.append(dict(K=K, N=N, count=cnt, eager_us=eager * 1e3,
                         graph_us=graph * 1e3, bytes=cbytes,
                         floor_us=floor_us, eff_pct=floor_us / (graph * 1e3) * 100,
                         gbps=cbytes / (graph * 1e3 * 1e-6) / 1e9))
        print("K=%5d N=%5d x%-3d eager=%7.1fus graph=%7.1fus floor=%6.1fus"
              " -> %3.0f%% of BW_eff (%4.0f GB/s)"
              % (K, N, cnt, eager * 1e3, graph * 1e3, floor_us,
                 rows[-1]["eff_pct"], rows[-1]["gbps"]), flush=True)
        del t, g
    json.dump({"bw_eff_gbps": BW_EFF, "calls_per_step": CALLS, "shapes": rows},
              open(f"{OUT}/p0_shapes_baseline.json", "w"), indent=1)
    lo = min(rows, key=lambda r: r["eff_pct"])
    hi = max(rows, key=lambda r: r["eff_pct"])
    print("worst %.0f%% (K=%d,N=%d)  best %.0f%% (K=%d,N=%d)"
          % (lo["eff_pct"], lo["K"], lo["N"], hi["eff_pct"], hi["K"], hi["N"]))
    # Weighted by occurrence count, not by shape -- this is the number the
    # predicted-e2e-gain formula consumes, and the cross-check against the
    # in-engine w4a16_gemv bucket from Task 3.
    step_us = sum(r["graph_us"] * r["count"] for r in rows)
    wmean = sum(r["eff_pct"] * r["count"] for r in rows) / CALLS
    floor_step = sum(r["floor_us"] * r["count"] for r in rows)
    print("count-weighted eff = %.0f%% of BW_eff  (arithmetic mean of shapes:"
          " %.0f%%)" % (wmean, sum(r["eff_pct"] for r in rows) / len(rows)))
    print("predicted in-engine GEMV = %.0f us/step/rank (floor if perfect:"
          " %.0f us)" % (step_us, floor_step))


if __name__ == "__main__":
    main()
