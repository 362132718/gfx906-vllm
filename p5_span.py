"""p5_span.py -- measure the TRUE decode step from a chrome trace.

  python3 p5_span.py <prof_dir> [moe_kernels_per_step]

Why: p5_moe.py / p5_prof.py computed the step as
      (T(request with N tokens) - T(request with 1 token on the SAME prompt)) / (N-1)
which is wrong when prefix caching is on: the second request's prefill is cached
and free, so a full cold prefill gets subtracted from it. The trace does not
care about any of that -- the MoE kernel runs exactly `k` times per decode step
(2 per layer x 40 layers = 80), so the inter-arrival time of every k-th MoE
kernel IS the wall step, measured on the GPU timeline.

Cross-check this against the device-busy total: busy/step must be < 100%, or the
step estimate is impossible.
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

prof = sys.argv[1] if len(sys.argv) > 1 else "/work/p5prof/prof"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 80          # MoE kernels per step
MOE_RE = re.compile(r"fused_moe_kernel|moe_kernel", re.I)
HOST_CATS = {"cpu_op", "cuda_runtime", "hip_runtime", "cuda_driver",
             "gpu_user_annotation", "user_annotation", "overhead", "trace",
             "python_function"}

files = sorted(glob.glob(os.path.join(prof, "**", "*.json*"), recursive=True))
if not files:
    print("no traces under %s" % prof)
    sys.exit(1)

for fn in files:
    try:
        raw = (gzip.open(fn, "rt", errors="ignore").read()
               if fn.endswith(".gz") else open(fn, errors="ignore").read())
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print("  !! %s: %s" % (os.path.basename(fn), e))
        continue
    evs = data.get("traceEvents", data if isinstance(data, list) else [])
    moe_ts, busy, nk = [], defaultdict(float), 0
    span_lo, span_hi = None, None
    for e in evs:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        ts = float(e.get("ts") or 0.0)
        dur = float(e.get("dur") or 0.0)
        if span_lo is None or ts < span_lo:
            span_lo = ts
        if span_hi is None or ts + dur > span_hi:
            span_hi = ts + dur
        if cat in HOST_CATS or not cat.startswith(("kernel", "gpu", "device")):
            continue
        busy[e.get("name") or "?"] += dur
        nk += 1
        if MOE_RE.search(e.get("name") or ""):
            moe_ts.append(ts)

    moe_ts.sort()
    # NOTE: a prefill forward also fires K MoE calls (2 per layer x L layers), so
    # moe_calls/K counts forwards, not decode steps, and the first forward is a
    # prefill whose duration is ~100x a decode step. Take the median of the
    # every-K-th inter-arrival so the single prefill gap cannot skew it.
    deltas = [(moe_ts[i + K] - moe_ts[i]) / 1000.0
              for i in range(0, len(moe_ts) - K)]
    deltas_sorted = sorted(deltas)
    n_forwards = len(moe_ts) / float(K)
    total_busy = sum(busy.values())

    print("=== %s ===" % os.path.basename(fn)[:60])
    print("  kernels=%d  moe_calls=%d  forwards=%.2f" % (nk, len(moe_ts), n_forwards))
    print("  trace span            : %.1f ms" % ((span_hi - span_lo) / 1000.0))
    if deltas_sorted:
        med = deltas_sorted[len(deltas_sorted) // 2]
        lo = deltas_sorted[max(0, len(deltas_sorted) // 20)]
        hi = deltas_sorted[min(len(deltas_sorted) - 1, len(deltas_sorted) * 19 // 20)]
        print("  TRUE decode step (median of every-%d-th MoE inter-arrival):" % K)
        print("      median %.2f ms  (p5 %.2f, p95 %.2f)  -> %.2f tok/s"
              % (med, lo, hi, 1000.0 / med))
        print("  device-busy total     : %.1f ms  = %.1f%% of the %.1f ms span"
              % (total_busy / 1000.0, total_busy / (span_hi - span_lo) * 100.0,
                 (span_hi - span_lo) / 1000.0))
    else:
        print("  not enough MoE calls to anchor the step")
    print("  top-3 by busy:")
    for n, us in sorted(busy.items(), key=lambda kv: -kv[1])[:3]:
        print("    %-46s %9.1f ms" % (re.sub(r"<.*?>", "", n)[:46], us / 1000.0))
