# p0_parse_trace.py -- aggregate vLLM's per-worker chrome traces into one
# kernel table (/work/p0/p0_kernels.json) that p0_buckets.py turns into the
# spec section 1(b) five-bucket table.
#
# Run INSIDE the dev container (or anywhere /work is mounted):
#   docker exec vllm-gfx906 python3 /work/p0/p0_parse_trace.py
#
# Why per-rank and why "union": each TP rank has its own trace file and its own
# card, and the two ranks are symmetric, so rank 0 alone is a valid picture of
# one card. Within a rank, kernels on different streams overlap, so summing
# durations would double-count; the merged interval union is the honest GPU-busy
# time. Both numbers are emitted (sum_us, union_us) because their difference is
# itself evidence about stream overlap.
import glob
import gzip
import json
import os
import re
from collections import Counter, defaultdict

PROF_DIR = os.environ.get("PROF_DIR", "/work/p0/prof")
OUT = os.environ.get("OUT", "/work/p0")
# GEMV calls per decode step per rank, from the eager shape census
# (p0_shapes.log). Used only to derive the number of steps inside the recorded
# window, and to cross-check that the captured graph launches what eager did.
GEMV_PER_STEP = int(os.environ.get("GEMV_PER_STEP", "256"))
GEMV_RE = re.compile(r"gemm_half_q_half_gptq", re.I)
# Host-side categories: never counted as device work.
HOST_CATS = {"cpu_op", "cuda_runtime", "hip_runtime", "cuda_driver",
             "gpu_user_annotation", "user_annotation", "overhead", "trace"}


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d) if isinstance(d, dict) else d


def union_us(intervals):
    """Merged length of [start, end) intervals -- true busy time under overlap."""
    iv = sorted(intervals)
    tot = 0.0
    cur_s = cur_e = None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def parse_rank(files):
    dev = defaultdict(lambda: [0.0, 0])       # kernel name -> [us, count]
    host = defaultdict(lambda: [0.0, 0])      # runtime call name -> [us, count]
    cats = Counter()
    intervals = []                            # device (start, end)
    gaps = []
    for path in files:
        for e in load(path):
            if e.get("ph") != "X" or "dur" not in e:
                cats[str(e.get("ph"))] += 1
                continue
            cat = str(e.get("cat", ""))
            cats[cat] += 1
            name = str(e.get("name", ""))
            ts, dur = float(e["ts"]), float(e["dur"])
            if cat in HOST_CATS:
                host[name][0] += dur
                host[name][1] += 1
                continue
            if not (cat == "kernel" or cat.startswith("gpu_")):
                cats["UNSEEN-CATEGORY:" + cat] += 1
                continue
            dev[name][0] += dur
            dev[name][1] += 1
            intervals.append((ts, ts + max(dur, 0.0)))
    intervals.sort()
    for (a_s, a_e), (b_s, _b_e) in zip(intervals, intervals[1:]):
        gaps.append(b_s - a_e)
    span = intervals[-1][1] - intervals[0][0] if intervals else 0.0
    return {
        "files": [os.path.basename(p) for p in files],
        "rows": sorted(([n, v[0], v[1]] for n, v in dev.items()),
                       key=lambda r: -r[1]),
        "host_rows": sorted(([n, v[0], v[1]] for n, v in host.items()),
                            key=lambda r: -r[1])[:20],
        "sum_us": sum(v[0] for v in dev.values()),
        "union_us": union_us(intervals),
        "span_us": span,
        "max_gap_us": max(gaps) if gaps else 0.0,
        "n_events": len(intervals),
        "cats": dict(cats),
    }


def main():
    allf = sorted(glob.glob(f"{PROF_DIR}/**/*.json*", recursive=True))
    if not allf:
        raise SystemExit("FATAL: no traces under %s -- did profiler_config "
                         "reach the workers, and did stop_profile() run?" % PROF_DIR)
    by_rank = defaultdict(list)
    for f in allf:
        m = re.search(r"rank(\d+)", os.path.basename(f))
        by_rank[m.group(1) if m else "?"].append(f)
    out = {"schema": "p0-trace-v1", "prof_dir": PROF_DIR,
           "ranks": {r: parse_rank(fs) for r, fs in sorted(by_rank.items())}}
    # Steps in the window, derived from the once-per-step GEMV census.
    prim = out["ranks"][next(iter(out["ranks"]))]
    gemv = [r for r in prim["rows"] if GEMV_RE.search(r[0])]
    gemv_n = sum(r[2] for r in gemv)
    out["gemv"] = {"names": [[r[0], r[1], r[2]] for r in gemv],
                   "launches": gemv_n,
                   "per_step_expected": GEMV_PER_STEP,
                   "steps": round(gemv_n / GEMV_PER_STEP, 2) if gemv_n else 0}
    json.dump(out, open(f"{OUT}/p0_kernels.json", "w"))
    for r, d in out["ranks"].items():
        print("rank %s: %s | %.1f ms busy(sum) / %.1f ms busy(union) over "
              "%.1f ms span | worst gap %.1f ms | %d device events"
              % (r, ",".join(d["files"]), d["sum_us"] / 1e3, d["union_us"] / 1e3,
                 d["span_us"] / 1e3, d["max_gap_us"] / 1e3, d["n_events"]))
        print("    cats: %s" % d["cats"])
    print("GEMV kernels: %d launches over %s steps (census expects %d/step)"
          % (gemv_n, out["gemv"]["steps"], GEMV_PER_STEP))
    print("TOP25 by device time (us per window, primary rank):")
    for n, us, c in prim["rows"][:25]:
        print("  %9.1f  %7d  %s" % (us, c, n[:100]))


if __name__ == "__main__":
    main()
