# p2_ab_gate.py -- judge G-P3 from what the two arms left on disk.
#
# Run in the container after p2_ab.sh:  python3 /work/p2/p2_ab_gate.py
#
# Reads, per arm: p0_wall.json (perf), p0_kernels.json (per-rank device rows),
# p0_window.json (trace window), p0_step_out.json (greedy ids),
# p0_step_shapes.csv.<pid> (what apply_weights actually saw).
#
# Why the launch count comes from the trace and not from _P2_HITS: the probe is
# a per-process list with no export path, so nothing in the arm's output would
# carry it (pitfall #26). Its append does now run on every dispatch -- Task 3
# moved the M test into the custom op impl, which dynamo does not trace -- but
# reading it would still take a hook into the workers. Device events are already
# per rank and already on disk, and they cross-check the incumbent count too.
import csv
import collections
import glob
import json
import os
import sys

BASE = os.environ.get("BASE", "/work/p2")
ANCHOR_OFF_MS = 33.008
ANCHOR_BAND_MS = 0.3
GATE_ON_MS = 31.9 + 0.3
GATE_TPS = 30.6
SUSPICIOUSLY_FAST_MS = 30.0        # below this something is not running at all
EXPECT_PER_STEP = 64               # 48 x (5120,8192) + 16 x (5120,7168)
# Incumbent fused GEMV rate per step per rank, from the P0.5 traces
# (16384 launches over the 64-step window). The on arm must lose exactly the
# EXPECT_PER_STEP calls that P2 takes over.
INC_PER_STEP = 256
rows_out = []


def chk(name, ok, detail):
    rows_out.append((bool(ok), name, detail))
    print("%-6s %s | %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def load(arm, fname):
    path = os.path.join(BASE, "ab_%s" % arm, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def kernel_counts(arm, pat):
    """-> {rank: [launches, us]} for device kernels whose name contains pat."""
    k = load(arm, "p0_kernels.json")
    if k is None:
        return None
    out = {}
    for rank, d in k["ranks"].items():
        n = sum(r[2] for r in d["rows"] if pat in r[0])
        us = sum(r[1] for r in d["rows"] if pat in r[0])
        out[rank] = [n, us]
    return out


def inc_expect(arm, steps, expect):
    """Incumbent fused-GEMV rate per rank must match the hand-off arithmetic."""
    got = kernel_counts(arm, "gemm_half_q_half_gptq")
    if got is None:
        chk("incumbent rate (%s arm)" % arm, False, "no p0_kernels.json")
        return
    per_rank = {r: round(v[0] / float(steps), 1) for r, v in got.items()}
    chk("incumbent rate ~%d/step (%s arm)" % (expect, arm),
        all(abs(c - expect) <= 0.1 * expect for c in per_rank.values()),
        "per-step %s" % per_rank)


def main():
    off, on = load("off", "p0_wall.json"), load("on", "p0_wall.json")
    if not off or not on:
        chk("G-P3 data present", False, "missing p0_wall.json for one arm")
        return finish()
    chk("G-P3 data present", True,
        "off %.3f ms / on %.3f ms" % (off["per_step_ms"], on["per_step_ms"]))

    chk("anchor reproduces (arm off)",
        abs(off["per_step_ms"] - ANCHOR_OFF_MS) <= ANCHOR_BAND_MS,
        "%.3f ms vs %.3f +/- %.1f (drift => conditions changed, A/B void)"
        % (off["per_step_ms"], ANCHOR_OFF_MS, ANCHOR_BAND_MS))
    tps = 1000.0 / on["per_step_ms"]
    chk("on arm step <= %.1f ms" % GATE_ON_MS, on["per_step_ms"] <= GATE_ON_MS,
        "%.3f ms, delta %.3f ms, %.2f tok/s"
        % (on["per_step_ms"], off["per_step_ms"] - on["per_step_ms"], tps))
    chk("on arm tok/s >= %.1f" % GATE_TPS, tps >= GATE_TPS, "%.2f tok/s" % tps)
    chk("on arm not suspiciously fast",
        on["per_step_ms"] >= SUSPICIOUSLY_FAST_MS,
        "%.3f ms (floor %.1f: a step this cheap means work went missing)"
        % (on["per_step_ms"], SUSPICIOUSLY_FAST_MS))

    win = load("on", "p0_window.json") or {}
    steps = win.get("active_iterations") or 64
    for arm, want in (("off", 0), ("on", EXPECT_PER_STEP * steps)):
        got = kernel_counts(arm, "p2_gemv")
        if got is None:
            chk("dispatch seen in trace (%s)" % arm, False, "no p0_kernels.json")
            continue
        tot = sum(v[0] for v in got.values())
        per_rank = {r: round(v[0] / float(steps), 1) for r, v in got.items()}
        if want == 0:
            chk("env lever reaches workers (arm off)", tot == 0,
                "p2_gemv launches=%d (must be 0, else off arm ran P2 too)" % tot)
            inc_expect(arm, steps, INC_PER_STEP)
        else:
            ok = all(abs(c - EXPECT_PER_STEP) <= 0.1 * EXPECT_PER_STEP
                     for c in per_rank.values())
            chk("per-step P2 dispatch ~%d on both ranks" % EXPECT_PER_STEP, ok,
                "total=%d over %s steps/rank, per-step %s"
                % (tot, steps, per_rank))
            inc_expect(arm, steps, INC_PER_STEP - EXPECT_PER_STEP)

    for arm in ("off", "on"):
        files = sorted(glob.glob(os.path.join(
            BASE, "ab_%s" % arm, "p0_step_shapes.csv.*")))
        chk("shape census: both ranks dumped (%s)" % arm, len(files) >= 2,
            "%d file(s)" % len(files))
        for path in files:
            mult, ms = collections.Counter(), set()
            with open(path) as f:
                for r in csv.DictReader(f):
                    m, k, n = int(r["m"]), int(r["k"]), int(r["n"])
                    ms.add(m)
                    mult[(k, n)] += 1
            if not ms:
                continue          # APIServer/EngineCore import the module but
                # never call apply_weights, so their dump is legitimately empty
            # The census can only ever show the prefill M: apply_weights is
            # inside the dynamo-traced region, so graph-replayed M=1 steps never
            # run the append (pitfall #7). M=1 dispatch is proven by the device
            # trace counts above, not here -- what THIS row proves is the 48+16
            # multiplicity, i.e. that the whitelist covers exactly those modules.
            want = {(5120, 8192): 48, (5120, 7168): 16}
            got = {fam: mult.get(fam, 0) for fam in want}
            chk("  %s multiplicities 48+16" % os.path.basename(path).split(".")[-1],
                got == want, "got %s at M in %s (M=1 by trace, not here)"
                % (got, sorted(ms)))

    a = load("off", "p0_step_out.json")
    b = load("on", "p0_step_out.json")
    if a and b:
        ta, tb = a["token_ids"], b["token_ids"]
        i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        n_diff = sum(1 for x, y in zip(ta, tb) if x != y)
        print("INFO   greedy: len %d/%d, first diff at token %s, %d differ%s"
              % (len(ta), len(tb), i, n_diff,
                 "" if i is None else
                 " -- attribution by teacher-forced margin is Task 4 (L3)"),
              flush=True)
    return finish()


def finish():
    bad = [n for ok, n, _ in rows_out if not ok]
    print("G-P3 %s (%d/%d checks)"
          % ("FAIL: " + "; ".join(bad) if bad else "PASS",
             len(rows_out) - len(bad), len(rows_out)), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
