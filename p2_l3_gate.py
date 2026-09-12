# p2_l3_gate.py -- G-P4 L3 verdict: teacher-forced margin attribution + top-8.
#
# Run in the container after BOTH arms of p2_l3.sh:
#   docker exec -i vllm-gfx906 python3 /work/p2/p2_l3_gate.py
#
# The plan's literal criterion is "存在 M₀ 使 margin > M₀ 处零翻转". Read on its
# own that is vacuous: M₀ = (largest flip margin) always works. What makes it a
# gate is what the same engine costs itself. So compute two numbers:
#
#   M0_swap  = max min-margin over positions where the two ARMS disagree on top-1
#              on the FORCED DECODE route (p2_l3_run pass 2b). It has to be the
#              decode route: P2 only fires at M==1, so a cross-arm comparison of the
#              prefill rows would be taken at ~zero flips and PASS while measuring
#              nothing. Those prefill-vs-prefill flips are still reported, as a
#              control that should be empty.
#   M0_floor = max min-margin over positions where one arm's PREFILL route
#              disagrees with its own forced DECODE route (same weights, same
#              engine, same tokens, same prefix -- only the evaluation route
#              differs), measured on the OFF arm
#
# PASS requires M0_swap <= M0_floor, i.e. every cross-arm disagreement happens in
# the margin band where this engine already disagrees with itself. High-margin
# flips (above the engine's own floor) would be a real defect, and the gate stops
# for them. Flip counts and rates are reported, never gated: the spec's point is
# that a total flip rate is not a correctness criterion.
#
# A position the probe dropped (the arm's own greedy choice had already left the
# forced track) is not compared: it is a top-1 flip one position earlier, which the
# swap set covers. The two arms' drop sets need NOT match -- on the on arm the forced
# pick is P2's, so a disagreement between the decode routes shows up as a position one
# arm kept and the other dropped. The swap set is therefore the INTERSECTION, and the
# gate stops only if that intersection falls below 90% of either arm's rows (a silent
# shrink to nothing is the failure mode worth catching). The asymmetry itself is
# reported, per prompt, as evidence about the routes.
#
# min-margin is the WEAKER of the two margins at a position -- a flip only counts
# as "high margin" if both sides were confident.
#
# Second condition, with warn upgraded to FAIL (plan Task 4 Step 2.4): at every
# cross-arm flip the new arm's top-1 must sit inside the incumbent's top-8 and
# vice versa. Upstream treats that as a warning because their test skips when the
# GPU is absent; we have no such escape hatch.
#
# Third condition, added after the first dry run: dispatch proof. M0_swap <= M0_floor
# is satisfied by "nothing was swapped", so the margin gate on its own goes green in
# exactly the situation that should stop the project -- a whitelist that silently
# stopped matching. p2_l3_run counts real calls of _p2_w4a16_gemm_impl inside each TP
# worker and the gate compares ROUNDS, not zero-vs-nonzero: the load-time warm-up
# ignores the kill-switch, so every arm reports one round of 64 layers and only a live
# M==1 branch adds the size-1 graph's capture rounds. See dispatch_proof().
# [[gfx906-env-pitfalls]] #33 #34
# #30 is the same trap seen from the other side (a control arm that cannot fire).
import glob
import json
import os
import sys

OUT = os.environ.get("OUT", "/work/p2/l3")
TOPN = 8


def by_pos(rows):
    return {r["i"]: r for r in rows}


def head(room, k=TOPN):
    """(top1_tid, margin, [tids..k]) from a scored row's rank-ordered top list."""
    top = [t for t in room["top"] if t[1] is not None]
    top.sort(key=lambda t: t[2] if t[2] is not None else 1 << 30)
    if len(top) < 2:
        return None
    return top[0][0], top[0][1] - top[1][1], [t[0] for t in top[:k]]


def flip(a, b, ia, ib):
    """(pos, min_margin, a_top1, b_top1, a_topN, b_topN) if a and b differ."""
    ha, hb = head(a), head(b)
    if not ha or not hb:
        return None
    if ha[0] == hb[0]:
        return None
    return (ia, min(ha[1], hb[1]), ha[0], hb[0], ha[2], hb[2])


def load(arm):
    p = os.path.join(OUT, "l3_%s.json" % arm)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def dispatch_proof(arm):
    """What p2_l3_run's per-worker counters say about THIS arm's routing.

    The margin gate below can only mean something if the on arm really ran P2 on the
    route being compared. Counters come from inside the workers (the hook sits on the
    op impl's plain function), are flushed from the first call because vLLM hard-kills
    TP workers, and are keyed by pid because both ranks report dev0.

    Magnitude caveat, learned the hard way on the first smoke: the hook is only
    reachable from the impl's `M == 1` branch and a decode step under
    FULL_DECODE_ONLY is a graph replay that runs no Python, so a healthy on arm
    reports 3 rounds x 64 layers = 192 per rank (load warm-up, then the lazy size-1
    graph's own warm-up + capture), NOT 64 per step. Treat it as "the M==1 branch is
    inside the graph this arm replays"; the per-step number is device-side and belongs
    to G-P3. One round of 64 is present in BOTH arms, because the load-time warm-up
    ignores the kill-switch on purpose (it exists to pull the code object out of
    capture), so no arm's count is ever legitimately zero.
    """
    out = {"files": 0, "total": 0, "pids_with_hits": 0, "by_family": {},
           "per_pid": {}}
    for path in sorted(glob.glob(os.path.join(OUT, "p2hits_%s_*.json" % arm))):
        try:
            d = json.load(open(path))
        except Exception as e:                 # a truncated flush is still evidence
            out["per_pid"][os.path.basename(path)] = "UNREADABLE %r" % (e,)
            continue
        n = sum(d["hits"].values())
        out["files"] += 1
        out["total"] += n
        if n:
            out["pids_with_hits"] += 1
        out["per_pid"][str(d["pid"])] = n
        for k, v in d["hits"].items():
            out["by_family"][k] = out["by_family"].get(k, 0) + v
    return out


def main():
    off, on = load("off"), load("on")
    if not off or not on:
        print("FAIL missing l3_off.json / l3_on.json in %s" % OUT, flush=True)
        return 1
    assert off["arm"] == "off" and on["arm"] == "on"
    print("L3: arms off/on, %d prompt(s), ntok=%s" % (len(off["rows"]), off["ntok"]),
          flush=True)
    swap, floor, ctrl, free_flips = [], [], [], []
    n_pos = n_free_pos = 0
    topn_bad, free_div, dropped, free_skipped = [], [], [], []
    asym = []

    for pi, (ro, rn) in enumerate(zip(off["rows"], on["rows"])):
        pl = ro["prompt_len"]
        assert ro["comp_scored"] == rn["comp_scored"], \
            "prompt %d: the two arms scored different sequences" % pi
        assert ro["prompt_len"] == rn["prompt_len"], "prompt %d: prompt lens" % pi
        pf_o, pf_n = by_pos(ro["prefill"]), by_pos(rn["prefill"])
        df_o, df_n = by_pos(ro["decode_forced"]), by_pos(rn["decode_forced"])
        dropped.append((pi, len(ro.get("decode_forced_dropped", [])),
                        len(rn.get("decode_forced_dropped", []))))
        shared = set(df_o) & set(df_n)
        asym.append((pi, len(df_o), len(df_n),
                     sorted(set(df_o) - set(df_n)), sorted(set(df_n) - set(df_o))))
        # Equal drop sets are NOT a validity precondition. A drop means this arm's
        # prefill-routed greedy pick at t-1 disagreed with the forced sequence, and on
        # the on arm the forced pick is P2's -- so any position where the two decode
        # routes disagree can drop in one arm only. Demanding set(df_o) == set(df_n)
        # therefore demands that the swap changed nothing, the mirror image of the
        # hollow gate this whole file exists to prevent. Off's own atomicAdd jitter
        # (71-72% of positions move between two identical incumbent runs) puts a floor
        # under the asymmetry too.
        # What must hold is that the comparison did not quietly shrink: the shared set
        # is the only thing both arms scored on the identical prefix seq[:t-1], so a
        # near-empty intersection means there is no gate left.
        if not shared or len(shared) < 0.9 * min(len(df_o), len(df_n)):
            print("FAIL p%d: only %d of off %d / on %d forced positions are shared"
                  " (<90%%) -- the swap set would be too thin to gate"
                  % (pi, len(shared), len(df_o), len(df_n)), flush=True)
            return 1
        # ---- the gate: cross-arm, decode route (where P2 lives), same prefix ----
        for i in sorted(shared):
            n_pos += 1
            f = flip(df_o[i], df_n[i], i, i)
            if f:
                swap.append((pi, f))
                # f = (pos, min_margin, off_top1, on_top1, off_topN, on_topN)
                a_in, b_in = f[3] in f[4], f[2] in f[5]
                if not (a_in and b_in):
                    topn_bad.append((pi, f[0], f[2], f[3], a_in, b_in))
        # ---- control: cross-arm prefill rows. P2 never runs at M>1, so this
        #      should be empty; if it is not, something besides the kernel moved.
        for i in sorted(set(pf_o) & set(pf_n)):
            f = flip(pf_o[i], pf_n[i], i, i)
            if f:
                ctrl.append((pi, f))
        # ---- floor: off arm, prefill route vs its own forced decode route ----
        for i in sorted(set(pf_o) & set(df_o)):
            f = flip(pf_o[i], df_o[i], i, i)
            if f:
                floor.append((pi, f))
        # ---- free run (plan Step 2.4): top-N(8) inclusion, warn -> FAIL ----
        # Row j is conditioned on prompt + comp_own[0..j-1], so the arms are only
        # comparable while their own completions still agree. Past j0 (the first
        # differing token) a "flip" would just be two different prefixes. j0 itself
        # IS comparable and is the interesting one: it is the divergence point.
        a, b = ro["comp_own"], rn["comp_own"]
        j0 = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
                  min(len(a), len(b)))
        dc_o = by_pos([{**r, "i": r["i"] + pl} for r in ro["decode"]])
        dc_n = by_pos([{**r, "i": r["i"] + pl} for r in rn["decode"]])
        for j in range(0, j0 + 1):
            i = j + pl
            if i not in dc_o or i not in dc_n:
                continue
            n_free_pos += 1
            f = flip(dc_o[i], dc_n[i], i, i)
            if f:
                free_flips.append((pi, f))
                if f[3] not in f[4] or f[2] not in f[5]:
                    topn_bad.append((pi, f[0], f[2], f[3],
                                     f[3] in f[4], f[2] in f[5]))
        free_skipped.append((pi, max(0, min(len(a), len(b)) - (j0 + 1))))
        if a != b:
            free_div.append((pi, j0, len(a), len(b)))
    if not n_pos:
        print("FAIL: zero probed decode positions -- nothing was gated", flush=True)
        return 1

    dp_off, dp_on = dispatch_proof("off"), dispatch_proof("on")
    # NOT "off == 0". The load-time warm-up (wna16_exllama.py:395) calls the P2 impl
    # directly and never reads VLLM_WNA16_P2_GEMV, so a completely dead route still
    # reports one full round: 64 P2-routed layers per rank. The first run of this
    # gate measured off=60/rank on a healthy pair, and "off == 0" would have failed it.
    # What discriminates is the ROUND COUNT. vLLM captures the size-1 graph lazily, at
    # the first decode step of size 1, and capture costs a warm-up run plus the capture
    # itself -- two more rounds through the M==1 branch. So off = 1 x 64 (observed 60:
    # the counter flushes every 10th call, so a file is a floor on its round total)
    # versus on = 3 x 64 = 192 (observed 190), both at the same 48:16 family split.
    # Requiring >= 1 whole extra round kills both hollow cases: a route that never
    # fires gives on == off, and a leaked kill-switch that makes BOTH arms live also
    # gives on == off, so neither can pass.
    ok_dispatch = (dp_off["total"] > 0 and dp_off["pids_with_hits"] >= 2
                   and dp_on["pids_with_hits"] >= 2
                   and dp_on["total"] >= 2 * dp_off["total"]
                   and len(dp_on["by_family"]) >= 1)
    print("\ndispatch proof (M==1 branch rounds; off = warm-up only, on = warm-up +"
          " capture warm-up + capture): off total=%d over %d pid(s), on total=%d over"
          " %d pid(s) (want on >= 2*off on >=2 pids each), families=%s"
          % (dp_off["total"], dp_off["pids_with_hits"], dp_on["total"],
             dp_on["pids_with_hits"], dict(sorted(dp_on["by_family"].items()))),
          flush=True)
    print("   per-pid off=%s on=%s" % (dp_off["per_pid"], dp_on["per_pid"]),
          flush=True)
    if not dp_on["files"]:
        print("   NOTE: no on-arm counter files at all -- the hook never installed"
              " (see '[p2count]' lines in p2_l3_on.log), so this run has NO routing"
              " evidence", flush=True)
    print("   [dispatch] -> %s" % ("PASS" if ok_dispatch else "FAIL"), flush=True)

    m0_swap = max((f[1] for _, f in swap), default=0.0)
    m0_floor = max((f[1] for _, f in floor), default=0.0)
    print("\npositions compared (forced decode route, both arms): %d" % n_pos,
          flush=True)
    print("cross-arm flips      : %d (%.2f%%)  worst min-margin = %.4f nats"
          % (len(swap), 100.0 * len(swap) / max(n_pos, 1), m0_swap), flush=True)
    print("noise floor flips    : %d  (off: prefill route vs its own decode route)"
          " worst min-margin = %.4f nats"
          % (len(floor), m0_floor), flush=True)
    print("control (cross-arm prefill) flips: %d%s"
          % (len(ctrl), "" if ctrl else "  <- as expected: P2 never runs at M>1"),
          flush=True)
    print("free-run cross-arm flips: %d over %d shared positions (%d skipped as"
          " post-divergence, per prompt %s)"
          % (len(free_flips), n_free_pos, sum(c for _, c in free_skipped),
             free_skipped), flush=True)
    print("probe dropped positions (arm left the forced track): %s" % dropped,
          flush=True)
    print("forced-track asymmetry off/on rows, positions kept by one arm only:"
          " %s" % [(pi, o, n, "off_only=%s" % a, "on_only=%s" % b)
                   for pi, o, n, a, b in asym], flush=True)
    for pi, f in sorted(swap, key=lambda t: -t[1][1])[:12]:
        print("   flip p%d pos%-5d min_margin=%.4f  off=%d on=%d"
              % (pi, f[0], f[1], f[2], f[3]), flush=True)
    for pi, f in sorted(floor, key=lambda t: -t[1][1])[:6]:
        print("   floor p%d pos%-5d min_margin=%.4f  prefill=%d decode=%d"
              % (pi, f[0], f[1], f[2], f[3]), flush=True)
    print("top-%d inclusion violations: %d" % (TOPN, len(topn_bad)), flush=True)
    for t in topn_bad[:12]:
        print("   p%d pos%-5d off_top1=%d on_top1=%d  on_in_off_top8=%s"
              " off_in_on_top8=%s" % t, flush=True)
    print("free-run divergence off vs on: %d prompt(s) %s"
          % (len(free_div), free_div or "(all identical)"), flush=True)

    ok_swap = m0_swap <= m0_floor + 1e-12
    # A missing floor is not a free pass: without it there is no band to compare
    # the flips against, so the gate would be asserting "0 <= 0" on an instrument
    # that produced half its numbers. Only an empty swap set may pass an empty floor.
    ok_floor = bool(floor) or not swap
    ok_topn = not topn_bad
    print("\n[cross-margin] M0_swap %.4f <= M0_floor %.4f -> %s"
          % (m0_swap, m0_floor, "PASS" if (ok_swap and ok_floor) else "FAIL"),
          flush=True)
    print("[floor established] %s" % ("yes" if ok_floor else
          "NO -- flips cannot be attributed, and prefill-vs-decode on one engine "
          "always flips a few percent, so an empty floor here means the instrument "
          "is broken"), flush=True)
    print("[top-%d inclusion] %d violations -> %s"
          % (TOPN, len(topn_bad), "PASS" if ok_topn else "FAIL"), flush=True)
    print("[dispatch proof] off=%d on=%d -> %s"
          % (dp_off["total"], dp_on["total"],
             "PASS" if ok_dispatch else "FAIL"), flush=True)
    passed = bool(ok_swap and ok_floor and ok_topn and ok_dispatch)
    print("G-P4 L3 %s" % ("PASS" if passed else "FAIL"), flush=True)
    json.dump({"route": "forced decode (M=1)", "n_pos": n_pos,
               "m0_swap": m0_swap, "m0_floor": m0_floor,
               "dispatch_off": dp_off, "dispatch_on": dp_on,
               "swap_flips": [[pi, f[0], f[1], f[2], f[3]] for pi, f in swap],
               "n_floor_flips": len(floor),
               "n_control_prefill_flips": len(ctrl),
               "n_free_run_flips": len(free_flips),
               "n_free_run_compared": n_free_pos,
               "free_run_skipped": [list(t) for t in free_skipped],
               "dropped_positions": [list(t) for t in dropped],
               "forced_track_asymmetry": [
                   [pi, o, n, a, b] for pi, o, n, a, b in asym],
               "floor_worst": [[pi, f[0], f[1]] for pi, f in
                               sorted(floor, key=lambda t: -t[1][1])[:20]],
               "topn_violations": [list(t) for t in topn_bad],
               "free_run_divergence": free_div,
               "pass": passed},
              open(os.path.join(OUT, "p2_l3_verdict.json"), "w"), indent=1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
