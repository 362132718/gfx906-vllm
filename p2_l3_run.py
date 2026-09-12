# p2_l3_run.py -- G-P4 L3 data: teacher-forced margins on the DELIVERY engine.
#
# Driven per arm by p2_l3.sh (arm off first, then arm on). Run detached:
#   docker exec -d vllm-gfx906 bash -lc "cd /work/p2 && ARM=off \
#     VLLM_WNA16_P2_GEMV=0 VLLM_CACHE_ROOT=/tmp/p2cache_l3_off \
#     OUT=/work/p2/l3 python3 p2_l3_run.py > /work/p2/p2_l3_off.log 2>&1"
#
# Why teacher-forcing and not "the tokens must match" (spec L3, and the reason
# the whole gate is phrased this way): two correct-but-different fp16 add orders
# diverge on the SAME model within the first few dozen tokens, and a free run
# then compares different sequences, which says nothing. So:
#
#   pass 1  free-run greedy 256 with logprobs=8   -> the sequence AND the
#           decode-route top-k at every position.
#   pass 2a score prompt+completion with prompt_logprobs=32, max_tokens=1
#           -> the PREFILL-route top-32 at every position of an identical token
#           sequence (one shot, cheap).
#   pass 2b one request per scored position: prompt = seq[:t-1], max_tokens=2,
#           logprobs=32, take the SECOND token's distribution
#           -> the DECODE-route (M=1) top-32 at position t on that same prefix.
#
# 2b is not a refinement, it is what makes the gate non-vacuous: the P2 swap only
# exists on the M==1 route, so a cross-arm comparison of 2a rows (M>1, incumbent on
# both arms) would come out at ~zero flips and PASS while measuring nothing. 2a is
# kept because it is the other half of the noise floor (prefill vs decode inside one
# arm) and as an arm-independent control.
#
# The scored sequence is ALWAYS arm off's completion, written to
# {OUT}/l3_seqs.json by the off run and read back by the on run. If the two
# arms' free runs differ, that difference is a result (reported as free_run_diff)
# -- it must not be allowed to change what each arm is asked to score.
#
# Comparing pass 1 with pass 2 inside ONE arm is the noise floor the gate is
# measured against: same weights, same engine, same tokens, only prefill-vs-
# decode differs. vLLM flips a few percent of positions that way on its own.
#
# max_logprobs=32 is needed because the engine default is 20 (EngineArgs
# .max_logprobs); the criteria only need top-8 plus a margin, but asking for 32
# and getting an error is louder than silently being capped at 20.
import json
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")     # pitfall #3/#28
ARM = os.environ.get("ARM", "on")
OUT = os.environ.get("OUT", "/work/p2/l3")
NTOK = int(os.environ.get("NTOK", "256"))
# Smoke knob: the per-position decode probe costs one engine round-trip per
# position (4 x NTOK of them). PROBE_N bounds it to the first N positions so the
# plumbing can be proved in minutes before committing to the full run.
PROBE_N = int(os.environ.get("PROBE_N", "0")) or NTOK
MODEL = os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
SEQS = os.path.join(OUT, "l3_seqs.json")
# Deliberately mixed: a long explanatory prompt, a code prompt, an enumeration
# prompt (many near-ties), a very short one (all-decode, high determinism).
PROMPTS = [
    "Explain in detail why the sky is blue and what determines its shade at "
    "dawn. ",
    "Write a Python function that parses an ISO-8601 timestamp without using "
    "any third-party library, and explain each step. ",
    "List the twelve months of the year, each with its number of days and one "
    "agricultural task. ",
    "The capital of France is",
]


def top_list(d):
    """{token_id: Logprob} -> [[tid, logprob, rank], ...] sorted by rank.

    The gate reads element 0 as "top-1", so ordering is load-bearing: rank is
    normally present, but if it were ever None for every entry a rank-only sort
    would leave the dict order (= token id order) and the gate would score noise.
    Tie-break on descending logprob so that cannot happen silently.
    """
    if not d:
        return []

    def key(kv):
        lp = kv[1]
        return (lp.rank if lp.rank is not None else 1 << 30,
                -(lp.logprob if lp.logprob is not None else float("-inf")))

    out = []
    for tid, lp in sorted(d.items(), key=key):
        out.append([int(tid), None if lp.logprob is None else round(
            float(lp.logprob), 6), lp.rank])
    return out


def summarise(out, kind):
    """One entry per scored position: top-k list + who the actual token was."""
    rows = []
    if kind == "prefill":
        pls = out.prompt_logprobs
        seq = out.prompt_token_ids
        for i, d in enumerate(pls):
            if not d:                       # position 0 has no logprobs
                continue
            rows.append({"i": i, "tok": int(seq[i]), "top": top_list(d)})
    else:
        co = out.outputs[0]
        for i, d in enumerate(co.logprobs):
            rows.append({"i": i, "tok": int(co.token_ids[i]), "top": top_list(d)})
    return rows


def forced_decode_probe(llm, seq, prompt_len, ntok, topk):
    """Decode-route top-k at every scored position, on a FORCED prefix.

    pass 2a (prompt_logprobs) scores the whole shared sequence, but at M>1 both
    arms run the incumbent, so it can never show the swap. The swap only exists on
    the M==1 route, so each position needs its own decode step:

      prompt = seq[:t-1], max_tokens=2  ->  g0 comes from the prefill's last
      position, and the distribution that picks g1 (logprobs[1]) is produced by a
      real M=1 decode step conditioned on seq[:t].  That row is the decode-route
      twin of pass 2a's position t.

    Serial submission is not a perf choice: `M` is the flattened token count of the
    whole step, so with several sequences in flight a "decode step" is M=k and the
    P2 branch does not fire on the on arm either -- both arms would run the
    incumbent and the gate would measure nothing while looking green.

    If g0 != seq[t-1] the engine's own greedy choice left the forced track, so row
    t is conditioned on a different context than the other arm's; it is dropped and
    listed. That is not lost evidence: g0 != seq[t-1] IS a top-1 flip at t-1 on the
    decode route, which is already compared at t-1.
    """
    from vllm import SamplingParams       # local: must not precede the env backstop
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=topk,
                        ignore_eos=True)
    rows, dropped = [], []
    t0 = time.time()
    for n, t in enumerate(range(prompt_len, prompt_len + ntok)):
        out = llm.generate([{"prompt_token_ids": seq[:t - 1]}], sp,
                           use_tqdm=False)[0]
        co = out.outputs[0]
        if int(co.token_ids[0]) != int(seq[t - 1]):
            dropped.append({"i": t, "want_ctx_tok": int(seq[t - 1]),
                            "got": int(co.token_ids[0])})
            continue
        rows.append({"i": t, "tok": int(seq[t]), "top": top_list(co.logprobs[1])})
        if (n + 1) % 50 == 0:
            print("[%s] probe %d/%d (%.0f s, %.0f ms/pos)"
                  % (ARM, n + 1, ntok, time.time() - t0,
                     1e3 * (time.time() - t0) / (n + 1)), flush=True)
    print("[%s] forced-decode done: %d rows, %d dropped, %.0f s"
          % (ARM, len(rows), len(dropped), time.time() - t0), flush=True)
    return rows, dropped


# ---- non-vacuity proof: count real P2 dispatches, per worker, per arm ----------
# Without this the L3 gate is structurally unable to fail. If the whitelist stopped
# matching (a partition change, a .so that failed to load, a renamed env), BOTH arms
# would run the incumbent, the cross-arm swap set would be empty, M0_swap would be 0,
# and the gate would print PASS while measuring nothing -- the exact shape of the
# tautological control arm recorded in [[gfx906-env-pitfalls]] #30. The counters make
# that state impossible to reach quietly: the on arm must show dispatches, the off
# arm must show none (with the kill-switch env `_p2_route` stays False at load time,
# so the op is never called at all -- not even by the load-time warm-up).
#
# What the number MEANS: this function sits behind the op impl's `M == 1` test, and a
# decode step under FULL_DECODE_ONLY is a graph replay that runs no Python. So a
# whole on-arm run yields ~64 layers x (capture + warm-up) = ~128 calls per rank, not
# 64/step. That is still exactly the proof wanted -- "the M==1 branch was compiled
# INTO the size-1 graph this arm replays" -- but it is a routing proof, not a
# per-step count; the per-step authority stays the device trace (G-P3, 64/rank/step).
#
# Hook point is the PLAIN FUNCTION _p2_w4a16_gemm_impl, called by the registered op
# impl by global name at run time, so the patch is visible under torch.compile and
# graph capture (proven in p2_l2_dump.py). It must be installed at MODULE level: the
# spawn workers re-run this file as __mp_main__, so anything patched inside main()
# is absent exactly where P2 executes ([[gfx906-env-pitfalls]] #29).
def _install_p2_counter(cleanup):
    import glob as _glob
    import torch
    from vllm.model_executor.kernels.linear.mixed_precision import (
        wna16_exllama as W)

    # This runs at module level, i.e. before main()'s makedirs. Without it the very
    # first flush raises FileNotFoundError, the outer try swallows it, and the run
    # loses its dispatch proof while still printing a green-looking verdict.
    os.makedirs(OUT, exist_ok=True)
    # Parent only: every TP worker re-runs this installer at import, and an
    # unconditional sweep would have worker 2 unlink worker 1's live file.
    if cleanup:
        for stale in _glob.glob("%s/p2hits_%s_*.json" % (OUT, ARM)):
            os.remove(stale)                   # never read a previous run's floor
    orig = W._p2_w4a16_gemm_impl
    pid = os.getpid()
    path = "%s/p2hits_%s_%d.json" % (OUT, ARM, pid)
    hits = {}
    # vLLM hard-kills its TP workers, so atexit never runs.
    box = [0]

    def flush():
        with open(path, "w") as f:
            json.dump({"pid": pid, "arm": ARM, "hits":
                       {"%dx%d" % k: v for k, v in hits.items()}}, f)

    def hook(x, w_q, w_s, w_zp, part, out):
        rc = orig(x, w_q, w_s, w_zp, part, out)          # never alter behaviour
        try:
            # No tensor reads here: `x.abs().amax()` would sync per call and the
            # whole point is to observe the engine while it runs normally.
            k = (int(x.shape[1]), int(out.shape[1]))
            hits[k] = hits.get(k, 0) + 1
            box[0] += 1
            # Flush at once, then every 10. A 200-call cadence wrote NOTHING on a
            # working on arm: this function is only reached from the op impl's
            # `M == 1` branch, and under FULL_DECODE_ONLY a decode step is a graph
            # replay that runs no Python at all. The calls in a whole run are 3
            # rounds x 64 P2-routed layers = 192 per rank -- the load-time warm-up,
            # then the lazy size-1 graph's warm-up run and its capture -- which sits
            # under any 200 threshold, and a step of 50 would leave the tally up to
            # 49 short of the truth it is being reported as.
            #
            # Round 1 of those 3 is the warm-up at wna16_exllama.py:395, which does
            # NOT read VLLM_WNA16_P2_GEMV (it exists to load the code object outside
            # capture). So the OFF arm reports 64 too, and a gate condition of
            # "off == 0" is unsatisfiable. What separates the arms is the round
            # count: 1 x 64 dead, 3 x 64 live.
            #
            # So read this number for what it is: "the M==1 branch executed inside
            # this compiled engine's workers, on these families" -- routing proof,
            # not a per-step count. Per-step remains device-side (G-P3's trace:
            # 64 launches/step/rank), because a replay is invisible to Python.
            if box[0] == 1 or box[0] % 10 == 0:
                flush()
        except Exception as e:                            # a probe must not break a run
            print("[p2count-FAIL] %r" % (e,), flush=True)
        return rc

    W._p2_w4a16_gemm_impl = hook
    flush()
    print("[p2count] installed pid=%d arm=%s" % (pid, ARM), flush=True)


try:
    _install_p2_counter(__name__ == "__main__")
except Exception as _e:
    # Loud, not silent: an uninstalled counter means the gate cannot prove it had
    # teeth, and that must be visible in the same log as the verdict.
    print("[p2count] NOT INSTALLED %r -- the L3 gate will have no dispatch proof"
          % (_e,), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("env:", {k: os.environ.get(k) for k in (
        "VLLM_WNA16_EXLLAMA", "VLLM_WNA16_P2_GEMV", "VLLM_CACHE_ROOT")},
        flush=True)
    print("arm=%s ntok=%d prompts=%d" % (ARM, NTOK, len(PROMPTS)), flush=True)
    from vllm import LLM, SamplingParams
    from vllm.tokenizers import get_tokenizer

    tok = get_tokenizer(MODEL)
    ids0 = [list(map(int, tok.encode(p, add_special_tokens=False)))
            for p in PROMPTS]
    print("prompt token counts: %s" % [len(i) for i in ids0], flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
        attention_backend="TRITON_ATTN", disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN", max_logprobs=32, seed=0,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes":
                                [1, 2, 4, 8, 16, 24, 32, 48]},
    )
    print("engine up in %.0f s (compiled)" % (time.time() - t0), flush=True)

    # ---- pass 1: free-running greedy, with the decode-route top-8 ----
    # One request at a time, and that is not a perf choice: `M` is the flattened
    # token count of the whole step, so four concurrent sequences make every decode
    # step M=4 -- the on arm would never take the M==1 branch (P2 off during the run
    # meant to show it), and off's sequence would come from a different GEMM shape
    # than the serial pass 2b probes, which would show up as dropped positions.
    sp = SamplingParams(temperature=0.0, max_tokens=NTOK, ignore_eos=True,
                        logprobs=8)
    runs = []
    for pi, i in enumerate(ids0):
        out = llm.generate([{"prompt_token_ids": i}], sp, use_tqdm=False)[0]
        n = len(out.outputs[0].token_ids)
        assert n == NTOK, "prompt %d stopped early: %d != %d" % (pi, n, NTOK)
        runs.append({"prompt_ids": i, "comp": list(out.outputs[0].token_ids),
                     "decode": summarise(out, "decode")})
        print("[%s] free run %d/%d done (%.0f s)"
              % (ARM, pi + 1, len(ids0), time.time() - t0), flush=True)
    if ARM == "off":
        json.dump([{"prompt_ids": r["prompt_ids"], "comp": r["comp"]}
                   for r in runs], open(SEQS, "w"))
        print("wrote %s (the sequence both arms score)" % SEQS, flush=True)
    else:
        if not os.path.exists(SEQS):
            raise SystemExit("FATAL: %s missing -- the on arm must score the off"
                             " arm's own completions or the positions are not"
                             " comparable. Run arm=off first." % SEQS)
        ref = json.load(open(SEQS))
        for pi2, (r, e) in enumerate(zip(runs, ref)):
            same = r["comp"] == e["comp"]
            r["free_run_matches_off"] = same
            if not same:
                # Length first: "different" with no differing element in the common
                # prefix means the two runs used different NTOK, and the bare next()
                # here raised StopIteration and took the whole on arm down on a smoke
                # whose NTOK was smaller than the off run's.
                assert len(r["comp"]) == len(e["comp"]), \
                    ("prompt %d: this arm ran %d tokens, off's sequence has %d --"
                     " the arms used different NTOK, so nothing here is comparable"
                     % (pi2, len(r["comp"]), len(e["comp"])))
                j = next((k for k, (a, b) in enumerate(zip(r["comp"], e["comp"]))
                          if a != b), len(r["comp"]))
                print("NOTE free-run diverges from off at token %d (a RESULT,"
                      " not an input choice)" % j, flush=True)
            r["comp_scored"] = e["comp"]      # always off's sequence

    # ---- pass 2: teacher-force the SAME sequence, prefill-route top-32 ----
    sp2 = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=32)
    scored = [{"prompt_ids": r["prompt_ids"],
               "comp": r.get("comp_scored", r["comp"])} for r in runs]
    tf = llm.generate([{"prompt_token_ids": s["prompt_ids"] + s["comp"]}
                       for s in scored], sp2, use_tqdm=False)
    res = []
    for pi, (r, out) in enumerate(zip(runs, tf)):
        seq = r["prompt_ids"] + (r["comp_scored"] if "comp_scored" in r
                                 else r["comp"])
        prefill = summarise(out, "prefill")
        # The decode-route twin of those rows, on the same forced prefix. Positions
        # before prompt_len are prompt tokens: the free run never ran a decode step
        # there, so the gate's floor would have nothing to compare. PROBE_N bounds
        # it for a smoke run.
        df_rows, df_drop = forced_decode_probe(
            llm, seq, len(r["prompt_ids"]), min(PROBE_N, NTOK), 32)
        e = {"arm": ARM, "prompt_len": len(r["prompt_ids"]),
             "n_scored": NTOK, "n_probed": min(PROBE_N, NTOK),
             "comp_own": r["comp"], "comp_scored": r["comp_scored"]
             if "comp_scored" in r else r["comp"],
             "prefill": prefill,
             "decode_forced": df_rows, "decode_forced_dropped": df_drop,
             "decode": r["decode"]}
        if "free_run_matches_off" in r:
            e["free_run_matches_off"] = r["free_run_matches_off"]
        res.append(e)
        print("[%s] prompt %d: prefill rows=%d decode_forced=%d dropped=%d "
              "decode(free)=%d" % (ARM, pi, len(prefill), len(df_rows),
                                   len(df_drop), len(r["decode"])), flush=True)
    path = os.path.join(OUT, "l3_%s.json" % ARM)
    json.dump({"arm": ARM, "ntok": NTOK, "rows": res,
               "conditions": {"clock_mhz_env": os.environ.get("CLOCK_MHZ"),
                              "cache_root": os.environ.get("VLLM_CACHE_ROOT"),
                              "p2_gemv": os.environ.get("VLLM_WNA16_P2_GEMV"),
                              "probe_n": PROBE_N}},
              open(path, "w"))
    print("wrote %s (%.1f MB) in %.0f s"
          % (path, os.path.getsize(path) / 1e6, time.time() - t0), flush=True)
    print("L3-RUN-DONE arm=%s" % ARM, flush=True)


if __name__ == "__main__":
    main()
