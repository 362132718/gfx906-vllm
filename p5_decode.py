"""p5_decode.py -- an honest decode-step measurement for the MoE model.

Why this file exists (2026-09-11): every decode number we produced today used

    prefill_s = time(generate(prompt, max_tokens=1))
    req_s     = time(generate(prompt, max_tokens=N))      # SAME prompt
    step      = (req_s - prefill_s) / (N - 1)

The second call hits vLLM's prefix cache (on by default), so its prefill is
~free while `prefill_s` is a *cold* prefill. The formula therefore subtracts a
full prefill from a request that never paid it, understating the step by
P/(N-1) -- with P ~ 2.3 s and N = 64 that is ~37 ms, larger than the step
itself. Symptom that exposed it: graph mode reported 9.73 ms/step while the
torch-profiler bucket table showed ~32.5 ms/step of device-busy per rank.

Fix, two independent guarantees:
  1. prefix caching OFF (`enable_prefix_caching=False`), and
  2. two DIFFERENT prompts of the SAME token length, both cold:
         T_A = P + 1*S          (max_tokens = 1)
         T_B = P + N*S          (max_tokens = N, different prompt)
         S   = (T_B - T_A) / (N - 1)
     The equal-length cold prefill P cancels exactly.

Prompts carry a random nonce so prompt_token_ids are unique, and the REAL
length is read back from the tokenizer (never inferred from the text).

Env in: EAGER, MAX_LEN, MAX_NUM_SEQS, GMU, PP, MM, ATTN_BACKEND, OUT,
        N_GEN, CTX, CONC
Env out: <OUT>/decode.json + DECODE_* markers on stdout.
"""
import json
import os
import random
import string
import time
import traceback

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama-moe")

from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/p5dec")
EAGER = os.environ.get("EAGER", "0") == "1"
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "48"))
GMU = float(os.environ.get("GMU", "0.90"))
PP = int(os.environ.get("PP", "1"))
MM = os.environ.get("MM", "1") == "1"
CONC = int(os.environ.get("CONC", "8"))
# Default OFF (measurement hygiene: prefix caching inflates prefill).
# Set PREFIX_CACHE=1 to reproduce the delivery config, where serve.sh
# leaves prefix caching at vLLM default = ON, which also flips the
# mamba cache mode to "align" and changes the Mamba block count.
PREFIX_CACHE = os.environ.get("PREFIX_CACHE", "0") == "1"
REPS = int(os.environ.get("REPS", "3"))
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "TRITON_ATTN")
N_GEN = int(os.environ.get("N_GEN", "64"))
CTX = int(os.environ.get("CTX", "1600"))
MODEL = os.environ.get("MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()


def make_prompt(tok, target_len, nonce):
    """Repeat filler to ~target_len tokens, then read the length back."""
    head = "Document %s. " % nonce
    body = " ".join(random.Random(nonce).choice(WORDS) for _ in range(target_len))
    ids = tok.encode(head + body, add_special_tokens=False)[:target_len]
    return tok.decode(ids), len(ids)


def timed(llm, prompts, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    return dt, outs


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"eager": EAGER, "max_len": MAX_LEN, "pp": PP, "mm": MM,
           "n_gen": N_GEN, "prefix_cache": PREFIX_CACHE, "max_num_seqs": MAX_NUM_SEQS}
    print("DECODE_START eager=%s max_len=%d pp=%d mm=%s ctx=%d conc=%d"
          % (EAGER, MAX_LEN, PP, MM, CTX, CONC), flush=True)

    kw = dict(
        model=MODEL, dtype="float16",
        tensor_parallel_size=(1 if PP > 1 else 2),
        gpu_memory_utilization=GMU, max_model_len=MAX_LEN,
        max_num_seqs=MAX_NUM_SEQS, attention_backend=ATTN_BACKEND,
        disable_custom_all_reduce=True,
        enable_prefix_caching=PREFIX_CACHE,
    )
    if PP > 1:
        kw["pipeline_parallel_size"] = PP
    if MM:
        kw["limit_mm_per_prompt"] = {"image": 1, "video": 1}
        kw["mm_encoder_attn_backend"] = "TRITON_ATTN"
    else:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if EAGER:
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                    "cudagraph_capture_sizes": [1, 2, 4, 8,
                                                                16, 24, 32, 48]}
    t0 = time.perf_counter()
    try:
        llm = LLM(**kw)
    except Exception as e:  # noqa: BLE001
        res["load_error"] = "%s: %s" % (type(e).__name__, e)
        res["tb"] = traceback.format_exc()[-3000:]
        json.dump(res, open(f"{OUT}/decode.json", "w"), indent=1)
        print("DECODE_FAIL load %r" % (e,), flush=True)
        return
    res["load_s"] = time.perf_counter() - t0
    print("DECODE_STEP load OK %.0fs" % res["load_s"], flush=True)

    tok = llm.get_tokenizer()

    # ---- single stream ------------------------------------------------------
    # WARMUP + REPS are load-bearing, not polish. The first request after engine
    # init pays one-time costs (triton JIT, allocator growth, first-touch), and
    # because the step is derived as (tB - tA)/(N-1) any inflation of tA shrinks
    # the difference and UNDER-reports the step. Measured on 2026-09-11: without
    # warmup the delivery config reported 8.89 ms/step = 112 tok/s, which is
    # impossible -- the profiler puts single-rank device-busy at ~18 ms/step.
    # So: warm up on a third prompt, then repeat the (A,B) pair REPS times and
    # take the median. Different nonces keep the two prompts distinct so prefix
    # caching cannot collapse the prefill term.
    try:
        pc, _ = make_prompt(tok, CTX, "warmup")
        timed(llm, [pc], 8)                       # warmup, discarded
        steps, tAs, tBs = [], [], []
        for r in range(REPS):
            pa, la = make_prompt(tok, CTX, "singleA%d" % r)
            pb, lb = make_prompt(tok, CTX, "singleB%d" % r)
            assert abs(la - lb) <= 2, (la, lb)
            tA, _ = timed(llm, [pa], 1)           # cold prefill + 1 step
            tB, oB = timed(llm, [pb], N_GEN)      # cold prefill + N steps
            nB = len(oB[0].outputs[0].token_ids)
            steps.append((tB - tA) / max(nB - 1, 1))
            tAs.append(tA); tBs.append(tB)
        steps.sort()
        step = steps[len(steps) // 2]             # median
        tA = sorted(tAs)[len(tAs) // 2]
        tB = sorted(tBs)[len(tBs) // 2]
        res["single"] = {"ctx": la, "t_prefill_plus_1": tA, "t_prefill_plus_N": tB,
                         "n_gen": nB, "step_ms": step * 1e3, "tok_per_s": 1 / step,
                         "prefill_tok_per_s": la / max(tA - step, 1e-9),
                         "reps": REPS, "all_steps_ms": [s * 1e3 for s in steps],
                         "warmup": True}
        print("DECODE_STEP single OK ctx=%d  step=%.2f ms  %.2f tok/s  prefill=%.0f tok/s  reps=%d %s"
              % (la, step * 1e3, 1 / step, la / max(tA - step, 1e-9), REPS,
                 ["%.2f" % (s * 1e3) for s in steps]), flush=True)
    except Exception as e:  # noqa: BLE001
        res["single_error"] = "%s: %s" % (type(e).__name__, e)
        res["single_tb"] = traceback.format_exc()[-2500:]
        print("DECODE_STEP single FAIL %r" % (e,), flush=True)

    # ---- concurrency: same two-prompt trick, CONC prompts each --------------
    try:
        ma = [make_prompt(tok, CTX, "concA%d" % i)[0] for i in range(CONC)]
        mb = [make_prompt(tok, CTX, "concB%d" % i)[0] for i in range(CONC)]
        tA, _ = timed(llm, ma, 1)
        tB, oB = timed(llm, mb, N_GEN)
        nB = sum(len(o.outputs[0].token_ids) for o in oB)
        step = (tB - tA) / max(N_GEN - 1, 1)
        agg = nB / (tB - tA)
        res["conc"] = {"n": CONC, "t_prefill_plus_1": tA, "t_prefill_plus_N": tB,
                       "tot_tok": nB, "step_ms": step * 1e3,
                       "agg_tok_per_s": agg, "per_stream": agg / CONC}
        print("DECODE_STEP conc%d OK agg=%.1f tok/s (%.2f per stream, step %.2f ms)"
              % (CONC, agg, agg / CONC, step * 1e3), flush=True)
    except Exception as e:  # noqa: BLE001
        res["conc_error"] = "%s: %s" % (type(e).__name__, e)
        print("DECODE_STEP conc FAIL %r" % (e,), flush=True)

    json.dump(res, open(f"{OUT}/decode.json", "w"), indent=1)
    print("DECODE_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
