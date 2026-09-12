"""gdn_shortq_probe.py -- long-context COHERENCE check via a short question.

Why this exists: gdn_longctx_probe.py only proved the NaN fingerprint is absent
(no single-token infinite repeat). Its filler was a 10-word cycle, so the model
just parroted the cycle and the output said nothing about coherence.

This probe asks an unambiguous short question AFTER a long filler context, and
checks for a specific expected substring. Three questions x two context lengths,
same engine, greedy:

    short (256 tok)  -> control: the question must be answerable at all
    long  (14000 tok)-> the actual test

A model whose GDN readout has overflowed into NaN cannot answer any of them, and
the failure is context-dependent by construction, so the short/long pair is the
discriminator.

Capture sizes MUST match p5_decode.py / gdn_longctx_probe.py ([1..32]); any
change alters the torch.compile cache key and costs a full 6-minute recompile
(measured 2026-09-11).

Env in: CTX_LONG, CTX_SHORT, N_GEN, MAX_LEN, MAX_NUM_SEQS, GMU, MM, OUT
"""
import json
import os
import re
import time
import traceback

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama-moe")

from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/shortq")
MODEL = os.environ.get("MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
GMU = float(os.environ.get("GMU", "0.92"))
MM = os.environ.get("MM", "1") == "1"
CTX_LONG = int(os.environ.get("CTX_LONG", "14000"))
CTX_SHORT = int(os.environ.get("CTX_SHORT", "256"))
N_GEN = int(os.environ.get("N_GEN", "96"))
# ignore_eos=True made the model spray <|im_start|> forever after a
# correct answer, which looks like a collapse but is purely a sampling
# artifact. Default OFF so natural termination is observable.
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"

FILLER = ("The archive records routine maintenance notes for the facility. "
          "Nothing in this section requires action or follow-up. ")

# (question, expected-substring regex) -- substring matching tolerates a
# thinking model's  thinking preamble as long as the answer is present.
QUESTIONS = [
    ("What is 17 times 23? Reply with only the number.", r"391"),
    ("In one word, name the colour of a clear daytime sky.", r"(?i)blue"),
    ("Write the word BANANA three times separated by spaces.", r"(?i)banana\s+banana\s+banana"),
]


def make_prompt(tok, ctx, nonce, question):
    """Filler to ~ctx tokens, then a clearly delimited short question."""
    head = "Document %s. " % nonce
    body = FILLER * max(1, ctx // 12)
    tail = "\n\nQuestion: " + question + "\nAnswer:"
    ids = tok.encode(head + body, add_special_tokens=False)
    ids = ids[:ctx]
    ids = ids + tok.encode(tail, add_special_tokens=False)
    return tok.decode(ids), len(ids)


def analyse(ids, tok, expect):
    n = len(ids)
    run = best = 1
    for i in range(1, n):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, run)
    text = tok.decode(ids)
    matched = bool(re.search(expect, text))
    return {"n": n, "distinct": len(set(ids)), "longest_run": best,
            "matched": matched, "text": text[:220],
            "collapsed": best >= max(8, int(0.8 * n))}


def run_one(llm, tok, label, ctx, nonce, qi, q, expect, res):
    prompt, clen = make_prompt(tok, ctx, nonce, q)
    sp = SamplingParams(temperature=0.0, max_tokens=N_GEN,
                         ignore_eos=IGNORE_EOS)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    toks = list(outs[0].outputs[0].token_ids)
    a = analyse(toks, tok, expect)
    # healthy = stopped on its own before the cap (EOS honoured)
    a["terminated_naturally"] = len(toks) < N_GEN
    a.update({"ctx_tokens": clen, "question": q, "expect": expect, "req_s": dt})
    res["%s_q%d" % (label, qi)] = a
    print("SHORTQ %-5s q%d ctx=%-6d len=%-3d run=%-3d matched=%-5s %s"
          % (label, qi, clen, a["n"], a["longest_run"], a["matched"],
             "COLLAPSED" if a["collapsed"] else ""), flush=True)
    print("        out: %r" % a["text"][:150], flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"ctx_long": CTX_LONG, "ctx_short": CTX_SHORT, "n_gen": N_GEN}
    print("SHORTQ_START long=%d short=%d questions=%d" % (CTX_LONG, CTX_SHORT,
                                                          len(QUESTIONS)), flush=True)
    kw = dict(model=MODEL, dtype="float16", tensor_parallel_size=2,
              gpu_memory_utilization=GMU, max_model_len=MAX_LEN,
              max_num_seqs=MAX_NUM_SEQS, attention_backend="TRITON_ATTN",
              disable_custom_all_reduce=True, enable_prefix_caching=True)
    if MM:
        kw["limit_mm_per_prompt"] = {"image": 1, "video": 1}
        kw["mm_encoder_attn_backend"] = "TRITON_ATTN"
    else:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    # MUST equal the sizes used by the previous probes or torch.compile redoes
    # the whole 6-minute AOT pass.
    kw["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32]}
    try:
        llm = LLM(**kw)
    except Exception as e:  # noqa: BLE001
        print("SHORTQ_FAIL load %r" % (e,), flush=True)
        res["load_error"] = traceback.format_exc()[-2000:]
        json.dump(res, open(f"{OUT}/shortq.json", "w"), indent=1)
        return
    print("SHORTQ_STEP load OK", flush=True)
    tok = llm.get_tokenizer()
    try:
        for qi, (q, expect) in enumerate(QUESTIONS):
            run_one(llm, tok, "short", CTX_SHORT, "s%d" % qi, qi, q, expect, res)
            run_one(llm, tok, "long", CTX_LONG, "L%d" % qi, qi, q, expect, res)
    except Exception as e:  # noqa: BLE001
        print("SHORTQ_FAIL run %r" % (e,), flush=True)
        res["run_error"] = traceback.format_exc()[-2000:]

    sh = [v["matched"] for k, v in res.items() if k.startswith("short_q")]
    lg = [v["matched"] for k, v in res.items() if k.startswith("long_q")]
    coll = [k for k, v in res.items() if isinstance(v, dict) and v.get("collapsed")]
    res["short_ok"] = "%d/%d" % (sum(sh), len(sh))
    res["long_ok"] = "%d/%d" % (sum(lg), len(lg))
    res["collapsed_keys"] = coll
    print("SHORTQ_SUMMARY short=%s long=%s collapsed=%s"
          % (res["short_ok"], res["long_ok"], coll or "none"), flush=True)
    json.dump(res, open(f"{OUT}/shortq.json", "w"), indent=1)
    print("SHORTQ_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
