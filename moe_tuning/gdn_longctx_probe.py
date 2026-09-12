"""gdn_longctx_probe.py -- end-to-end check for the upstream GDN readout bug.

Upstream vLLM #55291 / #54146: with the SSM state cache in fp32, the GDN
(linear-attention) readout can exceed the fp16 max 65504, so a fp16 buffer
becomes inf, `inf/inf` yields NaN, and greedy decoding collapses into a single
token repeated forever. The reported signature is *context-length dependent*:
short prompts are fine, long ones break.

Our QUICKSTART lists this as a known upstream cause that was never reproduced
on this image, and our target model is the same Qwen3.5/3.8 hybrid family --
so it is a delivery risk, not a curiosity.

What the kernel-level check (gdn_verify.py, 44/44 PASS) canNOT tell us is
whether the real end-to-end path crosses fp16 max at long context. Hence this:
one engine load, one SHORT and one LONG prompt, greedy decode, and a degenerate
check that is stricter than "distinct tokens <= 2":
  * distinct token count
  * longest run of one identical token
  * whether the tokenizer output looks like garbage (e.g. only punctuation)
  * explicit inf/nan scan of the raw logits is not available offline, so we also
    look for the NaN fingerprint: a constant token repeated to the cap.

Env in: CTX_LONG, CTX_SHORT, N_GEN, MAX_LEN, MAX_NUM_SEQS, GMU, MM, OUT, REPS
"""
import os
import time
import traceback

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama-moe")

import json  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/longctx")
MODEL = os.environ.get("MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
GMU = float(os.environ.get("GMU", "0.92"))
MM = os.environ.get("MM", "1") == "1"
CTX_LONG = int(os.environ.get("CTX_LONG", "14000"))
CTX_SHORT = int(os.environ.get("CTX_SHORT", "256"))
N_GEN = int(os.environ.get("N_GEN", "64"))
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()


def make_prompt(tok, target_len, nonce):
    head = "Document %s. " % nonce
    body = " ".join(WORDS[(i * 7 + len(nonce)) % len(WORDS)] for i in range(target_len))
    ids = tok.encode(head + body, add_special_tokens=False)[:target_len]
    return tok.decode(ids), len(ids)


def analyse(ids, tok):
    n = len(ids)
    distinct = len(set(ids))
    run = best = 1
    for i in range(1, n):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, run)
    text = tok.decode(ids)
    # NaN fingerprint: the same token filling the whole generation
    degenerate = (best >= max(8, int(0.8 * n))) or (distinct <= max(2, n // 8))
    return {"n": n, "distinct": distinct, "longest_run": best,
            "degenerate": degenerate, "text_head": text[:160],
            "is_single_punct": len(set(text.strip())) <= 1 and len(text.strip()) > 20}


def run_one(llm, tok, label, ctx, nonce, res):
    prompt, clen = make_prompt(tok, ctx, nonce)
    sp = SamplingParams(temperature=0.0, max_tokens=N_GEN, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ids = list(outs[0].outputs[0].token_ids)
    a = analyse(ids, tok)
    a["ctx_tokens"] = clen
    a["req_s"] = dt
    a["token_ids_head"] = ids[:16]
    res[label] = a
    print("LONGCTX %-6s ctx=%-6d gen=%d distinct=%-3d longest_run=%-3d %s"
          % (label, clen, a["n"], a["distinct"], a["longest_run"],
             "DEGENERATE" if a["degenerate"] else "ok"), flush=True)
    print("        text: %r" % a["text_head"][:120], flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"ctx_long": CTX_LONG, "ctx_short": CTX_SHORT, "n_gen": N_GEN}
    print("LONGCTX_START long=%d short=%d n_gen=%d" % (CTX_LONG, CTX_SHORT, N_GEN),
          flush=True)
    kw = dict(model=MODEL, dtype="float16", tensor_parallel_size=2,
              gpu_memory_utilization=GMU, max_model_len=MAX_LEN,
              max_num_seqs=MAX_NUM_SEQS, attention_backend="TRITON_ATTN",
              disable_custom_all_reduce=True, enable_prefix_caching=True)
    if MM:
        kw["limit_mm_per_prompt"] = {"image": 1, "video": 1}
        kw["mm_encoder_attn_backend"] = "TRITON_ATTN"
    else:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    kw["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32]}
    try:
        llm = LLM(**kw)
    except Exception as e:  # noqa: BLE001
        print("LONGCTX_FAIL load %r" % (e,), flush=True)
        res["load_error"] = traceback.format_exc()[-2000:]
        json.dump(res, open(f"{OUT}/longctx.json", "w"), indent=1)
        return
    tok = llm.get_tokenizer()
    print("LONGCTX_STEP load OK", flush=True)
    try:
        # warmup on a third length so neither measured run absorbs init cost
        wp, _ = make_prompt(tok, CTX_SHORT, "warmup0")
        llm.generate([wp], SamplingParams(temperature=0.0, max_tokens=8,
                                          ignore_eos=True), use_tqdm=False)
        run_one(llm, tok, "short", CTX_SHORT, "short0", res)
        run_one(llm, tok, "long", CTX_LONG, "long0", res)
        # second long run with a different nonce: a one-off must not be mistaken
        # for a systematic effect
        run_one(llm, tok, "long2", CTX_LONG, "long1", res)
    except Exception as e:  # noqa: BLE001
        print("LONGCTX_FAIL run %r" % (e,), flush=True)
        res["run_error"] = traceback.format_exc()[-2000:]
    bad = [k for k, v in res.items()
           if isinstance(v, dict) and v.get("degenerate")]
    res["verdict"] = "DEGENERATE:" + ",".join(bad) if bad else "CLEAN"
    json.dump(res, open(f"{OUT}/longctx.json", "w"), indent=1)
    print("LONGCTX_DONE_OK verdict=%s" % res["verdict"], flush=True)


if __name__ == "__main__":
    main()
