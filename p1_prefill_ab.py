"""P1-corrected: prefill A/B without the Prefix-Cache trap.

Why this file exists
--------------------
The first P1 run (p1_backend_ab.py) reported prefill at 2569 / 7739 / 7914 tok/s
against a delivery baseline of 434.8 / 338.1 / 249.9 -- i.e. 6x..30x too fast.
Cause: REPS identical prompts + vLLM's prefix cache ON. The second rep reuses the
KV built by the first, so `min()` over reps returns the *cached* run and the
prefill leg reads as nearly free. Decode is unaffected (it subtracts that same
cached prefill), which is why decode matched the baseline to within 0.5 tok/s.

This is exactly the trap the z100 author documents, and the one this repo has
already mis-measured three times. Two independent guards here:

  1. `enable_prefix_caching=False` on the engine  -- removes the mechanism.
  2. a unique nonce at the head of every request  -- so even if (1) is ignored
     by a future vLLM, no two requests can share a prefix.

Prompt length is read back from the engine (`prompt_token_ids`), never inferred
from the text, because decode()->encode() is not length-preserving.

Env in:  ATTN_BACKEND, OUT, MODEL, CTXS, REPS
Env out: <OUT>/prefill.json
"""
import json
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ARM = os.environ.get("ATTN_BACKEND", "TRITON_ATTN")
OUT = os.environ.get("OUT", "/work/pf")
MODEL = os.environ.get(
    "MODEL", "/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
CTXS = [int(x) for x in os.environ.get("CTXS", "2048,8192,16256").split(",")]
REPS = int(os.environ.get("REPS", "3"))
BASE = "Explain in detail why the sky is blue and what determines its shade at dawn. "


def make_prompt(tok, n, nonce):
    """~n tokens with a unique head, so no two requests share a prefix."""
    head = tok.encode("Request %d ref %d:" % (nonce, nonce * 7919),
                      add_special_tokens=False)
    body = tok.encode(BASE, add_special_tokens=False)
    reps = max(1, (n - len(head)) // len(body) + 1)
    return tok.decode((head + body * reps)[:n])


def gen(llm, prompt, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp, use_tqdm=False)
    return time.perf_counter() - t0, out[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"=== PREFILL ARM {ARM} (prefix caching OFF, unique nonce) ===", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
        attention_backend=ARM, disable_custom_all_reduce=True,
        enable_prefix_caching=False,          # guard #1
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48]},
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN",
    )
    print(f"engine ready after {time.perf_counter()-t0:.1f} s", flush=True)

    tok = llm.get_tokenizer()
    rows = []
    nonce = 1000
    for ctx in CTXS:
        best = None
        for _ in range(REPS):
            nonce += 1
            prompt = make_prompt(tok, ctx, nonce)      # guard #2
            dt, o = gen(llm, prompt, 1)
            n_actual = len(o.prompt_token_ids)
            print("    rep ctx~%6d actual=%6d  prefill %8.1f tok/s (%6.3f s)"
                  % (ctx, n_actual, n_actual / dt, dt), flush=True)
            if best is None or dt < best[0]:
                best = (dt, n_actual)
        dt, n_actual = best
        rows.append({"ctx_requested": ctx, "ctx_actual": n_actual,
                     "prefill_s": dt, "prefill_tps": n_actual / dt})
        print("  ctx %6d (actual %6d)  prefill %8.1f tok/s (%6.3f s)  <- min over %d distinct requests"
              % (ctx, n_actual, n_actual / dt, dt, REPS), flush=True)

    json.dump({"arm": ARM, "rows": rows, "reps": REPS,
               "prefix_caching": False}, open(f"{OUT}/prefill.json", "w"), indent=1)
    print(f"P1PF_ARM_DONE {ARM}", flush=True)


if __name__ == "__main__":
    main()
