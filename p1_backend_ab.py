"""P1: attention-backend A/B on the delivery geometry (single arm per process).

Answers v2-plan #63: is `--attention-backend ROCM_ATTN` faster than the shipped
TRITON_ATTN at prefill AND decode, at 2K / 8K / 16K context?

Context: ROCM_ATTN has two decode branches in 0.28, selected by
`platforms/rocm.py use_rocm_custom_paged_attention`. For our geometry
(head 256, block 784, gqa 5) that predicate is already False, so both backends
end up on Triton for decode -- the residual risk is the *context* (prefill)
kernel `context_attention_fwd` at head 256, which has never been run here.
So this probe must cover prefill, not just decode.

One process per arm (the engine is built once), driven by p1_backend_ab.sh.
Env in:  ATTN_BACKEND, OUT, MODEL, REPS
Env out: <OUT>/p1_wall.json

The `__main__` guard is mandatory: vLLM spawns TP workers with multiprocessing
"spawn", so an unguarded top-level LLM() makes every child re-enter the engine.
The env backstop is mandatory too: calling LLM() directly bypasses gfx906-serve,
and without VLLM_WNA16_EXLLAMA the W4A16 path silently falls back to Triton
(3.9 vs 22 tok/s) -- an invalid attribution that looks like a measurement.
"""
import json
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
_W4A16 = "exllama" if os.environ.get("VLLM_WNA16_EXLLAMA") == "1" else "triton"
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-{_W4A16}")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ARM = os.environ.get("ATTN_BACKEND", "TRITON_ATTN")
OUT = os.environ.get("OUT", "/work/p1")
MODEL = os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
REPS = int(os.environ.get("REPS", "2"))
CTXS = [int(x) for x in os.environ.get("CTXS", "2048,8192,16256").split(",")]
OUT_TOK = 64
BASE = "Explain in detail why the sky is blue and what determines its shade at dawn. "
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48]


def make_prompt(tok, n):
    """Exactly ~n tokens: build the id list, then decode it back."""
    ids = tok.encode(BASE, add_special_tokens=False)
    reps = n // len(ids) + 1
    return tok.decode((ids * reps)[:n])


def gen(llm, prompt, max_tokens):
    # ignore_eos is mandatory for an A/B: without it an early EOS truncates the
    # greedy run and the arm that stops first reads fast on a shorter window.
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    o = out[0]
    return dt, o


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"=== P1 arm {ARM} ===", flush=True)
    print("env:", {k: os.environ.get(k) for k in (
        "HSA_OVERRIDE_GFX_VERSION", "VLLM_TRITON_ATTN_NUM_SEGMENTS",
        "VLLM_WNA16_EXLLAMA", "VLLM_CACHE_ROOT")}, flush=True)

    t_load = time.perf_counter()
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
        attention_backend=ARM, disable_custom_all_reduce=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": CAPTURE_SIZES},
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN",
    )
    load_s = time.perf_counter() - t_load
    print(f"engine ready after {load_s:.1f} s", flush=True)

    tok = llm.get_tokenizer()
    rows = []
    for ctx in CTXS:
        prompt = make_prompt(tok, ctx)
        n_actual = len(tok.encode(prompt, add_special_tokens=False))
        # prefill-only: 1 output token, wall ~= TTFT
        pf = min(gen(llm, prompt, 1)[0] for _ in range(REPS))
        # decode: same prompt, 64 greedy tokens; subtract the prefill leg
        best = min((gen(llm, prompt, OUT_TOK) for _ in range(REPS)),
                   key=lambda t: t[0])
        req, o = best
        n_out = len(o.outputs[0].token_ids)
        dec = (req - pf) / max(n_out - 1, 1)
        row = {"ctx_requested": ctx, "ctx_actual": n_actual,
               "prefill_s": pf, "prefill_tps": n_actual / pf,
               "req_s": req, "n_out": n_out,
               "decode_ms_per_step": dec * 1e3, "decode_tps": 1 / dec,
               "token_ids_head": list(o.outputs[0].token_ids[:8])}
        rows.append(row)
        print("  ctx %6d (actual %6d)  prefill %7.1f tok/s (%6.1f s)   "
              "decode %6.2f tok/s (%6.1f ms/step)"
              % (ctx, n_actual, row["prefill_tps"], pf,
                 row["decode_tps"], dec * 1e3), flush=True)

    json.dump({"arm": ARM, "load_s": load_s, "rows": rows,
               "model": MODEL, "capture_sizes": CAPTURE_SIZES},
              open(f"{OUT}/p1_wall.json", "w"), indent=1)
    print(f"P1_ARM_DONE {ARM}", flush=True)


if __name__ == "__main__":
    main()
