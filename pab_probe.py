"""Generic one-arm probe for the remaining zero-recompile A/B items.

Used by pab_run.sh, which starts one process per arm (the engine config cannot
be changed at runtime). Serves P3 (GDN output-norm layout) and P4 (FP8 KV).

Env in:
  LABEL           arm name, also the output subdir            (required)
  OUT             root output dir                             (/work/ab)
  ATTN_BACKEND    default TRITON_ATTN
  KV_CACHE_DTYPE  e.g. fp8_e4m3; empty => leave at engine default
  MAX_LEN         default 16384
  MAX_NUM_SEQS    default 48
  CTXS            default 2048,8192
  REPS            default 2
  DO_PREFILL      1 => also time a 1-token request as the prefill leg
  MODEL           default the 27B AWQ W4A16

Env out: <OUT>/<LABEL>/wall.json
  rows[i] = {ctx_requested, ctx_actual, prefill_s, prefill_tps, req_s,
             n_out, decode_ms_per_step, decode_tps, token_ids}

Why it is shaped like this:
  * one engine per process  -- attention backend / kv-cache-dtype are fixed at
    construction, so an arm cannot be switched in-place.
  * greedy + ignore_eos     -- without ignore_eos the arm that hits EOS first
    reads fast on a shorter window; both per_step and tok/s become arm noise.
  * token_ids are recorded  -- the acceptance criterion is engine-level tok/s
    PLUS greedy token-id equality, per the project's kernel-swap rule. Offline
    "all tensors match" is explicitly not accepted (packed-view stride trap).
  * __main__ guard is mandatory: vLLM spawns TP workers with mp "spawn".
  * env backstop is mandatory: calling LLM() directly bypasses gfx906-serve and
    without VLLM_WNA16_EXLLAMA the W4A16 path silently falls back to Triton
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

LABEL = os.environ["LABEL"]
OUT = os.path.join(os.environ.get("OUT", "/work/ab"), LABEL)
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "TRITON_ATTN")
KV_DTYPE = os.environ.get("KV_CACHE_DTYPE", "")
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "48"))
CTXS = [int(x) for x in os.environ.get("CTXS", "2048,8192").split(",")]
REPS = int(os.environ.get("REPS", "2"))
DO_PREFILL = os.environ.get("DO_PREFILL", "1") == "1"
OUT_TOK = 64
MODEL = os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
BASE = "Explain in detail why the sky is blue and what determines its shade at dawn. "
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48]


def make_prompt(tok, n):
    ids = tok.encode(BASE, add_special_tokens=False)
    return tok.decode((ids * (n // len(ids) + 1))[:n])


def gen(llm, prompt, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp, use_tqdm=False)
    return time.perf_counter() - t0, out[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"=== ARM {LABEL} ===", flush=True)
    print("env:", {k: os.environ.get(k) for k in (
        "HSA_OVERRIDE_GFX_VERSION", "VLLM_TRITON_ATTN_NUM_SEGMENTS",
        "VLLM_WNA16_EXLLAMA", "VLLM_CACHE_ROOT",
        "VLLM_GFX906_GDN_OUTPUT_NORM")}, flush=True)
    print(f"knobs: backend={ATTN_BACKEND} kv_cache_dtype={KV_DTYPE or '<default>'} "
          f"max_len={MAX_LEN} max_num_seqs={MAX_NUM_SEQS}", flush=True)

    kw = {}
    if KV_DTYPE:
        kw["kv_cache_dtype"] = KV_DTYPE
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=MAX_LEN,
        max_num_seqs=MAX_NUM_SEQS, attention_backend=ATTN_BACKEND,
        disable_custom_all_reduce=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": CAPTURE_SIZES},
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN", **kw,
    )
    load_s = time.perf_counter() - t0
    print(f"engine ready after {load_s:.1f} s", flush=True)

    # KV capacity is what P4 is actually about -- pull it out of the engine log.
    try:
        import glob
        cap = None
        for lg in glob.glob("/tmp/*.log") + glob.glob(os.path.expanduser("~/*.log")):
            with open(lg, errors="ignore") as fh:
                for line in fh:
                    if "GPU KV cache size:" in line:
                        cap = line.strip()
        if cap:
            print("kv_capacity:", cap, flush=True)
    except Exception:  # noqa: BLE001
        cap = None

    tok = llm.get_tokenizer()
    rows = []
    for ctx in CTXS:
        prompt = make_prompt(tok, ctx)
        n_actual = len(tok.encode(prompt, add_special_tokens=False))
        pf = min(gen(llm, prompt, 1)[0] for _ in range(REPS)) if DO_PREFILL else 0.0
        req, o = min((gen(llm, prompt, OUT_TOK) for _ in range(REPS)),
                     key=lambda t: t[0])
        ids = list(o.outputs[0].token_ids)
        dec = (req - pf) / max(len(ids) - 1, 1)
        rows.append({"ctx_requested": ctx, "ctx_actual": n_actual,
                     "prefill_s": pf,
                     "prefill_tps": (n_actual / pf) if pf else None,
                     "req_s": req, "n_out": len(ids),
                     "decode_ms_per_step": dec * 1e3, "decode_tps": 1 / dec,
                     "token_ids": ids})
        print("  ctx %6d (actual %6d)  prefill %s  decode %6.2f tok/s (%6.1f ms/step)"
              % (ctx, n_actual,
                 ("%7.1f tok/s" % (n_actual / pf)) if pf else "   n/a   ",
                 1 / dec, dec * 1e3), flush=True)

    json.dump({"label": LABEL, "load_s": load_s, "backend": ATTN_BACKEND,
               "kv_cache_dtype": KV_DTYPE, "max_len": MAX_LEN,
               "max_num_seqs": MAX_NUM_SEQS, "rows": rows,
               "gdn_output_norm": os.environ.get("VLLM_GFX906_GDN_OUTPUT_NORM", "0"),
               "kv_capacity_line": cap},
              open(f"{OUT}/wall.json", "w"), indent=1)
    print(f"PAB_ARM_DONE {LABEL}", flush=True)


if __name__ == "__main__":
    main()
