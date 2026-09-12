"""p5_lin_census.py -- which fp16 linear escapes the P0.5 skinny path?

Context (2026-09-11): the delivery-mode decode profile shows
`Cijk_Ailk_Bljk_HHS_BH_MT64x64x16` at 40 calls/forward, 151 us each,
6.03 ms/step (25% of the step). The P0.5 gfx906 fast path
(`docker/patches/layers_utils.py:209-219`) only fires when

    VLLM_ROCM_USE_SKINNY_GEMM and on_gfx906() and dtype in (fp16, bf16)
    and bias is None and n == 1 and m % 4 == 0 and k % 8 == 0
    and k <= 8192 and skinny_operands_compatible

where  n = x.numel() // x.size(-1)   (= token count)
       m = weight.shape[0]           (= output features)
       k = weight.shape[1]           (= reduction dim)
and anything that fails falls through to torch.nn.functional.linear -> rocBLAS.

Analytical enumeration of this model predicts every fp16 linear SHOULD qualify
(hidden 2048, largest k = 4096), yet 40 calls/forward still land on rocBLAS.
So instead of guessing, count them: patch torch.nn.functional.linear, run one
short decode, and report the exact (n, m, k, has_bias, contiguity) histogram
plus the calling frames of the vLLM model code.

Env in: EAGER, MAX_LEN, MAX_NUM_SEQS, GMU, MM, OUT, N_GEN
"""
import collections
import json
import os
import time
import traceback

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama-moe")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/p5lin")
EAGER = os.environ.get("EAGER", "1") == "1"
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "48"))
GMU = float(os.environ.get("GMU", "0.92"))
MM = os.environ.get("MM", "0") == "1"
N_GEN = int(os.environ.get("N_GEN", "4"))
MODEL = os.environ.get("MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")

STATS = collections.Counter()
STACKS = {}
_orig_linear = F.linear
_ENABLED = {"on": False}


def _key(x, weight, bias):
    n = x.numel() // x.size(-1) if x.numel() else 0
    m = int(weight.shape[0])
    k = int(weight.shape[1])
    return (n, m, k, str(x.dtype).replace("torch.", ""),
            bias is not None, bool(weight.is_contiguous()),
            bool(x.is_contiguous()))


def _patched_linear(x, weight, bias=None, *a, **kw):
    if _ENABLED["on"] and isinstance(weight, torch.Tensor):
        try:
            key = _key(x, weight, bias)
            STATS[key] += 1
            if key not in STACKS:
                frames = [f for f in traceback.extract_stack()[:-1]
                          if "/vllm/" in f.filename]
                STACKS[key] = ["%s:%d %s" % (os.path.basename(f.filename), f.lineno, f.name)
                               for f in frames[-4:]]
        except Exception:  # noqa: BLE001
            pass
    return _orig_linear(x, weight, bias, *a, **kw)


def passes_p05(n, m, k, dt, has_bias, w_contig, x_contig):
    return (dt in ("float16", "bfloat16") and not has_bias and n == 1
            and m % 4 == 0 and k % 8 == 0 and k <= 8192 and w_contig)


def main():
    os.makedirs(OUT, exist_ok=True)
    F.linear = _patched_linear
    print("LIN_START patched=%s eager=%s" % (F.linear is _patched_linear, EAGER), flush=True)

    kw = dict(model=MODEL, dtype="float16", tensor_parallel_size=2,
              gpu_memory_utilization=GMU, max_model_len=MAX_LEN,
              max_num_seqs=MAX_NUM_SEQS, disable_custom_all_reduce=True,
              enable_prefix_caching=False)
    if MM:
        kw["limit_mm_per_prompt"] = {"image": 1, "video": 1}
        kw["mm_encoder_attn_backend"] = "TRITON_ATTN"
    else:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if EAGER:
        kw["enforce_eager"] = True
    t0 = time.perf_counter()
    llm = LLM(**kw)
    print("LIN_STEP load OK %.0fs" % (time.perf_counter() - t0), flush=True)

    tok = llm.get_tokenizer()
    ids = tok.encode("Census probe. " + "alpha bravo charlie delta " * 40,
                     add_special_tokens=False)
    prompt = tok.decode(ids)

    # warmup (kernels, allocator) -- not counted
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1,
                                          ignore_eos=True), use_tqdm=False)
    STATS.clear(); STACKS.clear()
    _ENABLED["on"] = True
    llm.generate([prompt + " run"], SamplingParams(temperature=0.0,
                                                   max_tokens=N_GEN,
                                                   ignore_eos=True), use_tqdm=False)
    _ENABLED["on"] = False
    print("LIN_STEP census OK, %d distinct shapes" % len(STATS), flush=True)

    rows = []
    for key, cnt in STATS.most_common():
        n, m, k, dt, hb, wc, xc = key
        rows.append({"n": n, "m": m, "k": k, "dtype": dt, "has_bias": hb,
                     "w_contig": wc, "x_contig": xc, "calls": cnt,
                     "p05": passes_p05(n, m, k, dt, hb, wc, xc),
                     "stack": STACKS.get(key, [])})

    esc = [r for r in rows if not r["p05"]]
    tot = sum(r["calls"] for r in rows)
    esc_calls = sum(r["calls"] for r in esc)
    print("\n=== 落到 rocBLAS（不满足 P0.5）的调用：%d 组 / %d 次（占全部 %d 次的 %.0f%%）==="
          % (len(esc), esc_calls, tot, 100.0 * esc_calls / max(tot, 1)))
    print("%6s %8s %8s %8s %6s %6s %6s %7s  %s"
          % ("n", "m", "k", "dtype", "bias", "wcont", "xcont", "calls", "why"))
    for r in sorted(esc, key=lambda r: -r["calls"])[:25]:
        why = []
        if r["has_bias"]:
            why.append("bias")
        if not r["w_contig"]:
            why.append("w_noncontig")
        if r["n"] != 1:
            why.append("n=%d" % r["n"])
        if r["k"] > 8192:
            why.append("k>8192")
        if r["k"] % 8:
            why.append("k%8")
        if r["m"] % 4:
            why.append("m%4")
        if r["dtype"] not in ("float16", "bfloat16"):
            why.append(r["dtype"])
        print("%6d %8d %8d %8s %6s %6s %6s %7d  %s"
              % (r["n"], r["m"], r["k"], r["dtype"], r["has_bias"],
                 r["w_contig"], r["x_contig"], r["calls"], ",".join(why) or "?"))
    print("\n=== 这些逃逸项的调用栈（前 8 组）===")
    for r in sorted(esc, key=lambda r: -r["calls"])[:8]:
        print("  (n=%d m=%d k=%d x%d) %s" % (r["n"], r["m"], r["k"], r["calls"],
                                             " | ".join(r["stack"])))

    json.dump({"total_calls": tot, "escaping_calls": esc_calls,
               "rows": rows}, open(f"{OUT}/lin_census.json", "w"), indent=1)
    print("\nLIN_DONE_OK -> %s/lin_census.json" % OUT)


if __name__ == "__main__":
    main()
