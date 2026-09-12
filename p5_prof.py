"""P5-prof: where does a MoE decode step actually spend its time?

Motivation (2026-09-11, user): llama.cpp runs this same 35B-A3B MoE *much* faster
than the 27B dense model on this box, but our vLLM run measured 17.73 tok/s
(eager) against the 27B's 33.4 tok/s (FULL_DECODE_ONLY graph). Two things must be
separated before drawing any conclusion:

  1. **Mode confound.** 17.73 came from `enforce_eager=True` (kernel launches
     straight from Python, ~1.3k launches/step at 27B scale), whereas 33.4 came
     from a captured decode graph. Never compare across those two modes.
  2. **Which kernel owns the step.** Mechanical reasoning says the MoE weights
     are tiny next to the 27B's (3B active at 4-bit ~ 0.5-1.5 GB/token vs 13.5 GB
     for 27B dense), so a step of 56 ms cannot be bandwidth-bound -- 56 ms would
     be ~100x the bandwidth floor. Something else owns the time.

So this run produces, in ONE engine build:
  * single-stream decode step time (graph mode)
  * 8-way concurrent aggregate throughput -> is it latency- or batched-bound?
  * a per-worker chrome trace, to be bucketed by p5_kernels.py

Mirrors p0_step.py's profiler plumbing on purpose (same profiler_config keys,
same start/stop protocol) because that path is already known to work with TP2's
spawned workers. A front-end torch.profiler sees nothing useful here: with TP=2
the kernels run in subprocesses.

Env in:  EAGER, MAX_LEN, OUT, ATTN_BACKEND, DELAY_ITERS, ACTIVE_ITERS, CTX
Env out: <OUT>/p5_wall.json, <OUT>/prof/**, P5PROF_* markers on stdout
"""
import glob
import json
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama-moe")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/p5prof")
PROF_DIR = f"{OUT}/prof"
EAGER = os.environ.get("EAGER", "0") == "1"
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
# GDN constraint (2026-09-11): one Mamba cache block per decode sequence; at 16K
# / gmu 0.90 only 53 exist, and graph capture refuses to start above that.
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "48"))
# PP: 0/1 -> TP2. 2 -> layer split (TP1+PP2). MM: 1 -> keep the vision tower.
GMU = float(os.environ.get("GMU", "0.90"))
PP = int(os.environ.get("PP", "1"))
MM = os.environ.get("MM", "1") == "1"
CONC = int(os.environ.get("CONC", "8"))
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "TRITON_ATTN")
DELAY_ITERS = int(os.environ.get("DELAY_ITERS", "8"))
ACTIVE_ITERS = int(os.environ.get("ACTIVE_ITERS", "64"))
CTX = int(os.environ.get("CTX", "1800"))
MODEL = os.environ.get(
    "MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")
BASE = "Explain in detail why the sky is blue and what determines its shade at dawn. "


def gen(llm, prompts, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0, outs


def main():
    os.makedirs(PROF_DIR, exist_ok=True)
    print("P5PROF_START eager=%s max_len=%d ctx=%d" % (EAGER, MAX_LEN, CTX), flush=True)
    kw = dict(
        model=MODEL, dtype="float16",
        tensor_parallel_size=(1 if PP > 1 else 2),
        gpu_memory_utilization=GMU, max_model_len=MAX_LEN,
        max_num_seqs=MAX_NUM_SEQS, attention_backend=ATTN_BACKEND,
        disable_custom_all_reduce=True,
    )
    if PP > 1:
        kw["pipeline_parallel_size"] = PP
    if MM:
        # must mirror gfx906-serve, or the vision tower profile_run OOMs
        kw["limit_mm_per_prompt"] = {"image": 1, "video": 1}
        kw["mm_encoder_attn_backend"] = "TRITON_ATTN"
    else:
        # MM=0 also drops the encoder cache; without that a PP2 stage OOMs
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    kw["profiler_config"] = {"profiler": "torch",
                             "torch_profiler_dir": PROF_DIR,
                             "torch_profiler_with_stack": False,
                             "torch_profiler_use_gzip": True,
                             "delay_iterations": DELAY_ITERS,
                             "max_iterations": ACTIVE_ITERS}
    if EAGER:
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48]}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    load_s = time.perf_counter() - t0
    print("P5PROF_STEP load OK %.1fs" % load_s, flush=True)

    tok = llm.get_tokenizer()
    ids = tok.encode(BASE, add_special_tokens=False)
    prompt = tok.decode((ids * (CTX // len(ids) + 1))[:CTX])
    ctx_actual = len(tok.encode(prompt, add_special_tokens=False))

    # --- single-stream: prefill leg then a 64-token decode leg -----------------
    t1 = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1,
                                          ignore_eos=True), use_tqdm=False)
    prefill_s = time.perf_counter() - t1
    req_s, outs = gen(llm, [prompt], 64)
    n_out = len(outs[0].outputs[0].token_ids)
    step_s = (req_s - prefill_s) / max(n_out - 1, 1)
    print("P5PROF_STEP single OK ctx=%d prefill=%.3fs decode=%.1f tok/s (%.2f ms/step)"
          % (ctx_actual, prefill_s, 1 / step_s, step_s * 1e3), flush=True)

    # --- 8-way concurrent: is the step latency-bound or batch-bound? -----------
    conc = {}
    for n in (CONC,):
        prompts = [prompt] * n
        t2 = time.perf_counter()
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                             ignore_eos=True), use_tqdm=False)
        pf_n = time.perf_counter() - t2
        req_n, outs_n = gen(llm, prompts, 32)
        tot = sum(len(o.outputs[0].token_ids) for o in outs_n)
        agg = tot / (req_n - pf_n)
        conc[n] = {"prefill_s": pf_n, "req_s": req_n, "tot_tok": tot,
                   "agg_tok_per_s": agg, "per_stream": agg / n}
        print("P5PROF_STEP conc%d OK aggregate=%.1f tok/s (%.2f per stream)"
              % (n, agg, agg / n), flush=True)

    res = {"model": MODEL, "eager": EAGER, "max_len": MAX_LEN, "ctx_actual": ctx_actual,
           "load_s": load_s, "prefill_s": prefill_s, "n_out": n_out,
           "decode_step_ms": step_s * 1e3, "decode_tps": 1 / step_s,
           "concurrent": conc, "delay_iters": DELAY_ITERS,
           "active_iters": ACTIVE_ITERS}
    json.dump(res, open(f"{OUT}/p5_wall.json", "w"), indent=1)

    # --- trace window ----------------------------------------------------------
    for stale in glob.glob(f"{PROF_DIR}/*"):
        os.remove(stale)
    long_prompt = prompt
    prof_tokens = DELAY_ITERS + ACTIVE_ITERS + 40
    t3 = time.perf_counter()
    llm.start_profile("p5")
    gen(llm, [long_prompt], prof_tokens)
    torch.cuda.synchronize()
    llm.stop_profile()
    time.sleep(3)
    files = sorted(glob.glob(f"{PROF_DIR}/**/*.json*", recursive=True))
    print("P5PROF_STEP trace OK %d files, %.1f MB, %.1f s"
          % (len(files), sum(os.path.getsize(f) for f in files) / 1e6,
             time.perf_counter() - t3), flush=True)
    json.dump({"traces": files, "prof_tokens": prof_tokens,
               "delay_iterations": DELAY_ITERS, "active_iterations": ACTIVE_ITERS,
               "decode_step_ms": step_s * 1e3},
              open(f"{OUT}/p5_window.json", "w"), indent=1)
    print("P5PROF_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
