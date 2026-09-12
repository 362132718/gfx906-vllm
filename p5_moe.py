"""P5: MoE smoke on the delivery image.

Target model: cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
  arch  Qwen3_5MoeForConditionalGeneration, 40 layers (30 GDN + 10 full attn),
  256 experts top-8 + 1 shared, moe_intermediate 512, hidden 2048.
  quant AWQ 4-bit, zero_point=True, group_size=32, version=gemm;
  modules_to_not_convert excludes ALL attention/GDN/shared_expert/lm_head/vision
  and MTP -- so ONLY the 256 routed experts are quantized. Everything else is
  fp16 and rides our P0.5 path.

What the image already proved offline (p5_moe_probe.py):
  * Qwen3_5MoeForConditionalGeneration is present; vLLM 0.28.0+rocm724
  * AWQ MoE backend resolves to TRITON on gfx906
    (TritonWNA16Experts._supports_current_device()=True, MarlinExperts=False)
  * moe_align_block_size executes on gfx906
  * awq_gemm / awq_dequantize symbols present
  * group_size 32 is in AWQ_TRITON_SUPPORTED_GROUP_SIZES
WHAT IT HAS NOT PROVED: the Triton WNA16 MoE main kernel actually JITs, is
numerically sane, and is not catastrophically slow. That is this file's job.

TP1 IS NOT POSSIBLE: 24.3 GB of weights on a 16 GB card. Start at TP2.

Env in:
  MAX_LEN   default 8192
  EAGER     1 => --enforce-eager (shortest path to a first failure)
  OUT_TOK   default 32
  PROMPT    single-request prompt
Env out: <OUT>/p5_result.json, plus P5_STEP markers on stdout.
"""
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
from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/p5")
MAX_LEN = int(os.environ.get("MAX_LEN", "8192"))
EAGER = os.environ.get("EAGER", "0") == "1"
MM = os.environ.get("MM", "1") == "1"
OUT_TOK = int(os.environ.get("OUT_TOK", "32"))
MODEL = os.environ.get(
    "MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")
PROMPT = os.environ.get(
    "PROMPT", "Explain in one paragraph why the sky is blue.")
# GDN/Mamba constraint (measured 2026-09-11): every decode sequence parks one
# Mamba cache block. At 16K context / gmu 0.90 only 53 blocks exist, so V1's
# default max_num_seqs=256 makes CUDA-graph capture refuse to start:
#   ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (53).
# Eager mode never runs that check, which is why the first arm passed.
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "53"))
# GMU: gpu_memory_utilization. 0.90 leaves stage 1 of a PP2 run with ~450 MB free
# and the multimodal encoder cache then asks for 564 MB -> OOM (measured
# 2026-09-11). MM=0 removes that cache; GMU adds headroom on top.
GMU = float(os.environ.get("GMU", "0.90"))
# PP: pipeline_parallel_size. 0/1 -> keep TP2. 2 -> layer split, TP1.
PP = int(os.environ.get("PP", "1"))
# SPEC: raw JSON for speculative_config, e.g.
#   SPEC='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
SPEC = os.environ.get("SPEC", "")


def step(name, ok, extra=""):
    print("P5_STEP %s %s %s" % (name, "OK" if ok else "FAIL", extra), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"model": MODEL, "max_len": MAX_LEN, "eager": EAGER, "steps": {}}
    print("env:", {k: os.environ.get(k) for k in (
        "HSA_OVERRIDE_GFX_VERSION", "VLLM_TRITON_ATTN_NUM_SEGMENTS",
        "VLLM_WNA16_EXLLAMA", "VLLM_CACHE_ROOT")}, flush=True)

    # ---- 1. load (TP2 mandatory, or PP2 layer-split) -----------------------
    t0 = time.perf_counter()
    try:
        tp = 1 if PP > 1 else 2
        kw = dict(
            model=MODEL, dtype="float16", tensor_parallel_size=tp,
            gpu_memory_utilization=GMU, max_model_len=MAX_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            disable_custom_all_reduce=True,
        )
        if PP > 1:
            kw["pipeline_parallel_size"] = PP
        if SPEC:
            kw["speculative_config"] = json.loads(SPEC)
        res["tp"] = tp
        res["pp"] = PP
        res["spec"] = SPEC
        res["max_num_seqs"] = MAX_NUM_SEQS
        res["gmu"] = GMU
        # MUST mirror gfx906-serve (docker/entrypoint/serve.sh:106-108):
        #   --limit-mm-per-prompt {"image":1,"video":1}
        #   --mm-encoder-attn-backend TRITON_ATTN
        # Omitting the mm encoder backend sends the vision tower to
        # mm_encoder_attention._forward_sdpa, which on the dummy profile_run
        # sequence asks for a 128 GiB allocation and kills the engine during
        # startup (measured 2026-09-11). It is a probe bug, not a model bug.
        if MM:
            kw["limit_mm_per_prompt"] = {"image": 1, "video": 1}
            kw["mm_encoder_attn_backend"] = "TRITON_ATTN"
        else:
            kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        if EAGER:
            kw["enforce_eager"] = True
        else:
            kw["compilation_config"] = {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8],
            }
        llm = LLM(**kw)
        res["load_s"] = time.perf_counter() - t0
        step("load", True, "%.0fs" % res["load_s"])
    except Exception as e:  # noqa: BLE001
        res["load_error"] = "%s: %s" % (type(e).__name__, e)
        res["load_traceback"] = traceback.format_exc()[-4000:]
        step("load", False, repr(e)[:200])
        json.dump(res, open(f"{OUT}/p5_result.json", "w"), indent=1)
        print("P5_DONE_FAIL load", flush=True)
        return

    # ---- 2. what did the engine actually pick? ----------------------------
    try:
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonWNA16Experts,
        )
        info = {
            "moe_ternary_ok": bool(TritonWNA16Experts._supports_current_device()),
            "is_rocm": bool(current_platform.is_rocm()),
        }
        res["backend_info"] = info
        step("backend_info", True, json.dumps(info))
    except Exception as e:  # noqa: BLE001
        step("backend_info", False, repr(e)[:160])

    # ---- 3. text generation -------------------------------------------------
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOK, ignore_eos=True)
        t1 = time.perf_counter()
        out = llm.generate([PROMPT], sp, use_tqdm=False)
        dt = time.perf_counter() - t1
        o = out[0]
        n = len(o.outputs[0].token_ids)
        txt = o.outputs[0].text
        res["gen"] = {"s": dt, "n_tok": n, "tok_per_s": n / dt,
                      "text": txt[:400], "token_ids": list(o.outputs[0].token_ids)}
        # a MoE that is silently broken usually spews repeats or garbage
        degenerate = len(set(o.outputs[0].token_ids)) <= max(2, n // 8)
        res["gen"]["degenerate"] = degenerate
        step("generate", True, "%.2f tok/s n=%d degenerate=%s" % (n / dt, n, degenerate))
        print("--- text ---\n%s\n------------" % txt[:400], flush=True)
    except Exception as e:  # noqa: BLE001
        res["gen_error"] = "%s: %s" % (type(e).__name__, e)
        res["gen_traceback"] = traceback.format_exc()[-4000:]
        step("generate", False, repr(e)[:200])

    # ---- 4. decode throughput at a modest context --------------------------
    try:
        filler = " ".join(["alpha bravo charlie delta echo"] * 300)   # ~1.9K tok
        sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        t2 = time.perf_counter()
        llm.generate([filler], sp1, use_tqdm=False)
        prefill_s = time.perf_counter() - t2
        sp2 = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
        t3 = time.perf_counter()
        o2 = llm.generate([filler], sp2, use_tqdm=False)[0]
        req_s = time.perf_counter() - t3
        n2 = len(o2.outputs[0].token_ids)
        dec = (req_s - prefill_s) / max(n2 - 1, 1)
        res["decode"] = {"prefill_s": prefill_s, "req_s": req_s, "n_tok": n2,
                         "decode_ms_per_step": dec * 1e3, "decode_tps": 1 / dec}
        step("decode", True, "%.2f tok/s (%.1f ms/step)" % (1 / dec, dec * 1e3))
    except Exception as e:  # noqa: BLE001
        res["decode_error"] = "%s: %s" % (type(e).__name__, e)
        res["decode_traceback"] = traceback.format_exc()[-3000:]
        step("decode", False, repr(e)[:200])

    json.dump(res, open(f"{OUT}/p5_result.json", "w"), indent=1)
    print("P5_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
