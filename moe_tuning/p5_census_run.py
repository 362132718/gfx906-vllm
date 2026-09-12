"""p5_census_run.py -- minimal engine run whose only job is to exercise decode
so that lin_census_hook.py (loaded via .pth in every worker process) can count
torch.nn.functional.linear calls.

Deliberately does no patching itself: the hook must live in the worker
processes, and the parent cannot reach them.

Env in: EAGER, MAX_LEN, MAX_NUM_SEQS, GMU, MM, N_GEN, CTX, OUT
"""
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-exllama-moe")

from vllm import LLM, SamplingParams  # noqa: E402

OUT = os.environ.get("OUT", "/work/census")
EAGER = os.environ.get("EAGER", "1") == "1"
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "48"))
GMU = float(os.environ.get("GMU", "0.92"))
MM = os.environ.get("MM", "0") == "1"
N_GEN = int(os.environ.get("N_GEN", "6"))
CTX = int(os.environ.get("CTX", "600"))
MODEL = os.environ.get("MODEL", "/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit")


def main():
    os.makedirs(OUT, exist_ok=True)
    # hook output dir must be visible to the workers too -> use /work
    os.environ["LIN_CENSUS_OUT"] = "/work/census/lin"
    print("CENSUS_START eager=%s hook_out=%s" % (EAGER, os.environ["LIN_CENSUS_OUT"]),
          flush=True)

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
    print("CENSUS_STEP load OK %.0fs" % (time.perf_counter() - t0), flush=True)

    tok = llm.get_tokenizer()
    ids = tok.encode("Census. " + "alpha bravo charlie delta echo " * 40,
                     add_special_tokens=False)[:CTX]
    prompt = tok.decode(ids)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=N_GEN,
                                          ignore_eos=True), use_tqdm=False)
    print("CENSUS_STEP generate OK", flush=True)
    time.sleep(3)   # let the workers dump
    print("CENSUS_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
