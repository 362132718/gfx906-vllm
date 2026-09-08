import os
import time

os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

MODEL = "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16"


def main():
    eager = os.environ.get("ENFORCE_EAGER", "1") == "1"
    util = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
    print("CONFIG: enforce_eager=%s gpu_mem_util=%.2f exllama=%s"
          % (eager, util, os.environ.get("VLLM_WNA16_EXLLAMA", "<unset>")))
    t0 = time.time()
    cgm = os.environ.get("CUDAGRAPH_MODE", "")
    kwargs = {}
    if cgm:
        kwargs["compilation_config"] = CompilationConfig(cudagraph_mode=cgm)
        print("CONFIG: compilation cudagraph_mode=%s" % cgm)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=2,
        gpu_memory_utilization=util,
        enforce_eager=eager,
        max_model_len=256,
        attention_backend="TRITON_ATTN",
        quantization="compressed-tensors",
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        **kwargs,
    )
    print("LOAD_SECONDS: %.1f" % (time.time() - t0))

    sp = SamplingParams(temperature=0, max_tokens=40)
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Once upon a time",
    ]
    outs = llm.generate(prompts, sp)
    for o in outs:
        print("PROMPT:", repr(o.prompt))
        print("OUTPUT:", repr(o.outputs[0].text))
        print("-" * 40)

    first = [o.outputs[0].text for o in outs]
    outs2 = llm.generate(prompts, sp)
    second = [o.outputs[0].text for o in outs2]
    print("DETERMINISTIC:", first == second)
    if first != second:
        for a, b in zip(first, second):
            if a != b:
                print("A:", repr(a))
                print("B:", repr(b))

    t1 = time.time()
    n = 8
    big = llm.generate(["Explain what a GPU is in detail."] * n, SamplingParams(temperature=0, max_tokens=64))
    dt = time.time() - t1
    toks = sum(len(o.outputs[0].token_ids) for o in big)
    print("PERF_BATCH8: %d tokens in %.2fs = %.1f tok/s" % (toks, dt, toks / dt))

    # M=1 sequential decode: the shape the ExLlama bridge exists for.
    for _ in range(2):
        llm.generate(["Ignore."], SamplingParams(temperature=0, max_tokens=4))
    t2 = time.time()
    one = llm.generate(["Explain what a GPU is in detail."],
                       SamplingParams(temperature=0, max_tokens=64))
    dt2 = time.time() - t2
    toks2 = sum(len(o.outputs[0].token_ids) for o in one)
    print("PERF_BATCH1: %d tokens in %.2fs = %.1f tok/s" % (toks2, dt2, toks2 / dt2))


if __name__ == "__main__":
    main()
