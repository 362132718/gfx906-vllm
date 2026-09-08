import os
import time

os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

MODEL = "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16"

# Reference outputs produced by the eager ExLlama path (verified
# deterministic, char-identical with Triton oracle, /tmp/exl_run.log):
EXPECTED = {
    "Hello, my name is":
        " Ilya. I am 19 years old. I am a student. I study at a university. "
        "I study English, history, and mathematics. I like to read books. "
        "I like to",
    "The capital of France is":
        " Paris.\nParis is a city in France.\nParis is the capital of "
        "France.\nParis is a city in France.\nParis is the capital of "
        "France.\nParis is a city in",
}


def main():
    print("CONFIG: graph=FULL_DECODE_ONLY eager=False")
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_model_len=512,
        max_num_seqs=48,
        attention_backend="TRITON_ATTN",
        quantization="compressed-tensors",
        disable_custom_all_reduce=os.environ.get("B4_CUSTOM_AR", "0") != "1",
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 48],
        ),
    )
    print("LOAD_SECONDS: %.1f" % (time.time() - t0))

    sp = SamplingParams(temperature=0, max_tokens=40)
    prompts = list(EXPECTED)
    outs = llm.generate(prompts, sp)
    ok = True
    for o in outs:
        got = o.outputs[0].text
        exp = EXPECTED[o.prompt]
        match = got == exp
        ok &= match
        print("PROMPT:", repr(o.prompt))
        print("OUTPUT:", repr(got))
        print("MATCHES_EAGER_BASELINE:", match)
    print("ALL_MATCH:", ok)

    outs2 = llm.generate(prompts, sp)
    det = all(a.outputs[0].text == b.outputs[0].text for a, b in zip(outs, outs2))
    print("DETERMINISTIC:", det)

    one = prompts[:1]
    llm.generate(one, SamplingParams(temperature=0, max_tokens=8))
    t0 = time.time()
    o = llm.generate(one, SamplingParams(temperature=0, max_tokens=64))
    n = len(o[0].outputs[0].token_ids)
    print("PERF_BATCH1: %d tokens in %.2fs = %.1f tok/s"
          % (n, time.time() - t0, n / (time.time() - t0)))

    b8 = (prompts * 3 + ["Once upon a time"])[:8]
    t0 = time.time()
    o8 = llm.generate(b8, SamplingParams(temperature=0, max_tokens=32))
    n8 = sum(len(x.outputs[0].token_ids) for x in o8)
    dt8 = time.time() - t0
    print("PERF_BATCH8: %d tokens in %.2fs = %.1f tok/s" % (n8, dt8, n8 / dt8))
    assert ok, "graph output diverged from eager baseline"
    assert det, "non-deterministic under graphs"


if __name__ == "__main__":
    main()
