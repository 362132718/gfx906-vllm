# b3_ablate.py -- isolate which layer breaks graph output.
#   B3_MODE=eager      : enforce_eager=True                    (no compile, no graph)
#   B3_MODE=graphonly  : compilation mode=NONE + FULL_DECODE_ONLY graph, no inductor
#   B3_MODE=full       : default compile + FULL_DECODE_ONLY   (known bad: all "!")
import os
import sys

os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode

MODEL = "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16"
EXPECTED_PREFIX = {
    "Hello, my name is": " Ilya. I am 19 years old.",
    "The capital of France is": " Paris.",
}


def build(mode):
    common = dict(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.85, max_model_len=512, max_num_seqs=48,
        attention_backend="TRITON_ATTN", quantization="compressed-tensors",
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if mode == "eager":
        return LLM(enforce_eager=True, **common)
    if mode == "graphonly":
        cc = CompilationConfig(mode=CompilationMode.NONE,
                               cudagraph_mode="FULL_DECODE_ONLY",
                               cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 48])
        return LLM(enforce_eager=False, compilation_config=cc, **common)
    cc = CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY",
                           cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 48])
    return LLM(enforce_eager=False, compilation_config=cc, **common)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
    print("B3_MODE:", mode, flush=True)
    llm = build(mode)
    sp = SamplingParams(temperature=0, max_tokens=24)
    outs = llm.generate(list(EXPECTED_PREFIX), sp)
    ok = True
    for o in outs:
        got = o.outputs[0].text
        pre = EXPECTED_PREFIX[o.prompt]
        m = got.startswith(pre)
        ok = ok and m
        print("OUTPUT:", repr(got), "PREFIX_OK:", m, flush=True)
    print("MODE_RESULT:", "GOOD" if ok else "BAD", flush=True)


if __name__ == "__main__":
    main()
