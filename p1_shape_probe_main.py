# p1_shape_probe_main.py -- the LLM payload for p1_shape_probe.sh, as a REAL FILE
# because vLLM's spawn-mode workers re-import the main module; a `python3 -` stdin
# script has main module "<stdin>" and every worker dies with
# FileNotFoundError: '/<stdin>' (reproduced). The __main__ guard is mandatory --
# without it, spawn re-imports would recurse into LLM() forever.
import sys
import time

from vllm import LLM, SamplingParams


def main():
    model = sys.argv[1]
    llm = LLM(model=model, tensor_parallel_size=2, dtype="float16",
              max_model_len=4096, gpu_memory_utilization=0.85,
              disable_custom_all_reduce=True, enforce_eager=True,   # eager: python runs every step
              # Text-only on purpose: with {"image":1,"video":1} and no
              # --mm-encoder-attn-backend, profile_run's dummy ViT forward
              # materializes a 128 GiB attention tensor and OOMs (reproduced on
              # this exact probe). The delivery path avoids it via TRITON_ATTN,
              # but the shape census does not need mm at all: the five WNA16
              # GEMV shapes live in the LLM stack, the vision tower is not
              # quantized.
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)  # full window
    llm.generate(["Explain in one paragraph why HBM bandwidth limits decode."], sp)
    print("GENERATED 64 decode steps", flush=True)
    import subprocess
    rc = subprocess.run(["pkill", "-USR1", "-f", "VLLM::[W]or"])
    print("pkill -USR1 workers rc=%s (1 = none matched, trying EngineCore)" % rc.returncode)
    if rc.returncode != 0:
        subprocess.run(["pkill", "-USR1", "-f", "[V]LLM::EngineCore"])
    time.sleep(5)                       # let the signal handlers write their CSVs


if __name__ == "__main__":
    main()
