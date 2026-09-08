# p0_step.py -- one decode step, five buckets. Uses the same engine flags as
# gfx906-serve so attribution transfers to the delivery path.
#
# NOTE the __main__ guard is mandatory: vLLM spawns its TP workers with
# multiprocessing "spawn", so an unguarded top-level LLM() makes every child
# re-import this module and re-enter the engine ("current process has finished
# its bootstrapping phase").
#
# NOTE the env backstop is mandatory too: calling LLM() directly bypasses
# gfx906-serve, and this dev container does not carry the image's build-time
# ENV (see memory gfx906-env-pitfalls #3). Without VLLM_WNA16_EXLLAMA the W4A16
# path silently falls back to TritonW4A16LinearKernel and decode measures
# 3.9 tok/s instead of 22.0 -- an invalid attribution that looks like a real
# measurement unless you check.
#
# NOTE attribution uses vLLM's per-worker profiler, not a torch.profiler here:
# with TP=2 the kernels execute in spawned subprocesses, so a front-end
# profile sees only hipDeviceSynchronize and reports "kernels with device
# time: 0" -- an empty table that again looks like a real measurement.
# p0_parse_trace.py aggregates the worker traces.
import glob, json, os, time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
# serve.sh's fourth backstop: the compile-cache root is keyed by W4A16 kernel
# mode because vLLM's cache hash omits the kernel choice. Missing this line
# reproduces pitfall #4 verbatim (AOT artifact built under the Triton kernel
# reused under ExLlama -> "expected size 640==5120, stride 8192==1024").
_W4A16 = "exllama" if os.environ.get("VLLM_WNA16_EXLLAMA") == "1" else "triton"
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    f"{os.environ.get('HOME', '/root')}/.cache/vllm-w4a16-{_W4A16}")
# One env name for the whole chain (p0_parse_trace.py already reads OUT, and
# p0_buckets.py is being switched to it): an A/B run needs two archives that do
# not overwrite each other's p0_wall.json / prof/.
OUT = os.environ.get("OUT", "/work/p0")
PROF_DIR = f"{OUT}/prof"
# The delivery anchor is 47.0 ms/step with the cards boosting to 1700MHz. The
# host thermtrips during init at level 8 (two deaths in five minutes, junction
# 102-106C), so a long run may be pinned lower -- then the expected step scales
# with the clock and the perf gate has to know it, or a legitimately slower card
# reads as a broken kernel selection.
CLOCK_MHZ = float(os.environ.get("CLOCK_MHZ", "1700"))
# P0.5 legitimately takes the step from 53.7 ms to ~33 ms, which is BELOW this
# gate's 0.6x floor -- the floor exists only to catch the ~5x Triton W4A16
# fallback, so an A/B run passes its own post-change anchor in here instead of
# weakening the band for everyone.
EXPECTED_MS = float(os.environ.get("EXPECTED_MS", 47.0 * 1700.0 / CLOCK_MHZ))
# Capture sizes cost GPU heat in the init phase but do not change what the M=1
# graph contains, so an attribution run may shrink them.
CAPTURE_SIZES = json.loads(os.environ.get("CAPTURE_SIZES",
                                          "[1, 2, 4, 8, 16, 24, 32, 48]"))
# Worker-side recording window, in engine steps (1 step = 1 forward pass).
# delay skips the prefill step plus whatever ran while the start RPC landed;
# max closes the trace after ~ACTIVE clean decode steps, mid-request, so we do
# not drag a 256-step trace through the exporter.
DELAY_ITERS = 8
ACTIVE_ITERS = 64
PROFILED_REQ_TOKENS = 300          # >> DELAY + ACTIVE
os.environ["VLLM_WNA16_EXL_SHAPEDBG"] = f"{OUT}/p0_step_shapes.csv"

import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
PROMPT = "Explain in detail why the sky is blue and what determines its shade at dawn. " * 8
N = 256          # decode-dominant: prefill is ~2% of a 256-step request


def perf_gate(per_step_ms):
    """Abort if the delivery path is not actually live.

    EXPECTED_MS is the 1700MHz delivery step (47.0 ms) scaled to whatever level
    p0_clock.sh pinned. The band is deliberately wide: its only job is to catch
    the TritonW4A16LinearKernel fallback, whose signature is a ~5x step
    (258 ms at 1700MHz). A wrong CLOCK_MHZ shifts the whole band, so both
    messages name it -- check the pin before believing a failure here.
    """
    tps = 1000.0 / per_step_ms
    if not (0.6 * EXPECTED_MS <= per_step_ms <= 1.9 * EXPECTED_MS):
        raise SystemExit(
            "FATAL: decode step %.1f ms = %.1f tok/s, outside %.0f..%.0f ms for"
            " a %dMHz pin. ~5x the expectation means the Triton W4A16 fallback"
            " is live (VLLM_WNA16_EXLLAMA not in effect); also check TP=2 took"
            " effect and that CLOCK_MHZ matches `p0_clock.sh status`."
            % (per_step_ms, tps, 0.6 * EXPECTED_MS, 1.9 * EXPECTED_MS, CLOCK_MHZ))
    print("perf gate OK: %.1f ms/step = %.1f tok/s (expected %.1f ms at %dMHz)"
          % (per_step_ms, tps, EXPECTED_MS, CLOCK_MHZ), flush=True)


def gen(llm, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    return time.perf_counter() - t0, out[0]


def main():
    print("env:", {k: os.environ.get(k) for k in (
        "HSA_OVERRIDE_GFX_VERSION", "VLLM_TRITON_ATTN_NUM_SEGMENTS",
        "VLLM_WNA16_EXLLAMA", "VLLM_CACHE_ROOT")}, flush=True)
    print("conditions: clock_mhz=%d capture_sizes=%s window=delay%d+active%d"
          % (CLOCK_MHZ, CAPTURE_SIZES, DELAY_ITERS, ACTIVE_ITERS), flush=True)
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
        attention_backend="TRITON_ATTN", disable_custom_all_reduce=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": CAPTURE_SIZES},
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN",
        profiler_config={"profiler": "torch",
                         "torch_profiler_dir": PROF_DIR,
                         "torch_profiler_with_stack": False,
                         "torch_profiler_use_gzip": True,
                         "delay_iterations": DELAY_ITERS,
                         "max_iterations": ACTIVE_ITERS},
    )
    prefill = min(gen(llm, 1)[0] for _ in range(3))
    req_times = [gen(llm, N) for _ in range(3)]
    req, out = min(req_times, key=lambda t: t[0])
    per_step = (req - prefill) / (N - 1)
    print("prefill wall = %.1f ms ; request(%d tok) wall = %.1f ms"
          % (prefill * 1e3, N, req * 1e3), flush=True)
    print("decode wall/step = %.2f ms  (=> %.1f tok/s)"
          % (per_step * 1e3, 1 / per_step), flush=True)
    perf_gate(per_step * 1e3)
    json.dump({"prefill_ms": prefill * 1e3, "req_ms": req * 1e3,
               "per_step_ms": per_step * 1e3, "n_tok": N,
               "clock_mhz": CLOCK_MHZ, "capture_sizes": CAPTURE_SIZES},
              open(f"{OUT}/p0_wall.json", "w"))
    # Token ids, not just prose: swapping in a different GEMV kernel changes
    # fp16 rounding, so "the answer is unchanged" needs a greedy-decode
    # prefix-equality check rather than a vibe read of the text. (out is the
    # RequestOutput from gen(); the completion carries the ids.)
    json.dump({"token_ids": list(out.outputs[0].token_ids),
               "text": out.outputs[0].text},
              open(f"{OUT}/p0_step_out.json", "w"), ensure_ascii=False)

    os.makedirs(PROF_DIR, exist_ok=True)
    for stale in glob.glob(f"{PROF_DIR}/*"):
        os.remove(stale)              # a leftover trace would be parsed as fresh
    t0 = time.perf_counter()
    llm.start_profile("p0")
    gen(llm, PROFILED_REQ_TOKENS)     # window closes mid-request via max_iterations
    torch.cuda.synchronize()
    llm.stop_profile()
    time.sleep(3)                     # workers export after the RPC returns
    files = sorted(glob.glob(f"{PROF_DIR}/**/*.json*", recursive=True))
    print("worker traces: %d files, %.1f MB, %.1f s wall"
          % (len(files), sum(os.path.getsize(f) for f in files) / 1e6,
             time.perf_counter() - t0), flush=True)
    for f in files:
        print("   %8.1f MB  %s" % (os.path.getsize(f) / 1e6, f), flush=True)
    json.dump({"delay_iterations": DELAY_ITERS, "active_iterations": ACTIVE_ITERS,
               "profiled_tokens": PROFILED_REQ_TOKENS,
               "per_step_ms": per_step * 1e3, "traces": files},
              open(f"{OUT}/p0_window.json", "w"))
    if not files:
        raise SystemExit("FATAL: no worker traces -- profiler_config never "
                         "reached the workers, or start/stop was a no-op")


if __name__ == "__main__":
    main()
