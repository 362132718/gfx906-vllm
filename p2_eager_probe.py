# p2_eager_probe.py -- separate the remaining causes of G-P3's zero P2 hits.
#
# The on arm recorded 0 p2_gemv launches while the incumbent kept running
# 256/step/rank, and both arms' 256 greedy tokens were bit-identical:
#   (A) the loader never allocated _p2_part (a term of _p2_layer_ok is False in
#       the engine even though every term reads True offline), or
#   (B) the route's `M == 1` is folded at TRACE time -- apply_weights sits inside
#       the compiled region, so an artifact traced at the prefill M bakes False.
# With enforce_eager=True nothing is compiled, so the Python branch really runs
# on every M=1 step: launches > 0 => (B), launches == 0 => (A).
#
# Part 2 answers (A) without touching production code: vLLM spawns its TP
# workers with multiprocessing, and spawn RE-IMPORTS this __main__ module in
# every child, so a module-level patch of the kernel class lands in the workers.
# Each worker appends one JSONL row per layer -- the load-time terms and the
# post-load _p2_part state -- so the failing predicate is named, not inferred.
# (Incremental append, not atexit: vLLM hard-kills TP workers.)
#
# Run in the container, detached:
#   docker exec -d vllm-gfx906 bash -lc "cd /work/p2 && \
#     OUT=/work/p2/eager python3 p2_eager_probe.py > /work/p2/p2_eager_probe.log 2>&1"
import csv
import glob
import json
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
# Without this the engine answers "Using TritonW4A16LinearKernel" and the probe
# measures a delivery path that does not exist in the image (first eager run).
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
OUT = os.environ.get("OUT", "/work/p2/eager")
os.environ["VLLM_CACHE_ROOT"] = "/tmp/p2cache_eager"
os.environ["VLLM_WNA16_EXL_SHAPEDBG"] = f"{OUT}/shapes.csv"
MODEL = os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
os.makedirs(OUT, exist_ok=True)

DIAG = f"{OUT}/p2_diag.jsonl"


def _install_worker_diag():
    """Patch the kernel so every process records its own eligibility terms."""
    import torch
    from vllm.model_executor.kernels.linear.mixed_precision import (
        wna16_exllama as W)

    path = "%s.%d" % (DIAG, os.getpid())
    f = open(path, "a")

    def dump(kind, kern, extra):
        c = kern.config
        K, N = c.partition_weight_shape
        rec = {
            "kind": kind, "pid": os.getpid(),
            "K": int(K), "N": int(N),
            "P2_OK": W._P2_OK,
            "in_whitelist": (int(K), int(N)) in W._P2_SHAPES,
            "size_bits": int(c.weight_type.size_bits),
            "has_g_idx": bool(c.has_g_idx),
            "group_size": c.group_size,
            "act_type": str(c.act_type),
            "act_is_f16": c.act_type is torch.float16,
            "zero_points": bool(c.zero_points),
            "layer_ok": bool(W._p2_layer_ok(c, K, N)),
            "_t_P2_OK": bool(W._P2_OK),
            "_t_in_whitelist": (int(K), int(N)) in W._P2_SHAPES,
            "_t_size_bits4": int(c.weight_type.size_bits) == 4,
            "_t_no_g_idx": not c.has_g_idx,
            "_t_group128": c.group_size == W._P2_GROUP,
            "_t_act_is_f16": c.act_type is torch.float16,
            "part": ("missing" if not hasattr(kern, "_p2_part")
                     else ("none" if kern._p2_part is None else "set")),
            "SPLITK": W._P2_SPLITK,
        }
        rec.update(extra)
        f.write(json.dumps(rec) + "\n")
        f.flush()

    orig_load = W.WNA16ExllamaKernel.process_weights_after_loading

    def load(self, layer):
        orig_load(self, layer)
        dump("load", self, {})

    orig_apply = W.WNA16ExllamaKernel.apply_weights

    def apply(self, layer, x, bias=None):
        n = x.shape[:-1]
        M = int(x.reshape(-1, x.shape[-1]).shape[0])
        out = orig_apply(self, layer, x, bias)
        if M == 1:
            dump("apply1", self, {"hits_len": len(W._P2_HITS),
                                         "out_shape": list(n)})
        return out

    W.WNA16ExllamaKernel.process_weights_after_loading = load
    W.WNA16ExllamaKernel.apply_weights = apply
    return W


# Import-time on purpose: spawn children execute this file as __mp_main__, so
# only module-level code can reach the TP workers where the dispatch happens.
try:
    _W = _install_worker_diag()
except Exception as _e:  # pragma: no cover
    import sys
    print("IMPORT-FAIL %r" % (_e,), file=sys.stderr, flush=True)
    _W = None


def main():
    print("env:", {k: os.environ.get(k) for k in (
        "VLLM_WNA16_EXLLAMA", "VLLM_WNA16_P2_GEMV", "VLLM_CACHE_ROOT",
        "VLLM_WNA16_EXL_SHAPEDBG", "OUT")}, flush=True)
    if _W is None:
        raise SystemExit("patch install failed -- see the IMPORT-FAIL line")
    print("patch active | _P2_OK=%s SPLITK=%d shapes=%s"
          % (_W._P2_OK, _W._P2_SPLITK, sorted(_W._P2_SHAPES)), flush=True)

    from vllm import LLM, SamplingParams
    import torch

    t0 = time.time()
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
        enforce_eager=True,                      # <- the whole point
        attention_backend="TRITON_ATTN", disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN",
        profiler_config={"profiler": "torch", "torch_profiler_dir": f"{OUT}/prof",
                         "torch_profiler_with_stack": False,
                         "torch_profiler_use_gzip": True,
                         "delay_iterations": 4, "max_iterations": 8},
    )
    print("engine up in %.0f s (eager)" % (time.time() - t0), flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True)
    llm.generate(["Explain why the sky is blue." * 8], sp, use_tqdm=False)
    llm.start_profile("eager")
    llm.generate(["Explain why the sky is blue." * 8], sp, use_tqdm=False)
    torch.cuda.synchronize()
    llm.stop_profile()
    time.sleep(3)

    files = sorted(glob.glob(f"{OUT}/prof/**/*.json*", recursive=True))
    print("traces: %d" % len(files), flush=True)
    import gzip
    for f in files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            ev = json.load(fh).get("traceEvents", [])
        names = [str(e.get("name", "")) for e in ev
                 if e.get("ph") == "X" and str(e.get("cat", "")) == "kernel"]
        p2 = sum(1 for n in names if "p2_gemv" in n)
        red = sum(1 for n in names if "p2_reduce" in n)
        inc = sum(1 for n in names if "gemm_half_q_half_gptq" in n)
        print("%s p2_gemv=%d p2_reduce=%d incumbent=%d"
              % (os.path.basename(f)[-28:], p2, red, inc), flush=True)
        if p2:
            print("   p2 names: %s"
                  % sorted({n[:60] for n in names if "p2_" in n}), flush=True)

    for f in sorted(glob.glob(f"{DIAG}.*")):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        wl = [r for r in rows if r["in_whitelist"]]
        print("%s: %d rows (%d whitelist) | whitelist (layer_ok,part)=%s"
              % (os.path.basename(f), len(rows), len(wl),
                 sorted({(r["layer_ok"], r["part"]) for r in wl})), flush=True)
        sig = {}
        for r in rows:
            if r["layer_ok"]:
                continue
            bad = tuple(k for k in ("P2_OK", "in_whitelist", "size_bits4",
                                    "no_g_idx", "group128", "act_is_f16")
                        if not r["_t_" + k])
            sig[bad] = sig.get(bad, 0) + 1
        for bad, cnt in sorted(sig.items(), key=lambda kv: -kv[1]):
            print("   ineligible x%-4d fails=%s" % (cnt, list(bad)), flush=True)
        a1 = [r for r in rows if r["kind"] == "apply1"]
        print("   M=1 apply calls: %d (whitelist %d, of which layer_ok %d)"
              % (len(a1), len([r for r in a1 if r["in_whitelist"]]),
                 len([r for r in a1 if r["layer_ok"]])), flush=True)
        if wl:
            print("   whitelist sample: %s" % wl[0], flush=True)

    for f in sorted(glob.glob(f"{OUT}/shapes.csv.*")):
        ms = {}
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (int(r["m"]), int(r["k"]), int(r["n"]))
                ms[key] = ms.get(key, 0) + 1
        one = {k: v for k, v in ms.items() if k[0] == 1}
        print("%s: %d rows, M=1 distinct=%d, M=1 whitelist=%s"
              % (os.path.basename(f), sum(ms.values()), len(one),
                 {k[1:]: v for k, v in one.items()
                  if k[1:] in [(5120, 8192), (5120, 7168)]}), flush=True)
    print("PROBE DONE", flush=True)
    n_diag = sum(1 for f in glob.glob(f"{DIAG}.*") for l in open(f) if l.strip())
    print("VERDICT: %d kernel-side rows recorded. 0 rows means WNA16ExllamaKernel"
          " was never selected (Triton fallback) -- the run says nothing about"
          " (A) or (B); fix the env and re-run." % n_diag, flush=True)


if __name__ == "__main__":
    main()
