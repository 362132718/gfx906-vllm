# p2_l2_dump.py -- G-P4 L2 data: the REAL operands behind a whitelisted M=1 call.
#
# Run in the container, detached:
#   docker exec -d vllm-gfx906 bash -lc "cd /work/p2 && \
#     OUT=/work/p2/l2 python3 p2_l2_dump.py > /work/p2/p2_l2_dump.log 2>&1"
#
# Hook point is the PLAIN FUNCTION _p2_w4a16_gemm_impl, not apply_weights and
# not the custom op:
#   * apply_weights is inside the dynamo-traced region -> torch.save there kills
#     engine init ("torch.* op returned non-Tensor"), and a compiled/graph-
#     replayed step never runs Python at all (pitfalls #5/#7).
#   * the op cannot be re-patched after registration: direct_register_custom_op
#     captured the function OBJECT, so rebinding the module global is a no-op for
#     dispatch. _p2_gemv_or_fused_impl (registered) calls _p2_w4a16_gemm_impl by
#     GLOBAL NAME at runtime, so the patch is seen -- and it is seen on every
#     dispatch, compiled or not, because dynamo does not trace inside an op.
#
# enforce_eager=True is not the point (the hook survives either way) -- it just
# makes the run cheap and makes x a real activation rather than a capture
# placeholder. Consequence to state in the report: these activations come from
# the eager engine, so they are representative of the delivery path, not
# bit-identical to its x.
#
# Why only calls with a nonzero x are dumped: process_weights_after_loading
# warm-loads the kernel with _warm = torch.zeros(1, K) through this same
# function, once per eligible layer. Those calls have the right shapes and no
# information -- and an all-zero x is exactly what a real decode hidden state
# never is.
#
# Clones are mandatory: `out` is a view of the persistent slab (_cbuf[:M]) and
# `part` is reused scratch, so both are overwritten by the next layer's call.
import json
import os
import re
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
os.environ.setdefault("VLLM_TRITON_ATTN_NUM_SEGMENTS", "64")
# Missing this and the run measures TritonW4A16LinearKernel, which has no P2
# route at all -> zero dumps that look like "P2 never fires" (pitfall #3/#28).
os.environ.setdefault("VLLM_WNA16_EXLLAMA", "1")
OUT = os.environ.get("OUT", "/work/p2/l2")
os.environ["VLLM_CACHE_ROOT"] = "/tmp/p2cache_l2"
MODEL = os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
# Per (family, process): how many DISTINCT weight tensors to save. Two modules
# per family per rank is enough to show the gate is not one lucky layer, and a
# dump is ~22 MB (the [K/8,N] int32 weight dominates).
WANT = int(os.environ.get("WANT", "2"))
ALL_FAMS = ((5120, 8192), (5120, 7168))
# FAMS_RUN is a resume knob, not a feature. The first collection run crashed on a
# probe bug (json tuple keys) AFTER writing the four 8192 dumps, so re-collecting
# only the missing family is enough -- and a gate must not be run twice just to
# watch an instrument behave. Dump filenames carry the family, so a restricted run
# cannot overwrite what is already there. Use FAMS_RUN=all for a clean sweep.
_want = os.environ.get("FAMS_RUN", "7168")
FAMS = ALL_FAMS if _want == "all" else tuple(
    f for f in ALL_FAMS if str(f[1]) == _want)


def _install_dump_hook():
    import torch
    from vllm.model_executor.kernels.linear.mixed_precision import (
        wna16_exllama as W)

    orig = W._p2_w4a16_gemm_impl
    pid = os.getpid()
    dev = torch.cuda.current_device() if torch.cuda.is_initialized() else -1
    seen, hits, zero = {}, {}, {}       # keyed by (K, N)

    def counts():
        # str keys: json refuses a tuple key, and that raise inside this hook
        # took the whole engine down on the first run (the hook sits on the op's
        # call path, so any error here is a worker crash, not a lost log line).
        return {"pid": pid, "dev": dev,
                "hits": {"%dx%d" % k: v for k, v in hits.items()},
                "warmlike_zero_x": {"%dx%d" % k: v for k, v in zero.items()},
                "dumped": {"%dx%d" % k: v for k, v in seen.items()}}

    def flush_counts():
        with open("%s/counts.%d.json" % (OUT, pid), "w") as f:
            json.dump(counts(), f)

    def hook(x, w_q, w_s, w_zp, part, out):
        rc = orig(x, w_q, w_s, w_zp, part, out)      # never alter behaviour
        try:
            K, N = int(x.shape[1]), int(out.shape[1])
            if (K, N) not in FAMS or int(x.shape[0]) != 1:
                return rc
            hits[(K, N)] = hits.get((K, N), 0) + 1
            if float(x.abs().amax()) == 0.0:         # the load-time warm-up
                zero[(K, N)] = zero.get((K, N), 0) + 1
                return rc
            if len(seen.get((K, N), [])) >= WANT or w_q.data_ptr() in \
                    seen.get((K, N), []):
                if hits[(K, N)] % 500 == 0:
                    flush_counts()                    # a floor on total hits
                return rc
            seen.setdefault((K, N), []).append(w_q.data_ptr())
            path = "%s/l2_dump_%dx%d_dev%d_pid%d_%d.pt" % (
                OUT, K, N, dev, pid, len(seen[(K, N)]))
            torch.save({
                "x": x.clone(), "w_q": w_q.clone(), "w_s": w_s.clone(),
                "w_zp": w_zp.clone(), "part": part.clone(),
                "out": out.clone(),
                "meta": {"K": K, "N": N, "pid": pid, "dev": dev,
                         "splitk": W._P2_SPLITK, "hit": hits[(K, N)],
                         "x_shape": list(x.shape), "out_shape": list(out.shape),
                         "w_q_shape": list(w_q.shape),
                         "w_s_shape": list(w_s.shape),
                         "w_zp_shape": list(w_zp.shape),
                         "x_ptr": x.data_ptr(), "wq_ptr": w_q.data_ptr()},
            }, path)
            flush_counts()
            print("[dump] %s hit#%d -> %s" % ((K, N), hits[(K, N)], path),
                  flush=True)
        except Exception as e:                        # a probe must not break a run
            # The tensors are already on disk by the time this can fire, so
            # swallowing costs a missing log line at worst. Raising here killed
            # the run mid-collection the first time.
            print("[dump-FAIL] %r" % (e,), flush=True)
        return rc

    W._p2_w4a16_gemm_impl = hook
    return W, counts


# Import-time on purpose: spawn children execute this file as __mp_main__, so
# only module-level code reaches the TP workers where the dispatch happens
# (pitfall #29).
try:
    _W, _counts = _install_dump_hook()
except Exception as _e:
    import sys
    print("IMPORT-FAIL %r" % (_e,), file=sys.stderr, flush=True)
    _W, _counts = None, None


def main():
    os.makedirs(OUT, exist_ok=True)
    print("env:", {k: os.environ.get(k) for k in (
        "VLLM_WNA16_EXLLAMA", "VLLM_WNA16_P2_GEMV", "VLLM_CACHE_ROOT", "OUT")},
        flush=True)
    if _W is None:
        raise SystemExit("hook install failed -- see the IMPORT-FAIL line")
    print("hook active | _P2_OK=%s SPLITK=%d shapes=%s WANT=%d"
          % (_W._P2_OK, _W._P2_SPLITK, sorted(_W._P2_SHAPES), WANT), flush=True)
    if not _W._P2_OK:
        raise SystemExit("FATAL: _P2_OK False -- the .so is not loadable in this"
                         " container, so no whitelisted call will ever reach the"
                         " hook and an empty OUT dir would be misread")

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(
        model=MODEL, dtype="float16", tensor_parallel_size=2,
        gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
        enforce_eager=True,
        attention_backend="TRITON_ATTN", disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_encoder_attn_backend="TRITON_ATTN",
    )
    print("engine up in %.0f s (eager)" % (time.time() - t0), flush=True)

    # A couple of requests, several decode steps each: every step offers 64
    # whitelisted calls per rank, so WANT=2 per family per rank is reached by the
    # second decode step.
    sp = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    for _ in range(2):
        llm.generate(["Explain why the sky is blue and what sets the dawn hue. "
                      * 8], sp, use_tqdm=False)
    print("generate done %.0f s" % (time.time() - t0), flush=True)

    files = sorted(f for f in os.listdir(OUT) if f.startswith("l2_dump_"))
    print("dumps: %d" % len(files), flush=True)
    for f in files:
        p = os.path.join(OUT, f)
        print("   %7.1f MB  %s" % (os.path.getsize(p) / 1e6, f), flush=True)
    for f in sorted(os.listdir(OUT)):
        if f.startswith("counts."):
            # A worker killed mid-write leaves a truncated file (the crashed first
            # run did exactly that); an unreadable counter is a reporting problem,
            # not a reason to die before the completeness verdict below.
            try:
                c = json.load(open(os.path.join(OUT, f)))
            except Exception as e:
                c = "UNREADABLE %r" % (e,)
            print("   %s %s" % (f, c if isinstance(c, str) else json.dumps(c)),
                  flush=True)
    # Filename: l2_dump_{K}x{N}_dev{D}_pid{P}_{idx}.pt. A regex, not split("_")
    # indexing: the first version read parts[1] for the family and raised
    # ValueError *after* the data was safely on disk.
    # Completeness is judged against ALL_FAMS even when FAMS_RUN restricted this
    # run to the missing family -- a resume must still prove the earlier dumps
    # survived, and both families feed one gate. Group by pid, not dev: both
    # workers report dev0 (torch.cuda.current_device() at patch time), so dev
    # cannot separate the TP ranks; pid can, as in p2_ab_gate.py.
    by = {}
    bad = []
    for fn in files:
        m = re.fullmatch(r"l2_dump_(\d+)x(\d+)_dev(-?\d+)_pid(\d+)_(\d+)\.pt", fn)
        if m is None:
            bad.append("unparseable dump filename %r" % fn)
            continue
        K, N, _dev, pid, idx = m.groups()
        by.setdefault((int(K), int(N)), {}).setdefault("pid" + pid, []).append(idx)
    for fam in ALL_FAMS:
        per_pid = by.get(fam, {})
        if len(per_pid) < 2:
            bad.append("%s: %d worker(s) with dumps, need 2 (one per TP rank)"
                       % (fam, len(per_pid)))
            continue
        for pid, idxs in sorted(per_pid.items()):
            if len(idxs) < WANT:
                bad.append("%s pid%s: %d dumps, need %d" % (fam, pid, len(idxs), WANT))
    for fam in sorted(by):
        if fam not in ALL_FAMS:
            bad.append("%s: not a whitelisted family" % (fam,))
    if bad:
        raise SystemExit("L2 DUMP INCOMPLETE: %s (a family with too few dumps"
                         " cannot be gated -- check hook/P2_OK/hit counts)" % bad)
    for fam in ALL_FAMS:
        print("   complete %s: %s" % (fam, {k: sorted(v) for k, v in
                                            sorted(by[fam].items())}), flush=True)
    print("L2-DUMP-DONE", flush=True)


if __name__ == "__main__":
    main()
