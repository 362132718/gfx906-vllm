# p05_skinny_probe.py -- does vLLM's ROCm skinny GEMV answer on gfx906, and
# under what conditions must we NOT hand it a call? No patch installed yet: we
# call torch.ops._rocm_C.LLMM1 directly.
#   ssh x@192.168.10.11 'docker exec -i -e BW_EFF=931 vllm-gfx906 python3 /work/p0/p05_skinny_probe.py'
import json, os, time, torch
from vllm import _custom_ops as ops

BW_EFF = float(os.environ["BW_EFF"])          # 931 measured / 821 conservative
REAL = [("in_proj_ba", 48, 5120), ("lm_head", 124160, 5120)]
OUT = os.environ.get("OUT", "/work/p0")


def gtime(fn, inner=10, iters=200):
    """Graph-replay time per call. Eager timing adds ~10 us of Python and would
    completely hide the ~4 us answer for in_proj_ba; capture also proves the
    kernel is graph-safe (it allocates its own output)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters / inner * 1e6


def check(name, m, k, dtype=torch.float16, contiguous=True, rpb=4):
    torch.manual_seed(0)
    w = (torch.randn(k, m, device="cuda", dtype=torch.float16).t() if not contiguous
         else torch.randn(m, k, device="cuda", dtype=dtype) / k ** 0.5)
    x = torch.randn(1, k, device="cuda", dtype=dtype)
    ref = torch.sum(w.double() * x.double(), dim=1)    # elementwise: fp64 GEMV
    ten = torch.nn.functional.linear(x, w).view(-1).double()   # incumbent Tensile
    try:
        out = ops.LLMM1(w, x, rpb)
    except Exception as e:
        return {"case": name, "error": repr(e)[:200]}
    torch.cuda.synchronize()
    o = out.view(-1).double()
    det, a = "stable", ops.LLMM1(w, x, rpb).clone()
    for _ in range(20):
        if not torch.equal(a, ops.LLMM1(w, x, rpb)):
            det = "NONDETERMINISTIC"
            break
    us = gtime(lambda: ops.LLMM1(w, x, rpb))
    return {"case": name, "m": m, "k": k, "rpb": rpb,
            "dtype": str(dtype).split(".")[-1], "contig": contiguous,
            "gemv_us": round(us, 1),
            "tensile_us": round(gtime(lambda: torch.nn.functional.linear(x, w)), 1),
            "gb_s": round(m * k * 2 / (us * 1e-6) / 1e9, 1),
            "pct_bw_eff": round(100 * m * k * 2 / (us * 1e-6) / 1e9 / BW_EFF, 1),
            "rel_vs_fp64": float(((o - ref).abs().max() / ref.abs().max()).item()),
            "rel_tensile_vs_fp64": float(((ten - ref).abs().max() / ref.abs().max()).item()),
            "determinism": det, "out_shape": list(out.shape)}


rows = [check(n, m, k) for n, m, k in REAL]
rows += [check("in_proj_ba_bf16", 48, 5120, torch.bfloat16),
         check("lm_head_bf16", 124160, 5120, torch.bfloat16),
         check("noncontig_weight", 48, 5120, contiguous=False)]
rows += [check("rpb%d_m48" % r, 48, 5120, rpb=r) for r in (1, 2, 4, 8)]
for r in rows:
    print(json.dumps(r, ensure_ascii=False), flush=True)

# --- hard assertions: this table IS the gate, not a report -------------------
by = {r["case"]: r for r in rows}
for name, m, k in REAL:
    r = by[name]
    assert "error" not in r, "%s: LLMM1 does not run on gfx906 -> this plan is void" % name
    assert r["determinism"] == "stable", "%s is nondeterministic" % name
    assert r["rel_vs_fp64"] < 3 * r["rel_tensile_vs_fp64"] + 1e-5, \
        "%s: LLMM1 rel %s not within 3x of Tensile rel %s" % (
            name, r["rel_vs_fp64"], r["rel_tensile_vs_fp64"])
assert by["in_proj_ba"]["gemv_us"] < 0.25 * by["in_proj_ba"]["tensile_us"], \
    "in_proj_ba gain under 4x -- not worth a dispatch change"
assert by["lm_head"]["pct_bw_eff"] > 70, \
    "lm_head only %.1f%% of BW_eff -- the bandwidth path is not what we think" % \
    by["lm_head"]["pct_bw_eff"]
nc = by["noncontig_weight"]
assert "error" not in nc and nc["rel_vs_fp64"] > 1e-2, \
    "expected a transposed weight view to SILENTLY return garbage (got rel %s)." \
    "If LLMM1 now rejects it, the contiguity clause is redundant -- say so in the" \
    "gate comment instead of deleting it blindly." % nc.get("rel_vs_fp64", "error")
bf16_ok = all("error" not in by[c] and by[c]["rel_vs_fp64"] < 1e-2
              for c in ("in_proj_ba_bf16", "lm_head_bf16"))
os.makedirs(OUT, exist_ok=True)
json.dump({"bw_eff_gb_s": BW_EFF, "bf16_ok": bf16_ok, "rows": rows},
          open("%s/p05_skinny.json" % OUT, "w"), indent=1)
print("bf16_ok =", bf16_ok, "| ALL PASS ->", "%s/p05_skinny.json" % OUT)
