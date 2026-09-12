# p2_dynamo_probe.py -- does the M-based branch survive the compile mode the
# engine uses? 30 s on one layer instead of 15 min in a 27B engine.
#
# G-P3 failed because the shape test lived in apply_weights, i.e. INSIDE the
# dynamo-traced region: the region is traced once at the prefill M and the
# False branch gets baked, so every captured decode graph replays the incumbent.
# The fix moves the test into the custom op impl, which dynamo does not trace.
#
# Three arms, in this order:
#   [1] MECHANISM  a 3-line traced function that returns `x.shape[0] == 1`,
#                  traced at M=130 and called at M=1. If this does NOT say True,
#                  the folding is real and the probe has teeth.
#   [2] OLD ROUTE  the pre-fix routing shape test in a traced wrapper: expected
#                  P2 fired = False.
#   [3] NEW ROUTE  apply_weights (routes on a load-time constant; the op decides
#                  on M): expected P2 fired = True at M=1, False at M=8, and the
#                  compiled M=1 result bit-equal to the eager one (P2 has no
#                  atomicAdd, so equality is the legal test here).
import os
import sys

os.environ["VLLM_WNA16_EXL_SHAPEDBG"] = ""       # keep the census out of it
sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch  # noqa: E402
import p2_route_gate as G  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision import (  # noqa: E402
    wna16_exllama as W)

K, N = 5120, 7168
PRE, M1, M8 = 130, 1, 8
rc = 0


def fired(counter, fn, *a):
    """Run fn and report whether its own dispatch counter grew."""
    before = len(counter)
    out = fn(*a)
    return (len(counter) > before), out


def main():
    global rc
    print("_P2_OK=%s route-op=%s" % (W._P2_OK,
                                     hasattr(torch.ops.vllm, "p2_gemv_or_fused")))

    # [1] MECHANISM ----------------------------------------------------------
    @torch.compile(fullgraph=True, dynamic=True)
    def is_one(x):
        return x.shape[0] == 1

    is_one(torch.zeros(PRE, K, dtype=torch.float16, device="cuda"))
    try:
        verdict = is_one(torch.zeros(M1, K, dtype=torch.float16, device="cuda"))
    except Exception as e:
        verdict = "raised %s" % type(e).__name__
    print("[1] traced `x.shape[0] == 1` traced@%d called@%d -> %s"
          % (PRE, M1, verdict))
    if verdict is True:
        print("    NOTE: dynamo did NOT fold it here -- arm [3] is then the "
              "only thing that can explain G-P3, read it carefully")

    # [2]/[3] ROUTE ----------------------------------------------------------
    kern, layer = G.mk(K, N)
    w_q, w_s, w_zp, w_g_idx = kern._get_weight_params(layer)
    cbuf = kern._cbuf

    old_hits = []

    def old_route(x):
        # The pre-fix code, restored verbatim in spirit: M tested in the traced
        # region, op called from there. Counted locally -- _P2_HITS is appended
        # by the new op impl only, so the old route could never show a hit and
        # the arm would be unfalsifiable.
        M = x.shape[0]
        out = cbuf[:M] if M <= kern._mcap else torch.empty(
            (M, N), dtype=torch.float16, device=x.device)
        tdq = kern._tdq_empty if M <= 50 else torch.empty(
            (K, N), dtype=torch.float16, device=x.device)
        if (M == 1 and kern._p2_part is not None
                and os.environ.get("VLLM_WNA16_P2_GEMV", "1") == "1"
                and W._p2_layer_ok(kern.config, K, N)):
            old_hits.append(int(M))
            return torch.ops.vllm.p2_w4a16_gemm(
                x, w_q, w_s, w_zp, kern._p2_part, out)
        out.zero_()
        W.ops.gptq_gemm_out(out, x, w_q, w_zp, w_s, w_g_idx, True, True, 4,
                            tdq)
        return out

    def run_matrix(tag):
        bad = 0
        for name, fn, counter in (
                ("[2] OLD route", old_route, old_hits),
                ("[3] NEW route", lambda x: kern.apply_weights(layer, x),
                 W._P2_HITS)):
            eager_hit, eager_out = fired(counter, fn, G.actx(K, M1))
            cf = torch.compile(fn, fullgraph=True, dynamic=True)
            cf(G.actx(K, PRE))                  # trace at the prefill M, as 130
            got1 = fired(counter, cf, G.actx(K, M1))
            got8 = fired(counter, cf, G.actx(K, M8))
            print("%s %s: eager@M=1 fired=%s | compiled traced@%d: M=1 fired=%s"
                  " M=8 fired=%s"
                  % (tag, name, eager_hit, PRE, got1[0], got8[0]))
            if name.startswith("[3]"):
                eq = bool(torch.equal(got1[1], eager_out))
                nd, mx = G.adiff(got1[1], eager_out)
                print("      compiled@M=1 == eager@M=1 bit-for-bit: %s"
                      " (n_diff=%d max=%.1e)" % (eq, nd, mx))
                if not (got1[0] and not got8[0] and eq):
                    print("      FAIL: the fix does not hold under compile")
                    bad = 1
            else:
                print("      -> traced-region shape test %s survive it"
                      % ("DOES" if got1[0] else "DOES NOT"))
                if not got1[0]:
                    print("      the pre-fix route is dead here: this IS G-P3,"
                          " reproduced on one layer in seconds")
        return bad

    # The regime decides. With int specialization ON dynamo re-traces a
    # specialized M=1 artifact, so the old shape test still fires; vLLM compiles
    # ONE artifact per range and forbids that specialization, and then the
    # traced test is resolved against the trace-time shape. specialize_int is
    # the knob, verified against the first run of this probe.
    import torch._dynamo.config as dc
    for spec in (True, False):
        dc.specialize_int = spec
        print("--- regime: specialize_int=%s ---" % spec, flush=True)
        rc = run_matrix("  ")
    return rc


if __name__ == "__main__":
    sys.exit(main())
