# p2_repro_discriminator.py -- why did route_gate rows C/D disagree?
#
# p2_incumbent_determinism.py established: gptq_gemm_out is bit-reproducible when
# the SAME output buffer is reused ("slab vs slab eq=True") and not when a fresh
# buffer is allocated per call. p2_route_gate rows C/D compare two calls that
# both write kern._cbuf[:M] -- same buffer -- yet they disagreed on 71% of
# positions. One of the two observations is wrong, and which one decides whether
# the P2 patch is safe, so this isolates the four candidate causes instead of
# theorising about them:
#   [1] apply_weights, repeated          -> is the production entry point reproducible?
#   [2] apply_weights, env 1 vs 0        -> is the kill-switch read implicated?
#   [3] direct op into the same slab     -> cross-check the probe's eq=True
#   [4] direct op into fresh buffers     -> cross-check the probe's eq=False
# If [1] is reproducible and [2] is not, the env switch itself changes the
# result and the patch has a real defect. If [1] is ALSO non-reproducible, the
# probe's "slab vs slab eq=True" was the fluke and reproducibility is not a
# function of buffer identity -- then the gate's <=1-ULP instrument is simply
# the wrong ruler and gets replaced by an fp64-reference band.
import os
import sys

os.environ["VLLM_WNA16_EXLLAMA"] = "1"
sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch
from vllm import _custom_ops as ops

import p1_l0_mapping as l0
from vllm.model_executor.kernels.linear.mixed_precision import (
    wna16_exllama as W)

DEV, BITS = "cuda", 4
CASES = ((5120, 8192, 1), (5120, 8192, 2), (5120, 17408, 1))


def setenv(v):
    if v is None:
        os.environ.pop("VLLM_WNA16_P2_GEMV", None)
    else:
        os.environ["VLLM_WNA16_P2_GEMV"] = v


def ndiff(a, b):
    return int(((a.float() - b.float()).abs() > 0).sum())


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


def report(name, runs):
    base = runs[0]
    ns = [ndiff(base, r) for r in runs[1:]]
    mx = [maxdiff(base, r) for r in runs[1:]]
    alld = [r.data_ptr() for r in runs]
    print("   %-38s runs=%d n_diff_vs_0=%s max|d|=%s ptrs=%s"
          % (name, len(runs), ns, ["%.3e" % m for m in mx],
             "same" if len(set(alld)) == 1 else "differ"), flush=True)


def main():
    for K, N, M in CASES:
        kern, layer, codes, scales, zp = l0.make_layer_and_kernel(K, N)
        w_q, w_s, w_zp, g_idx = kern._get_weight_params(layer)
        torch.manual_seed(7)
        x = torch.randn(M, K, device=DEV, dtype=torch.float16)
        print("== K=%d N=%d M=%d  p2_part=%s _cbuf=0x%x =="
              % (K, N, M, kern._p2_part is not None, kern._cbuf.data_ptr()),
              flush=True)
        setenv(None)
        r = [kern.apply_weights(layer, x).clone() for _ in range(6)]
        report("[1] apply_weights x6 (env unset)", r)
        a = [kern.apply_weights(layer, x).clone() for _ in range(3)]
        setenv("0")
        b = [kern.apply_weights(layer, x).clone() for _ in range(3)]
        setenv("1")
        W._P2_HITS.clear()
        c = [kern.apply_weights(layer, x).clone() for _ in range(3)]
        print("   [2] env unset/0/1  n_diff(unset vs 0)=%s  "
              "(0 vs 1)=%s  hits=%d" %
              ([ndiff(u, v) for u, v in zip(a, b)],
               [ndiff(u, v) for u, v in zip(b, c)],
               len(W._P2_HITS)),
              flush=True)
        setenv(None)
        # [3]/[4] bypass apply_weights entirely: pure incumbent, two buffer
        # policies. tdq mirrors apply_weights' choice for M <= 50.
        tdq = torch.empty(0, dtype=torch.float16, device=DEV)
        out3 = []
        for _ in range(4):
            o = kern._cbuf[:M]
            o.zero_()
            ops.gptq_gemm_out(o, x, w_q, w_zp, w_s, g_idx, True, True, BITS, tdq)
            torch.cuda.synchronize()
            out3.append(o.clone())
        report("[3] direct op into _cbuf[:M] x4", out3)
        out4 = []
        for _ in range(4):
            o = torch.zeros(M, N, device=DEV, dtype=torch.float16)
            ops.gptq_gemm_out(o, x, w_q, w_zp, w_s, g_idx, True, True, BITS, tdq)
            torch.cuda.synchronize()
            out4.append(o.clone())
        report("[4] direct op into fresh zeros x4", out4)
        one = torch.empty(0, dtype=torch.float16, device=DEV)
        out5 = []
        for _ in range(4):
            o = kern._cbuf[:M]
            o.zero_()
            t2 = torch.empty(0, dtype=torch.float16, device=DEV)
            ops.gptq_gemm_out(o, x, w_q, w_zp, w_s, g_idx, True, True, BITS, t2)
            torch.cuda.synchronize()
            out5.append(o.clone())
            del t2
        report("[5] direct op, slab + fresh tdq x4", out5)
        del kern, layer, w_q, w_s, w_zp, out3, out4, out5, one
        torch.cuda.empty_cache()
    print("DISCRIM-DONE", flush=True)


if __name__ == "__main__":
    main()
