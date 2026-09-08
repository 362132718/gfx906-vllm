# p05_gate_test.py -- which calls must reach ops.LLMM1, and which must not?
# Both seams are monkeypatchable: rocm_unquantized_gemm_impl looks up ops.LLMM1
# at call time, and imports on_gfx906 from vllm.platforms.rocm *inside* the
# function body, so the import re-reads the module attribute on every call.
#   ssh x@192.168.10.11 'docker exec -i vllm-gfx906 python3 /work/p0/p05_gate_test.py'
import torch
import vllm._custom_ops as ops
import vllm.model_executor.layers.utils as U          # noqa: F401 (registers op)
import vllm.platforms.rocm as R
from vllm import envs

calls = []
_real = ops.LLMM1
ops.LLMM1 = lambda a, b, rpb: (calls.append(rpb), _real(a, b, rpb))[1]
torch.manual_seed(7)


def dispatch(n, m, k, dtype=torch.float16, bias=False, contig=True,
             gfx906=True, skinny=True):
    """One call through the registered custom op. Returns (#LLMM1 calls, out)."""
    w = (torch.randn(k, m, device="cuda", dtype=torch.float16).t() if not contig
         else torch.randn(m, k, device="cuda", dtype=dtype) / k ** 0.5)
    if contig and dtype != torch.float16:
        w = w.to(dtype)
    x = torch.randn(n, k, device="cuda", dtype=dtype)
    b = torch.ones(m, device="cuda", dtype=torch.float16) if bias else None
    envs.VLLM_ROCM_USE_SKINNY_GEMM = skinny
    orig, R.on_gfx906 = R.on_gfx906, (lambda: gfx906)
    calls.clear()
    try:
        y = torch.ops.vllm.rocm_unquantized_gemm(x, w, b)
    finally:
        R.on_gfx906 = orig
        del envs.VLLM_ROCM_USE_SKINNY_GEMM      # restore the lazy default
    torch.cuda.synchronize()
    return len(calls), y


def expect(label, n, want):
    print("%-42s LLMM1=%d want=%d %s" % (label, n, want, "ok" if n == want else "FAIL"))
    assert n == want, label


got, y = dispatch(1, 48, 5120)
expect("M=1 in_proj_ba-shaped -> skinny", got, 1)
assert tuple(y.shape) == (1, 48), y.shape
expect("M=1 replicated N=96 -> skinny", dispatch(1, 96, 5120)[0], 1)
expect("M=2 must stay on Tensile", dispatch(2, 48, 5120)[0], 0)
expect("bias present -> fallback", dispatch(1, 48, 5120, bias=True)[0], 0)
expect("k=12288 > 8192 -> fallback", dispatch(1, 48, 12288)[0], 0)
expect("m=50 (m%4!=0) -> fallback", dispatch(1, 50, 5120)[0], 0)
expect("non-contiguous weight -> fallback", dispatch(1, 48, 5120, contig=False)[0], 0)
expect("not gfx906 -> unchanged", dispatch(1, 48, 5120, gfx906=False)[0], 0)
expect("VLLM_ROCM_USE_SKINNY_GEMM=0 -> fallback",
       dispatch(1, 48, 5120, skinny=False)[0], 0)
# Task 1 measured bf16_ok=True, so the gate carries the bf16 dtype variant.
# If you ship the float16-only predicate instead, this expectation is 0.
expect("bf16 -> skinny (Task 1 bf16_ok)",
       dispatch(1, 48, 5120, dtype=torch.bfloat16)[0], 1)
print("ALL PASS")
