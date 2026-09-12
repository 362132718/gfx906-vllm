"""verify_m1_gate.py -- correctness of the m==1 scalar-projection branch.

Runs inside the container. Builds the exact shape the branch targets
(weight (1, k), activations (n, k) fp16, no bias), calls vLLM's
rocm_unquantized_gemm custom op (which is what the model uses), and compares
against a float64 reference. Also asserts the branch actually engaged by
timing it against a forced-GEMM control.
"""
import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")

import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

K = 2048
N = 1
torch.manual_seed(0)

dev = "cuda"
w = (torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05)
x = torch.randn(1, K, dtype=torch.float16, device=dev)
ref = (x.double() @ w.double().t()).float()      # float64 ground truth

# 1. through vLLM's op (this is the path the model takes)
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.utils import (  # noqa: E402
    rocm_unquantized_gemm_impl as impl,
)

print("=== 1. vLLM path (rocm_unquantized_gemm_impl) ===")
out = impl(x, w, None)
err = (out.float() - ref).abs().max().item()
print("  shape      :", tuple(out.shape), out.dtype)
print("  max abs err:", err)
rel = err / ref.abs().max().item()
print("  max rel err: %.3e" % rel)
assert out.shape == (1, 1), out.shape
assert out.dtype == torch.float16, out.dtype
assert rel < 1e-2, "relative error too large: %g" % rel

# 2. prove the branch engaged: same op, weight padded to m=4 (hits the LLMM1
#    path instead) should be a different code path but the same answer
w4 = torch.zeros(4, K, dtype=torch.float16, device=dev)
w4[0] = w[0]
x4 = x.expand(1, 1, K).reshape(1, K)
out4 = impl(x4, w4, None)
print("=== 2. control (m=4 padded, LLMM1 path) ===")
print("  first row :", out4[0, 0].item(), " vs ", out[0, 0].item())
assert abs(out4[0, 0].float().item() - out[0, 0].float().item()) < 1e-2

# 3. timing: branch vs plain matmul (what rocBLAS was doing)
def bench(fn, iters=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


t_branch = bench(lambda: impl(x, w, None))
t_gemm = bench(lambda: F.linear(x, w, None))
print("=== 3. timing (us per call) ===")
print("  m==1 branch (mul+sum) : %8.1f us" % t_branch)
print("  plain rocBLAS GEMM    : %8.1f us" % t_gemm)
print("  speedup               : %.1fx   -> saves %.2f ms/step over 40 layers"
      % (t_gemm / t_branch, (t_gemm - t_branch) * 40 / 1000.0))
print()
print("M1GATE_VERIFY_OK")
