# p1_jit_gate.py -- can we JIT-compile and launch a gfx906 HIP kernel in this
# container at all? P1's whole microbenchmark plan hangs on this, so prove it
# before designing anything around it. Nothing here touches site-packages.
import os, shutil, subprocess, sys, time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")   # ROCm ignores it; silences a warning
import torch
from torch.utils.cpp_extension import load_inline

print("torch", torch.__version__, "hip", torch.version.hip,
      "device", torch.cuda.get_device_name(0), flush=True)

_CPP = "torch::Tensor p1_double(torch::Tensor x, int64_t variant);"
_CU = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
__global__ void p1_double_k(const __half* x, __half* y, int n, int variant) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = __half2float(x[i]);
    // variant is only here to prove the runtime-switch pattern compiles.
    y[i] = __float2half(v * 2.0f + (float)(variant % 3));
  }
}
torch::Tensor p1_double(torch::Tensor x, int64_t variant) {
  auto y = torch::empty_like(x);
  int n = (int)x.numel();
  p1_double_k<<<(n + 255) / 256, 256>>>(
      (const __half*)x.data_ptr(), (__half*)y.data_ptr(), n, (int)variant);
  return y;
}
"""
os.makedirs("/tmp/p1_gate_build", exist_ok=True)
t0 = time.time()
mod = load_inline(
    name="p1_gate",
    cpp_sources=_CPP,
    cuda_sources=_CU,
    functions=["p1_double"],          # without this, no PyInit_ is generated
    extra_cuda_cflags=["--offload-arch=gfx906", "-O2"],
    verbose=False,
    build_directory="/tmp/p1_gate_build",
)
print("compile_seconds=%.1f" % (time.time() - t0), flush=True)

x = torch.randn(4096, device="cuda", dtype=torch.float16)
for v in (0, 1, 2):
    y = mod.p1_double(x, v)
    want = (x.float() * 2.0 + (v % 3)).half()
    bad = (y.float() - want.float()).abs().max().item()
    assert bad < 2e-3, "variant %d wrong: maxdiff %g" % (v, bad)
print("KERNEL_OK variants=0,1,2", flush=True)

# The wave64 assumption P2's whole mapping rests on.
prop = torch.cuda.get_device_properties(0)
ws = getattr(prop, "warp_size", None)
print("warp_size=%s" % ws, flush=True)
assert ws in (None, 64), "unexpected wave size %r -- wave64 math invalid" % ws

for tool in ("hipcc", "llvm-objdump", "objdump"):
    print("tool %-14s %s" % (tool, shutil.which(tool) or "MISSING"), flush=True)
rc = subprocess.run(["python3", "-c",
                     "import torch;print(torch.cuda.mem_get_info(0)[0]/1e9)"],
                    capture_output=True, text=True)
print("vram_free_GB=%s" % rc.stdout.strip(), flush=True)
print("JIT-OK", flush=True)
