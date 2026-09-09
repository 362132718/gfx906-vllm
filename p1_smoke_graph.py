# p1_smoke_graph.py -- verify the stream fix: graph-replayed timing must now be
# ABOVE the memory floor (the broken build measured 1.03 us for every variant,
# i.e. the graph replayed nothing).
import os
os.environ["VLLM_WNA16_EXLLAMA"] = "1"
import sys
import torch
from torch.utils.cpp_extension import load

mod = load(name="p1_kernels", sources=["/work/p1/p1_kernels.cu"],
           extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
           build_directory="/tmp/p1_bench_build", verbose=False)
sys.path.insert(0, "/work/p1")
import p1_l1_bench as b

K, N = 5120, 17408
x, qw, sc, zr = b.tensors(K, N, 1)
scratch = torch.zeros(N, device=b.DEV, dtype=torch.float32)
fn = lambda: mod.p1_gemv(x, qw, sc, zr, scratch, 1, 1, 8, True)
eager = b.timeit(fn, warm=8, reps=24, R=4)
gr = b.timeit(b.graphed(fn), warm=8, reps=24, R=4)
print("V1 sk1 u8 (5120,17408): eager=%.2f us graph=%.2f us (floor 56.9 us) %s" % (
    eager, gr, "GRAPH-OK" if gr > 40 else "GRAPH-STILL-BROKEN"))
