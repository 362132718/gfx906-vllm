"""Isolate the ExLlama fused W4A16 GEMM at 27B-scale shapes, one subprocess per case.

Motivation: the full 27B TP2 run with the fixed bridge coincided with the host
becoming completely unreachable. The single-GPU correctness probes only used
K<=5120 and N<=5120. This sweeps the shapes a 27B actually hits so a GPU lockup
is attributable to one (K, N, M, group) tuple instead of the whole engine.

Each case runs in its own process: a hard lock shows up as the last printed
line, and a clean Python exception is reported as ERR.

Run:  python3 exllama_shape_stress.py [--quick]
"""

import argparse
import json
import subprocess
import sys

CHILD = r"""
import json, sys, time
import torch
import vllm.model_executor.parameter as pmod
pmod.get_tensor_model_parallel_rank = lambda: 0
pmod.get_tensor_model_parallel_world_size = lambda: 1

from vllm import _custom_ops as ops

K, N, G, M, bits = [int(v) for v in sys.argv[1:6]]
dev = "cuda"
torch.manual_seed(0)
per_word = 32 // bits
words = torch.randint(-(2**31), 2**31, (K // per_word, N), dtype=torch.int32, device=dev)
scales = (torch.rand(K // G, N, device=dev, dtype=torch.float16) - 0.5) * 0.05
qzeros = torch.randint(0, 2**31, (K // G, N // per_word), dtype=torch.int32, device=dev)
g_idx = torch.empty((0,), dtype=torch.int32, device=dev)
a = torch.randn(M, K, device=dev, dtype=torch.float16)

torch.cuda.synchronize()
t0 = time.perf_counter()
out = ops.gptq_gemm(a, words, qzeros, scales, g_idx, True, True, bits)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1e3
assert tuple(out.shape) == (M, N), out.shape
fin = bool(torch.isfinite(out.float()).all().item())
# second call to make sure it is not a one-off
t1 = time.perf_counter()
for _ in range(5):
    out = ops.gptq_gemm(a, words, qzeros, scales, g_idx, True, True, bits)
torch.cuda.synchronize()
dt2 = (time.perf_counter() - t1) * 1e3 / 5
mem = torch.cuda.max_memory_allocated() / 2**20
print("RESULT " + json.dumps({"K": K, "N": N, "G": G, "M": M, "bits": bits,
                              "first_ms": round(dt, 3), "avg_ms": round(dt2, 3),
                              "finite": fin, "peak_mb": round(mem)}))
"""

# (K, N, group, bits) tuples a Qwen3-27B-class W4A16 model hits: hidden 5120,
# q/kv heads, MoE-less MLP, plus the vocab-sized lm_head.
CASES = [
    (5120, 5120, 128, 4),      # q/k/v/o projections
    (5120, 1024, 128, 4),      # kv only
    (5120, 17408, 128, 4),     # mlp gate/up
    (17408, 5120, 128, 4),     # mlp down  (largest K)
    (5120, 25600, 128, 4),     # lm_head   (largest N)
    (5120, 5120, 64, 4),
    (17408, 5120, 128, 8),     # 8-bit path
]
MS = [1, 8, 64]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only M=1 and M=64")
    args = ap.parse_args()
    ms = [1, 64] if args.quick else MS

    for K, N, G, bits in CASES:
        for M in ms:
            if (K // 32 * bits * N) * 2 > 6 * 2**30:  # skip >6GiB weight tensors
                print(f"SKIP K={K} N={N} M={M} bits={bits} (weight too large)")
                continue
            cmd = [sys.executable, "-c", CHILD, str(K), str(N), str(G), str(M), str(bits)]
            print(f"--> K={K} N={N} G={G} M={M} bits={bits}", flush=True)
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                print("    HANG: no result within 300s", flush=True)
                return 2
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                print("   ", line[0][7:], flush=True)
            else:
                err = (p.stderr.strip().splitlines() or ["<no stderr>"])[-1]
                print(f"    ERR rc={p.returncode}: {err[:200]}", flush=True)
    print("STRESS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
