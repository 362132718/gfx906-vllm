"""P6b: the missing half of the P6 decision.

p6_tflops.py measured the CANDIDATE (dequantize INT4 -> dense fp16 GEMM).
This measures the INCUMBENT: the ExLlama-fused W4A16 path our delivery image
actually uses at prefill M (i.e. ops.gptq_gemm(use_exllama=True)).

The P6 gate ("< 15 TFLOPS => do not build") was only a proxy. The real question
is comparative, so measure both at the same shapes and compare directly.

Timing only -- synthetic weights, correctness is not the point here.
"""
import json
import time

import torch
from vllm import _custom_ops as ops

DEV = "cuda"
GROUP = 32  # Z100's model was group32; ours is compressed-tensors


def timeit(fn, iters=15, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print("device:", torch.cuda.get_device_name(0))
    print("incumbent path: ops.gptq_gemm(..., use_exllama=True, use_exllama_v2=True, bit=4)")
    print(f"group_size={GROUP}")
    print()

    SHAPES = [("QKV ", 8192, 8192, 5120),
              ("Gate", 8192, 17408, 5120),
              ("Down", 8192, 5120, 8704)]

    out = []
    for name, M, N, K in SHAPES:
        a = torch.randn(M, K, dtype=torch.float16, device=DEV)
        b_q = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=DEV)
        b_zp = torch.randint(-2**31, 2**31 - 1, (K // GROUP, N), dtype=torch.int32, device=DEV)
        b_s = torch.randn(K // GROUP, N, dtype=torch.float16, device=DEV)
        g_idx = torch.empty(0, dtype=torch.int32, device=DEV)
        try:
            fn = lambda a=a, b_q=b_q, b_zp=b_zp, b_s=b_s, g_idx=g_idx: ops.gptq_gemm(
                a.contiguous(), b_q, b_zp, b_s, g_idx, True, True, 4)
            _ = fn()
            torch.cuda.synchronize()
            dt = timeit(fn)
            flops = 2.0 * M * N * K
            tf = flops / dt / 1e12
            print(f"  {name} {M}x{N}x{K}  incumbent {dt*1e3:8.2f} ms  {tf:6.2f} TFLOPS-equiv")
            out.append({"name": name, "M": M, "N": N, "K": K, "ms": dt * 1e3, "tflops_eq": tf})
        except Exception as e:  # noqa: BLE001
            print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:200]}")
            out.append({"name": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        finally:
            del a, b_q, b_zp, b_s, g_idx
            torch.cuda.empty_cache()

    print()
    print("== direct comparison at M=8192 (candidate vs incumbent) ==")
    CAND = {"QKV ": 66.12, "Gate": 81.95, "Down": 41.09}      # from p6_tflops.py
    DEQ = {"QKV ": 1.20, "Gate": 2.55, "Down": 1.34}
    for r in out:
        if "ms" not in r:
            continue
        n = r["name"]
        cand = CAND[n] + DEQ[n]
        inc = r["ms"]
        print(f"  {n} incumbent {inc:8.2f} ms | candidate(dequant+GEMM) {cand:8.2f} ms"
              f" | speedup {inc/cand:5.2f}x")

    print()
    ok = [r for r in out if "ms" in r]
    if ok:
        gains = [r["ms"] / (CAND[r["name"]] + DEQ[r["name"]]) for r in ok]
        avg = sum(gains) / len(gains)
        print(f"  mean speedup over the 3 projections: {avg:.2f}x")
        print("P6B_VERDICT " + json.dumps({"mean_speedup": avg,
                                           "build": avg > 1.25,
                                           "rows": out}))
        if avg > 1.25:
            print("  -> incumbent is materially slower: the prefill rewrite has real upside")
        else:
            print("  -> no material upside at these shapes: do NOT build")


if __name__ == "__main__":
    main()
