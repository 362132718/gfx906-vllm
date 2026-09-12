"""P6: fp16 GEMM throughput probe for gfx906 (MI50 / Radeon Pro VII).

Purpose: decide whether Z100's "dequantize INT4 -> dense fp16 GEMM" prefill
rewrite is worth building (their M=8192 was 3.22-3.58x, 8K model-level +96.5%).

Gate from the v3 plan:  < 15 TFLOPS  => fix rocBLAS first, do NOT build it
                        >= 15 TFLOPS => build it

Also prices the dequant step, because the whole idea rests on
"one dequant per layer, then thousands of activation rows reuse it".

Run detached (host has a silent-power-loss history):
    docker exec -d <c> bash -lc 'python3 /tmp/p6_tflops.py > /home/x/vllm-work/p6_tflops.log 2>&1'
"""
import json
import time

import torch

DEV = "cuda"
# MI50 / Radeon Pro VII datasheet fp16 (fp32 accumulate) peak, per card.
PEAK_FP16 = 26.2
PEAK_FP32 = 13.1


def bench(M, N, K, dtype, iters=20, warmup=5):
    a = torch.randn(M, K, dtype=dtype, device=DEV)
    b = torch.randn(K, N, dtype=dtype, device=DEV)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2.0 * M * N * K
    del a, b, c
    torch.cuda.empty_cache()
    return dt, flops / dt / 1e12


def main():
    print("device :", torch.cuda.get_device_name(0))
    print("hip    :", torch.version.hip, "| torch", torch.__version__)
    print("sclk   : see the host-side hwmon read recorded next to this log")
    print()

    # --- 1. machine reference: big square matmul (best case for rocBLAS) ---
    print("== reference: square fp16 GEMM (upper bound) ==")
    for M, N, K in [(8192, 8192, 8192), (4096, 4096, 4096)]:
        dt, tf = bench(M, N, K, torch.float16, iters=10, warmup=3)
        print(f"  {M}x{N}x{K}  {dt*1e3:8.2f} ms  {tf:6.2f} TFLOPS  ({tf/PEAK_FP16*100:5.1f}% of {PEAK_FP16})")
    print()

    # --- 2. the three real 27B prefill projections at M=8192 ---
    # Qwen3.8-27B: hidden 5120, intermediate 17408, KV-ish out 8192 (TP2 per-card
    # halves N, but the single-card full-shape number is the honest upper bound).
    print("== 27B prefill shapes, fp16 ==")
    SHAPES = [("QKV ", 8192, 8192, 5120),
              ("Gate", 8192, 17408, 5120),
              ("Down", 8192, 5120, 8704)]
    rows = []
    for name, M, N, K in SHAPES:
        dt, tf = bench(M, N, K, torch.float16, iters=15, warmup=4)
        rows.append((name, M, N, K, dt, tf))
        print(f"  {name} {M}x{N}x{K}  {dt*1e3:8.2f} ms  {tf:6.2f} TFLOPS")
    print()

    # --- 3. M sweep on the QKV shape: where does dense GEMM stop paying? ---
    print("== QKV shape (N=8192,K=5120) vs M, fp16 ==")
    for M in [1, 2, 4, 8, 32, 128, 512, 2048, 8192]:
        dt, tf = bench(M, 8192, 5120, torch.float16, iters=30, warmup=5)
        print(f"  M={M:<5} {dt*1e3:9.3f} ms  {tf:7.2f} TFLOPS")
    print()

    # --- 4. dtype comparison (BF16 is the known gfx906 trap; FP32 = the floor) ---
    print("== dtype comparison at QKV M=8192 ==")
    for nm, dt_ in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        try:
            d, tf = bench(8192, 8192, 5120, dt_, iters=10, warmup=3)
            print(f"  {nm:5} {d*1e3:8.2f} ms  {tf:6.2f} TFLOPS")
        except Exception as e:  # noqa: BLE001
            print(f"  {nm:5} FAILED: {type(e).__name__}: {e}")
    print()

    # --- 5. price the dequant step (the idea's whole premise) ---
    # One layer's INT4 weight unpacked to fp16 once, then reused by every row.
    print("== INT4 -> fp16 dequant cost (one-time per layer, per projection) ==")
    for name, N, K in [("QKV ", 8192, 5120), ("Gate", 17408, 5120), ("Down", 5120, 8704)]:
        packed = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=DEV)
        scale = torch.randn(N, K // 32, dtype=torch.float16, device=DEV)

        def dequant():
            lo = (packed & 0x0F).to(torch.float16)
            hi = ((packed >> 4) & 0x0F).to(torch.float16)
            inter = torch.stack((lo, hi), dim=-1).reshape(N, K)
            return inter  # true dequant also needs scale/zero-point; this is the floor

        for _ in range(3):
            out = dequant()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = dequant()
        torch.cuda.synchronize()
        dq_ms = (time.perf_counter() - t0) / 10 * 1e3
        bytes_moved = (N * K // 2) + (N * K * 2)
        print(f"  {name} {N}x{K}  dequant {dq_ms:7.3f} ms  "
              f"(floor est. {bytes_moved/800e9*1e3:6.3f} ms at 800 GB/s)")
        del packed, scale, out
        torch.cuda.empty_cache()
    print()

    # --- verdict ---
    tf_qkv = rows[0][5]
    print("== VERDICT ==")
    print(f"  QKV-shape dense fp16 GEMM at M=8192: {tf_qkv:.2f} TFLOPS")
    if tf_qkv >= 15:
        print(f"  >= 15 TFLOPS  -> BUILD the 'dequantize then dense GEMM' prefill path")
    else:
        print(f"  <  15 TFLOPS  -> fix rocBLAS first; do NOT build it yet")
    print()
    print("P6_RESULT " + json.dumps({
        "ref_8192": None,
        "shapes": [{"name": n, "M": M, "N": N, "K": K, "ms": d, "tflops": t}
                   for n, M, N, K, d, t in rows],
        "qkv_M8192_tflops": tf_qkv,
        "verdict": "build" if tf_qkv >= 15 else "do_not_build",
    }))


if __name__ == "__main__":
    main()
