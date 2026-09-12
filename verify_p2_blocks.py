"""Verify the P2 plan's code blocks actually run inside the delivery container.

Nothing here is the P2 implementation task -- it is a plan-executability check
that targets the "the plan cannot even start" defect class the review flagged
(P0-1 mutates_args, P0-2 type annotations, P0-3 duplicate definition, and the
rc-pass-through path added while dispositioning the review).

Run in the container:  python3 verify_p2_blocks.py
"""
import importlib.util
import sys

import torch

sys.path.insert(0, "/work/p1")


def load_block_module():
    spec = importlib.util.spec_from_file_location(
        "p2b", "/work/p2check/p2_patch_blocks.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # raises on the P0-1/P0-2 defects
    return m


def dequant_fp64(qw, sc, zr, K, N):
    """Zero自由 fp64 reference weights: code i of word (kw,n) at bits
    (i//2)*4 + (i%2)*16 (L0-pinned); GPTQv2 stored nibble == true zp."""
    u = qw.to(torch.int64) & 0xFFFFFFFF                    # [K/8, N] unsigned
    codes = torch.stack([(u >> (4 * (i // 2) + 16 * (i % 2))) & 0xF
                         for i in range(8)], 0)             # [8, K/8, N]
    code = codes.permute(1, 0, 2).reshape(K, N)
    zz = zr.to(torch.int64) & 0xFFFFFFFF                   # [K/128, N/8]
    zps = torch.stack([(zz >> (4 * j)) & 0xF for j in range(8)], 0)
    # output n reads nibble (n & 7) of word (g, n >> 3) => nibble is the
    # INNERMOST axis of N, not a slow-varying one.
    zp = zps.permute(1, 2, 0).reshape(K // 128, N)         # [K/128, N]
    return (code - zp.repeat_interleave(128, 0)) * \
        sc.to(torch.float64).repeat_interleave(128, 0)


def ulp_stats(got, ref64):
    ref16 = ref64.half()
    want = torch.nextafter(ref16, torch.full_like(ref16, float("inf")))
    up = (want.float() - ref16.float()).abs().clamp_min(1e-6)
    dn = (ref16.float() - torch.nextafter(
        ref16, torch.full_like(ref16, -float("inf"))).float()).abs().clamp_min(1e-6)
    step = torch.where(ref16.float().abs() > 0, (up + dn) / 2, up)
    d = (got.float() - ref16.float()).abs() / step
    return float(d.pow(2).mean().sqrt()), float(d.mean()), float(d.max())


def main():
    torch.manual_seed(0)
    K, N = 5120, 7168
    m = load_block_module()
    print("[1] patch blocks import: _P2_OK=%s (expect True)" % m._P2_OK)
    assert m._P2_OK, "loader says unavailable: is libp2gemv.so at the dist path?"
    assert hasattr(torch.ops.vllm, "p2_w4a16_gemm"), "custom op not registered"
    # The dispatch op is the G-P3 fix: it must register too, i.e. infer_schema
    # accepts `-> None` + mutates_args=["part","out"] on 9 tensor/int args.
    assert hasattr(torch.ops.vllm, "p2_gemv_or_fused"), \
        "dispatch op not registered"
    print("[2] custom op registered under torch.ops.vllm (mutates_args + "
          "annotations accepted by infer_schema)")

    qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), device="cuda",
                       dtype=torch.int32)
    sc = (torch.rand(K // 128, N, device="cuda") * 0.02 + 0.001).half()
    zr = torch.randint(0, 2**31 - 1, (K // 128, N // 8), device="cuda",
                       dtype=torch.int32)
    x = torch.randn(1, K, device="cuda").half()
    # part rows come from the SAME constant the impl passes -- never a literal.
    # Hardcoding 8 here while the kernel writes _P2_SPLITK rows is an
    # out-of-bounds write the .so cannot detect (it only sees a raw pointer).
    SK = m._P2_SPLITK
    assert K % 128 == 0 and (K // 128) % SK == 0, "SK must divide K/GROUP"
    part = torch.empty((SK, N), dtype=torch.float32, device="cuda")
    out = torch.empty(1, N, dtype=torch.float16, device="cuda")

    got = torch.ops.vllm.p2_w4a16_gemm(x, qw, sc, zr, part, out)
    torch.cuda.synchronize()
    ref64 = (x.double() @ dequant_fp64(qw, sc, zr, K, N)).view(N)
    rms, mean, mx = ulp_stats(got, ref64)
    print("[3] fp64 ULP  RMS=%.4f MEAN=%.4f MAX=%.1f  (gate: RMS <= 0.05)"
          % (rms, mean, mx))
    assert rms <= 0.05, "numeric regression: the plan's kernel is wrong"

    try:
        bad = torch.empty(1, 192, dtype=torch.float16, device="cuda")
        bq = torch.randint(-2**31, 2**31 - 1, (24, 64), device="cuda",
                           dtype=torch.int32)
        bs = torch.rand(2, 64, device="cuda").half()
        bz = torch.randint(0, 2**31 - 1, (2, 8), device="cuda", dtype=torch.int32)
        bp = torch.empty((SK, 64), dtype=torch.float32, device="cuda")
        bo = torch.empty(1, 64, dtype=torch.float16, device="cuda")
        torch.ops.vllm.p2_w4a16_gemm(bad, bq, bs, bz, bp, bo)
        print("[4] FAIL: a rejected shape silently passed")
    except RuntimeError as e:
        print("[4] rc pass-through works: %s" % str(e).splitlines()[0][:80])

    # capture smoke: the ctypes launch must land on torch's current stream.
    # A pre-vs-post-replay comparison alone is NOT falsifiable -- if the
    # launches were never captured, `out` is simply left alone and the
    # comparison still says True (this is exactly how P1 measured 1.03 us).
    # So poison the buffer after capture and require the replay to rewrite it.
    gr = torch.cuda.CUDAGraph()
    fn = lambda: torch.ops.vllm.p2_w4a16_gemm(x, qw, sc, zr, part, out)  # noqa: E731
    fn(); torch.cuda.synchronize()
    with torch.cuda.graph(gr):
        fn()
    a = out.clone()
    out.zero_()
    gr.replay(); torch.cuda.synchronize()
    b = out.clone()
    print("[5] replay rewrote the poisoned buffer: %s | bit-identical to "
          "pre-capture: %s" % (bool((b != 0).any()), bool(torch.equal(a, b))))
    assert bool(torch.equal(a, b)) and bool((b != 0).any()), \
        "graph replay did not run the kernel"
    print("PLAN-BLOCKS-OK")


if __name__ == "__main__":
    main()
