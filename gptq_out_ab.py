# gptq_out_ab.py -- capture-safe gptq_gemm_out must match legacy gptq_gemm
# BIT-EXACTLY per M (both hit the same fused kernel for M<=48; M=64 hits
# reconstruct+cuBLAS in both).
import torch
import vllm._custom_ops as ops  # noqa: F401

dev = "cuda"
bits = 4
K, N, G = 512, 256, 128
gen = torch.Generator(device="cpu").manual_seed(11)

per_word = 32 // bits
seq = torch.randint(0, 16, (K, N), generator=gen).to(torch.int64).to(dev)
plain = (seq[0::2] | (seq[1::2] << bits)).contiguous().to(torch.int32)
z = torch.randint(0, 16, (K // G, N), generator=gen).to(torch.int64).to(dev)
zw = z.view(K // G, N // 8, per_word)
zp32 = torch.zeros(K // G, N // 8, dtype=torch.int32, device=dev)
for i in range(per_word):
    zp32 = zp32 | (zw[:, :, i].to(torch.int32) << (bits * i))
scales = (torch.randn(K // G, N, generator=gen) * 0.01).to(dev).half()
a = torch.randn(64, K, generator=gen).to(dev).half()
gidx = torch.empty(0, dtype=torch.int32, device=dev)

fails = 0
for M in (1, 2, 8, 48, 64):
    ref = ops.gptq_gemm(a[:M], plain, zp32, scales, gidx, True, True, bits)
    c_out = torch.zeros((M, N), dtype=torch.float16, device=dev)
    if M > 50:  # reconstruct scratch, shared pool style
        tdq = torch.empty(K, N, dtype=torch.float16, device=dev)
    else:
        tdq = torch.empty(0, dtype=torch.float16, device=dev)
    ops.gptq_gemm_out(c_out, a[:M], plain, zp32, scales, gidx, True, True,
                      bits, tdq)
    d = (c_out.float() - ref.float()).abs().max().item()
    nan = bool(torch.isnan(c_out).any().item())
    # The fused exllama kernel is non-deterministic at ~1 fp16 ULP on wave64
    # (identically true for the pre-B3 binary), so small-M needs a tolerance;
    # the reconstruct+cuBLAS path (M>50) is bit-exact.
    tol = 0.0 if M > 50 else 0.05
    ok = d <= tol and not nan
    fails += 0 if ok else 1
    print("M=%2d maxdiff=%.3e tol=%.3f nan=%s -> %s"
          % (M, d, tol, nan, "OK" if ok else "FAIL"), flush=True)

print("ALL_EXACT" if fails == 0 else "%d FAILS" % fails)
raise SystemExit(1 if fails else 0)
