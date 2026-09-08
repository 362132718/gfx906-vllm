# gptq_out_ab2.py -- op-level reproduction across shapes, incl. the model's
# out_proj partition (K=3072,N=5120) that produced all-zero outputs.
import torch
import vllm._custom_ops as ops  # noqa: F401

dev = "cuda"
bits = 4


def run_shape(K, N, G):
    gen = torch.Generator(device="cpu").manual_seed(11)
    per_word = 32 // bits
    seq = torch.randint(0, 16, (K, N), generator=gen).to(torch.int64).to(dev)
    plain = (seq[0::2] | (seq[1::2] << bits)).contiguous().to(torch.int32)
    z = torch.randint(0, 16, (K // G, N), generator=gen).to(torch.int64).to(dev)
    zw = z.view(K // G, N // per_word, per_word)
    zp = torch.zeros(K // G, N // per_word, dtype=torch.int32, device=dev)
    for i in range(per_word):
        zp = zp | (zw[:, :, i].to(torch.int32) << (bits * i))
    sc = (torch.randn(K // G, N, generator=gen) * 0.01).to(dev).half()
    gidx = torch.empty(0, dtype=torch.int32, device=dev)
    fails = 0
    for M in (1, 2, 8, 48, 64, 1024, 8192):
        a = torch.randn(M, K, generator=gen).to(dev).half()
        ref = ops.gptq_gemm(a, plain, zp, sc, gidx, True, True, bits)
        c_out = torch.zeros((M, N), dtype=torch.float16, device=dev)
        tdq = (torch.empty(K, N, dtype=torch.float16, device=dev) if M > 50
               else torch.empty(0, dtype=torch.float16, device=dev))
        ops.gptq_gemm_out(c_out, a, plain, zp, sc, gidx, True, True, bits, tdq)
        d = (c_out.float() - ref.float()).abs().max().item()
        ref_zero = ref.float().abs().sum().item() == 0
        tol = 0.0 if M > 50 else 0.05
        ok = (not ref_zero) and d <= tol
        fails += 0 if ok else 1
        print("K=%d N=%d M=%4d maxdiff=%.3e ref_zero=%s -> %s"
              % (K, N, M, d, ref_zero, "OK" if ok else "FAIL"), flush=True)
    return fails


fails = run_shape(512, 256, 128) + run_shape(3072, 5120, 128)
print("ALL_EXACT" if fails == 0 else "%d FAILS" % fails)
raise SystemExit(1 if fails else 0)
