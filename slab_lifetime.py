# slab_lifetime.py -- does a persistent output slab survive the full engine
# lifecycle (KV alloc, empty_cache, etc.) with correct op writes?
import torch
import vllm._custom_ops as ops  # noqa

dev = "cuda"
bits = 4
K, N, G = 3072, 5120, 128
gen = torch.Generator(device="cpu").manual_seed(3)
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

slab = torch.zeros((64, N), dtype=torch.float16, device=dev)
tdq_empty = torch.empty(0, dtype=torch.float16, device=dev)


def write_once(a):
    ops.gptq_gemm_out(slab[:a.shape[0]], a, plain, zp, sc, gidx, True, True,
                      bits, tdq_empty)
    torch.cuda.synchronize()
    return slab[:a.shape[0]].clone()

a1 = torch.randn(1, K, generator=gen).to(dev).half()
A = write_once(a1)
# simulate engine lifecycle: big allocs + frees
big = [torch.empty(1400 * 1024 * 1024 // 2, dtype=torch.float16, device=dev) for _ in range(2)]
for b in big:
    b.fill_(1.0)
del big
torch.cuda.empty_cache()
A2 = write_once(a1)
a2 = torch.randn(8, K, generator=gen).to(dev).half()
B = write_once(a2)
slab_before = slab[:1].clone()
A3 = write_once(a1)  # after an M=8 write, does row0 come back identical?
print("slab A==A2 (after lifecycle):", torch.equal(A, A2))
print("slab A==A3 (after M=8 write):", torch.equal(A, A3))
fresh = torch.zeros((1, N), dtype=torch.float16, device=dev)
ops.gptq_gemm_out(fresh, a1, plain, zp, sc, gidx, True, True, bits, tdq_empty)
torch.cuda.synchronize()
print("slab A==fresh single:", torch.equal(A3, fresh))
# cross-row contamination: does M=8 write touch row 0..1 beyond M? 
print("A2 abs_sum", A2.abs().sum().item(), "fresh abs_sum", fresh.abs().sum().item())
ref = ops.gptq_gemm(a1, plain, zp, sc, gidx, True, True, bits)
d = (A.float() - ref.float()).abs().max().item()
print("first-write maxdiff vs legacy:", d)
