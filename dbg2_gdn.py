# dbg2.py -- clean minimal attribution for packed-decode mismatch
import math
import torch
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode as pk,
)

DEV = "cuda"
H, HV, K, V, B = 1, 1, 8, 8, 1
scale = K ** -0.5
kn = 1.0 / math.sqrt(K)

qkv = torch.ones(B, 2 * H * K + HV * V, device=DEV, dtype=torch.float16)
a = torch.zeros(B, HV, device=DEV, dtype=torch.float16)
b = torch.zeros(B, HV, device=DEV, dtype=torch.float16)
A_log = torch.full((HV,), -20.0, device=DEV, dtype=torch.float32)
dtb = torch.zeros(HV, device=DEV, dtype=torch.float32)
idx = torch.tensor([1], dtype=torch.int32, device=DEV)


def run(s0_entries):
    """s0_entries: list of (v,k,val)."""
    init = torch.zeros(B + 1, HV, V, K, device=DEV, dtype=torch.float32)
    for v, k, val in s0_entries:
        init[1, 0, v, k] = val
    out = torch.zeros(B, 1, HV, V, device=DEV, dtype=torch.float16)
    pk(mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dtb, scale=scale,
       initial_state=init, out=out, ssm_state_indices=idx,
       use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    return out[0, 0, 0, :].clone(), init[1, 0].clone()


def expect(s0_entries):
    """Hand reference with g~0, beta=0.5, q=k=ones/sqrt(K), v=1. CPU math."""
    S = torch.zeros(V, K)
    for v, k, val in s0_entries:
        S[v, k] = val
    q = torch.full((K,), kn * scale)
    kk = torch.full((K,), kn)
    pred = S @ kk
    u = 0.5 * (1.0 - pred)
    S2 = S + torch.outer(u, kk)
    o = S2 @ q
    return o.to(DEV), S2.to(DEV)


# case A: zero initial state
oA, _ = run([])
expA, _ = expect([])
print("A o[0],o[2] got %.6f %.6f  expect %.6f" % (oA[0], oA[2], expA[0]))

# case B: single-entry initial state at (v=2,k=5)
oB, sB = run([(2, 5, 1.0)])
expB, S2B = expect([(2, 5, 1.0)])
print("B o got:", [round(x, 5) for x in oB.tolist()])
print("B o exp:", [round(x, 5) for x in expB.tolist()])
d_ok = (sB - S2B).abs().max().item()
d_T = (sB - S2B.T).abs().max().item()
print("B state diff (V,K)=%.5f  vs transposed=%.5f  qdiff=%.5f"
      % (d_ok, d_T, (oB.float() - expB).abs().max().item()))

# case C: full-row initial state to expose k-vs-v axis
c_c, s_c = run([(v, k, 0.01 * (v - k)) for v in range(V) for k in range(K)])
exp_c, S2_c = expect([(v, k, 0.01 * (v - k)) for v in range(V) for k in range(K)])
print("C state diff=%.5f transposed=%.5f"
      % ((s_c - S2_c).abs().max(), (s_c - S2_c.T).abs().max()))
print("DONE")
