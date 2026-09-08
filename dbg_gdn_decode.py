# dbg_gdn_decode.py -- minimal hand-computable case to attribute the T5/T6 mismatch
import math
import torch
import torch.nn.functional as F
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

DEV = "cuda"
H, HV, K, V = 1, 1, 8, 8
B = 1
scale = K ** -0.5

# all-ones q,k,v; near-zero gating (g~0, beta=0.5)
qkv = torch.zeros(B, 2 * H * K + HV * V, device=DEV, dtype=torch.float16)
qkv[:, :H * K] = 1.0
qkv[:, H * K:2 * H * K] = 1.0
qkv[:, 2 * H * K:] = 1.0
a = torch.zeros(B, HV, device=DEV, dtype=torch.float16)
b = torch.zeros(B, HV, device=DEV, dtype=torch.float16)          # beta = sigmoid(0)=0.5
A_log = torch.full((HV,), -20.0, device=DEV, dtype=torch.float32)  # exp(A_log)~2e-9 -> g~0
dt_bias = torch.zeros(HV, device=DEV, dtype=torch.float32)

# S0 = 0
init = torch.zeros(B + 1, HV, V, K, device=DEV, dtype=torch.float32)
idx = torch.arange(1, B + 1, dtype=torch.int32, device=DEV)
out = torch.zeros(B, 1, HV, V, device=DEV, dtype=torch.float16)

fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
    scale=scale, initial_state=init, out=out,
    ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)

# hand reference: qn=kn=ones/sqrt(8); decay~1; pred = S0.k = 0; u = 0.5*(1-0)=0.5
kn = 1.0 / math.sqrt(K)
u = 0.5 * (1.0 - 0.0)
S_expected_val = u * kn                      # S[v,k] = u * k[k]
o_expected = V * 0 and (K * S_expected_val * kn * scale)  # sum_k S*q*scale
print("kernel o      =", out.flatten()[:4].tolist(), " expected ~", o_expected)
print("kernel S row0 =", init[1, 0, 0, :4].tolist(), " expected ~", S_expected_val)
print("S is constant across v?", bool((init[1, 0] - init[1, 0, 0:1]).abs().max() < 1e-5))
print("S is constant across k?", bool((init[1, 0] - init[1, 0, :, 0:1]).abs().max() < 1e-5))

# Now nonzero single-entry initial state to detect transpose
init2 = torch.zeros(B + 1, HV, V, K, device=DEV, dtype=torch.float32)
init2[1, 0, 2, 5] = 1.0     # S[v=2,k=5]=1
out2 = torch.zeros(B, 1, HV, V, device=DEV, dtype=torch.float16)
fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
    scale=scale, initial_state=init2, out=out2,
    ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
# expected: pred[v] = sum_k S[v,k]*kn = kn for v=2 else 0
pred = torch.zeros(V, device=DEV)
pred[2] = kn
u2 = 0.5 * (1.0 - pred)
S2 = torch.outer(u2, torch.full((K,), kn, device=DEV))
S2[2, 5] += u2[2] * kn - 0.0  # decayed old entry ~1*kn added below properly:
S2 = S2.clone()
# redo cleanly: S_new = S_old(decay~1)*1 + u outer k
S2 = torch.zeros(V, K, device=DEV)
S2[2, 5] = 1.0
S2 = S2 + torch.outer(u2, torch.full((K,), kn, device=DEV))
o2 = (S2 @ (torch.full((K,), kn * scale, device=DEV)))
print("\ntranspose-case o max diff:", (out2[0, 0, :, 0] * 0 + o2 - out2[0, 0, :, 0].float()).abs().max().item())
got = init2[1, 0]
diff = (got - S2).abs()
print("state diff:", diff.max().item())
if diff.max().item() > 1e-3:
    tv, tk = torch.unravel_index(diff.argmax(), diff.shape)
    print("worst at S[v=%d,k=%d] got=%.4f want=%.4f; transposed diff=%.4f"
          % (tv, tk, got[tv, tk], S2[tv, tk], (got - S2.T).abs().max().item()))
