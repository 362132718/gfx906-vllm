"""Read the ExLlama kernel's reconstructed weight matrix directly.

With scale=1.0 and zp=0 and x = identity, out[k, n] == the 4-bit code the kernel
believes sits at (k, n). Comparing that grid against the codes we packed shows
the exact permutation instead of guessing at conventions.
"""

import os

os.environ["VLLM_WNA16_EXLLAMA"] = "1"

import torch

import vllm.model_executor.parameter as pmod

pmod.get_tensor_model_parallel_rank = lambda: 0
pmod.get_tensor_model_parallel_world_size = lambda: 1

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.wna16_exllama import (
    WNA16ExllamaKernel,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

DEV = "cuda"
_WL = lambda *a, **k: None  # noqa: E731

K, N, G = 256, 64, 32

# codes[k, n] = k % 16  -> every K row has a distinct, recognisable nibble value
codes_k = (torch.arange(K, device=DEV, dtype=torch.int64) % 16)
codes = codes_k.view(K, 1).expand(K, N).contiguous()      # [K, N]
scale = torch.ones(K // G, N, dtype=torch.float16, device=DEV)
zp = torch.zeros(K // G, N, dtype=torch.int8, device=DEV)

shifts = torch.arange(8, device=DEV, dtype=torch.int64) * 4
codes_nk = codes.T.contiguous()                            # [N, K]
w_packed = torch.sum(codes_nk.view(N, K // 8, 8) << shifts.view(1, 1, 8), dim=2).int()
w_scale = scale.T.contiguous()                             # [N, K//G]
w_zp = torch.sum(
    zp.T.contiguous().view(N // 8, 8, K // G) << shifts.view(1, 8, 1), dim=1).int()

layer = torch.nn.Module()
layer.weight_packed = PackedvLLMParameter(
    input_dim=1, output_dim=0, weight_loader=_WL, packed_factor=8, packed_dim=1,
    data=w_packed)
layer.weight_scale = GroupQuantScaleParameter(
    output_dim=0, input_dim=1, weight_loader=_WL, data=w_scale)
layer.weight_zero_point = PackedvLLMParameter(
    input_dim=1, output_dim=0, weight_loader=_WL, packed_factor=8, packed_dim=0,
    data=w_zp)

c = MPLinearLayerConfig(
    full_weight_shape=(K, N), partition_weight_shape=(K, N),
    weight_type=scalar_types.uint4, act_type=torch.float16,
    group_size=G, zero_points=True, has_g_idx=False)
k = WNA16ExllamaKernel(c, "weight_packed", "weight_scale", "weight_zero_point", None)
k.process_weights_after_loading(layer)

print("packed qweight shape:", tuple(layer.weight_packed.shape),
      "expected [K//8, N] =", (K // 8, N))
print("qweight word[0..3, 0]:", layer.weight_packed[:4, 0].tolist())
print("expected word for k=0..7,n=0:",
      sum(int(codes[j, 0]) << (4 * j) for j in range(8)))
print("scales shape:", tuple(layer.weight_scale.shape), "g_idx:", layer.g_idx[:G*2].tolist())

x = torch.eye(K, dtype=torch.float16, device=DEV)
out = k.apply_weights(layer, x)
recon = out.float().round().clamp(0, 15).long()   # [K, N] reconstructed code grid

print("\nrecon[:6, :4] =", recon[:6, :4].tolist())
print("true [:6, :4] =", codes[:6, :4].tolist())

ok = (recon == codes)
print("exact match fraction: %.4f" % ok.float().mean().item())

# Try plausible row remappings to identify the permutation.
for label, idx in [
    ("nibble-reverse within each 8", None),
]:
    pass

rev = codes.view(K // 8, 8, N).flip(1).reshape(K, N)
print("matches nibble-reversed-in-word:", (recon == rev).float().mean().item())

# row interleave: k -> (k%8)*(K//8) + k//8  or transpose-of-view variants
for name, perm in [
    ("k -> 8*(k//8) + (7 - k%8)",
     codes.view(K // 8, 8, N).flip(1).reshape(K, N)),
    ("k -> (k%8)*(K//8)+(k//8)",
     codes.view(K // 8, 8, N).transpose(0, 1).reshape(K, N)),
    ("k -> (k//8) + (k%8)*(K//8)",
     codes.view(K // 8, 8, N).transpose(0, 1).contiguous().view(K, N)),
]:
    print("matches %-28s: %.4f" % (name, (recon == perm).float().mean().item()))

print("\nper-column distinct k-values seen correct at n=0:",
      (recon[:, 0] == codes[:, 0]).sum().item(), "/", K)
print("recon[:,0][:24] =", recon[:24, 0].tolist())
print("true [:,0][:24] =", codes[:24, 0].tolist())
