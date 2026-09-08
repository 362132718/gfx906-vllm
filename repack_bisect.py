"""Bisect the WNA16ExllamaKernel mismatch by disabling one input at a time.

If qweight+qzeros are all-zero the only remaining transform is the raw 4-bit
unpack, so a small error there means the zero-point repack is the culprit;
a large error means the weight packing itself is wrong.
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

print("SCHEMA:", torch.ops._C.gptq_gemm.default._schema)


def build(K, N, G, M, zero_zp=False, const_scale=None, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int64).to(DEV)
    zp = torch.zeros(K // G, N, dtype=torch.int8, device=DEV) if zero_zp else (
        torch.randint(0, 16, (K // G, N), generator=g, dtype=torch.int8).to(DEV)
    )
    if const_scale is not None:
        scale = torch.full((K // G, N), const_scale, dtype=torch.float16, device=DEV)
    else:
        scale = (
            torch.rand(K // G, N, generator=g, dtype=torch.float32) * 0.02 + 0.003
        ).half().to(DEV)

    shifts = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    w_packed = torch.sum(codes.view(N, K // 8, 8) << shifts.view(1, 1, 8), dim=2).int()
    w_scale = scale.T.contiguous()
    z = zp.T.contiguous().view(N // 8, 8, K // G)
    w_zp = torch.sum(z << shifts.view(1, 8, 1), dim=1).int()

    zp_full = zp.float().T.repeat_interleave(G, dim=1)
    scale_full = scale.float().T.repeat_interleave(G, dim=1)
    ref_w = (codes.float() - zp_full) * scale_full
    x = torch.randn(M, K, generator=g, dtype=torch.float32).half().to(DEV)
    ref = (x.float() @ ref_w.float().T).half()

    layer = torch.nn.Module()
    layer.weight_packed = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=_WL,
        packed_factor=8, packed_dim=1, data=w_packed)
    layer.weight_scale = GroupQuantScaleParameter(
        output_dim=0, input_dim=1, weight_loader=_WL, data=w_scale)
    layer.weight_zero_point = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=_WL,
        packed_factor=8, packed_dim=0, data=w_zp)
    return layer, x, ref


def probe(name, K=1024, N=1024, G=128, M=1, **kw):
    layer, x, ref = build(K, N, G, M, **kw)
    c = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4, act_type=torch.float16,
        group_size=G, zero_points=not kw.get("zero_zp", False) or True,
        has_g_idx=False,
    )
    k = WNA16ExllamaKernel(
        c, "weight_packed", "weight_scale", "weight_zero_point", None)
    k.process_weights_after_loading(layer)
    out = k.apply_weights(layer, x).reshape(M, N)
    d = (out.float() - ref.float()).abs()
    print("%-30s maxabs=%9.4f  ref_absmean=%7.4f  ex_absmean=%7.4f  ratio=%.3f"
          % (name, d.max().item(), ref.float().abs().mean().item(),
             out.float().abs().mean().item(),
             (out.float().abs().mean() / ref.float().abs().mean()).item()))
    print("      ex0[:6] =%s" % ["%.3f" % v for v in out[0, :6].tolist()])
    print("      ref0[:6]=%s" % ["%.3f" % v for v in ref[0, :6].tolist()])


probe("full (zp random)")
probe("zp = 0", zero_zp=True)
probe("zp = 0, scale = 1.0", zero_zp=True, const_scale=1.0)
probe("zp = 8, scale = 1.0", zero_zp=True, const_scale=1.0)
probe("zp random, scale=1.0", const_scale=1.0)
