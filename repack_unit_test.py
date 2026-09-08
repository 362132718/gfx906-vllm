"""Standalone numerical check of WNA16ExllamaKernel repack.

No vLLM engine: builds canonical compressed-tensors pack-quantized params,
runs the kernel's repack + GEMM, and compares against a plain-torch dequant.
Covers the shapes a TP2 run produces (column-parallel shards N, row-parallel
shards K) so a partitioning bug shows up without a 4-minute model load.
"""

import os

os.environ["VLLM_WNA16_EXLLAMA"] = "1"

import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.wna16_exllama import (
    WNA16ExllamaKernel,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types

DEV = "cuda"


def pack_along_last(codes, n_bits_per_word=8):
    """[..., L] small ints -> [..., L//8] int32, LSB-first nibbles."""
    shifts = torch.arange(n_bits_per_word, device=codes.device, dtype=torch.int64) * 4
    return torch.sum(
        codes.view(*codes.shape[:-1], -1, n_bits_per_word) << shifts, dim=-1
    ).to(torch.int32)


def pack_zp_along_n(zp, G):
    """zp [K//G, N] -> [N//8, K//G] int32, nibble j holds output n = 8*n8 + j."""
    shifts = torch.arange(8, device=zp.device, dtype=torch.int64) * 4
    z = zp.T.contiguous().view(-1, 8, zp.shape[0])          # [N//8, 8, K//G]
    return torch.sum(z << shifts.view(1, 8, 1), dim=1).to(torch.int32)


def build_case(K, N, G, M=4, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int64).to(DEV)
    zp = torch.randint(0, 16, (K // G, N), generator=g, dtype=torch.int8).to(DEV)
    scale = (
        torch.rand(K // G, N, generator=g, dtype=torch.float32) * 0.02 + 0.003
    ).half().to(DEV)

    zp_full = zp.float().T.repeat_interleave(G, dim=1)      # [N, K]
    scale_full = scale.float().T.repeat_interleave(G, dim=1)  # [N, K]
    ref_w = (codes.float() - zp_full) * scale_full           # [N, K]
    x = torch.randn(M, K, generator=g, dtype=torch.float32).half().to(DEV)
    ref = (x.float() @ ref_w.float().T).half()

    w_packed = pack_along_last(codes)                    # [N, K//8]
    w_scale = scale.T.contiguous()                       # [N, K//G]
    w_zp = pack_zp_along_n(zp, G)                        # [N//8, K//G]

    layer = torch.nn.Module()
    wl = lambda *a, **k: None  # noqa: E731  (never invoked in this test)
    layer.weight_packed = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=wl,
        packed_factor=8, packed_dim=1, data=w_packed,
    )
    layer.weight_scale = GroupQuantScaleParameter(
        output_dim=0, input_dim=1, weight_loader=wl, data=w_scale,
    )
    layer.weight_zero_point = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=wl,
        packed_factor=8, packed_dim=0, data=w_zp,
    )
    return layer, x, ref


def run(name, K, N, G=128, M=4):
    layer, x, ref = build_case(K, N, G, M=M)
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
        has_g_idx=False,
    )
    ok, why = WNA16ExllamaKernel.can_implement(cfg)
    if not ok:
        print("%-34s can_implement=False (%s)" % (name, why))
        return
    k = WNA16ExllamaKernel(
        cfg, "weight_packed", "weight_scale", "weight_zero_point", None
    )
    k.process_weights_after_loading(layer)
    shapes = {
        n: tuple(getattr(layer, n).shape)
        for n in ("weight_packed", "weight_scale", "weight_zero_point", "g_idx")
        if hasattr(layer, n)
    }
    out = k.apply_weights(layer, x)
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp(min=1e-3)
    print(
        "%-34s maxabs=%.4f maxrel=%.3f mismatch=%.3f%%  %s"
        % (
            name,
            diff.max().item(),
            (diff / denom).max().item(),
            100.0 * (diff > 0.05 * denom).float().mean().item(),
            shapes,
        )
    )


def init_parallel():
    import vllm.model_executor.parameter as pmod

    pmod.get_tensor_model_parallel_rank = lambda: 0
    pmod.get_tensor_model_parallel_world_size = lambda: 1


if __name__ == "__main__":
    init_parallel()
    torch.manual_seed(0)
    print("gptq_gemm present:", hasattr(torch.ops._C, "gptq_gemm"))
    run("TP1  K=1024 N=1024", K=1024, N=1024)
    run("col-parallel TP2 (N/2)   ", K=1024, N=512)
    run("row-parallel TP2 (K/2)   ", K=512, N=1024)
    run("row-parallel TP2 K/2 G128*", K=512, N=1024, G=128)
    run("TP1 larger K=5120 N=5120 ", K=5120, N=5120)
    run("col TP2 K=5120 N=2560    ", K=5120, N=2560)
    run("row TP2 K=2560 N=5120    ", K=2560, N=5120)
    run("M=1 decode TP1           ", K=1024, N=1024, M=1)
    run("M=1 decode row TP2       ", K=512, N=1024, M=1)
