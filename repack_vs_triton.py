"""Compare WNA16ExllamaKernel against TritonW4A16LinearKernel on identical params.

TritonW4A16 is the proven-good path (27B TP2 with it emits coherent, deterministic
text). Feeding both kernels the same canonical compressed-tensors pack-quantized
params makes any ExLlama repack bug show up without ambiguity about my own
dequant convention.
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
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    TritonW4A16LinearKernel,
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


def make_params(codes, zp, scale, w_packed, w_scale, w_zp):
    layer = torch.nn.Module()
    layer.weight_packed = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=_WL,
        packed_factor=8, packed_dim=1, data=w_packed.clone(),
    )
    layer.weight_scale = GroupQuantScaleParameter(
        output_dim=0, input_dim=1, weight_loader=_WL, data=w_scale.clone(),
    )
    layer.weight_zero_point = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=_WL,
        packed_factor=8, packed_dim=0, data=w_zp.clone(),
    )
    return layer


def build_case(K, N, G, M, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int64).to(DEV)
    zp = torch.randint(0, 16, (K // G, N), generator=g, dtype=torch.int8).to(DEV)
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
    return x, ref, (codes, zp, scale, w_packed, w_scale, w_zp)


def cfg_for(K, N, G):
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=G,
        zero_points=True,
        has_g_idx=False,
    )


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    rel = d / b.float().abs().clamp(min=1e-3)
    return d.max().item(), rel.max().item(), 100.0 * (d > 0.05).float().mean().item()


def run(name, K, N, G=128, M=4):
    x, ref, params = build_case(K, N, G, M)
    c = cfg_for(K, N, G)

    lt = make_params(*params)
    kt = TritonW4A16LinearKernel(c, "weight_packed", "weight_scale", "weight_zero_point", None)
    kt.process_weights_after_loading(lt)
    y_tri = kt.apply_weights(lt, x).reshape(M, N)

    le = make_params(*params)
    ke = WNA16ExllamaKernel(c, "weight_packed", "weight_scale", "weight_zero_point", None)
    ke.process_weights_after_loading(le)
    y_ex = ke.apply_weights(le, x).reshape(M, N)

    et = cmp(y_tri, ref)
    ee = cmp(y_ex, ref)
    eX = cmp(y_ex, y_tri)
    print("%-26s tri_vs_ref=(%.3f,%.2f) ex_vs_ref=(%.3f,%.2f) ex_vs_tri=(%.3f,%.2f)"
          % (name, et[0], et[1], ee[0], ee[1], eX[0], eX[1]))
    if eX[0] > 0.05:
        t = y_tri[0, :8].tolist()
        e = y_ex[0, :8].tolist()
        r = ref[0, :8].tolist()
        print("      row0 tri=%s" % ["%.3f" % v for v in t])
        print("      row0 ex =%s" % ["%.3f" % v for v in e])
        print("      row0 ref=%s" % ["%.3f" % v for v in r])


if __name__ == "__main__":
    torch.manual_seed(0)
    run("TP1 K=1024 N=1024", K=1024, N=1024)
    run("col-parallel N/2", K=1024, N=512)
    run("row-parallel K/2", K=512, N=1024)
    run("TP1 K=5120 N=5120", K=5120, N=5120)
    run("M=1 decode TP1", K=1024, N=1024, M=1)
    run("G=64 TP1", K=1024, N=1024, G=64)
