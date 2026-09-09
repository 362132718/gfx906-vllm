# p1_l0_mapping.py -- L0: pin the W4A16 bit-level mapping down.
#   L0a: the bridge's repack matches the documented shuffle_4bit_8 rule exactly
#        (pure python, zero tolerance -- layout questions must not be numeric).
#   L0b: the SHIPPED kernel, fed with a one-hot x = e_k, reproduces the analytic
#        dequant value for every (k, n) => the mapping is right where it counts.
# Nothing here touches site-packages; this only imports and calls.
import os
os.environ["VLLM_WNA16_EXLLAMA"] = "1"      # can_implement() requires it
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig)
from vllm.model_executor.kernels.linear.mixed_precision.wna16_exllama import (
    WNA16ExllamaKernel)
from vllm.scalar_type import scalar_types

DEV = "cuda"
G = 128
BITS = 4


def codes_for(K, N):
    """Codes that make every (k, n) position distinguishable and every single
    change of code distinguishable by value: 0..15, structured, not random.
    Random data would let an off-by-one in the interleave hide behind collisions.
    """
    k = torch.arange(K, device=DEV, dtype=torch.int32).view(-1, 1)
    n = torch.arange(N, device=DEV, dtype=torch.int32).view(1, -1)
    return ((k * 7 + n * 13) % 16).to(torch.int8)     # [K, N]


def pack_checkpoint_format(codes_KN):
    """compressed-tensors pack-quantized: [N, K//8] int32, along K, LSB-first."""
    K, N = codes_KN.shape
    c = codes_KN.t().contiguous().to(torch.int32)          # [N, K]
    w = c.view(N, K // 8, 8)
    bits = torch.arange(8, device=c.device, dtype=torch.int32) * BITS
    return ((w << bits.view(1, 1, -1)) & 0xFFFFFFFF).sum(dim=2, dtype=torch.int64) \
        .to(torch.int32).contiguous()                       # [N, K//8]


def expected_qweight(codes_KN, dst_bits):
    """Reference: what qweight must be, computed independently of the bridge."""
    K, N = codes_KN.shape
    grouped = codes_KN.to(torch.int64).view(K // 8, 8, N)
    return (grouped << dst_bits.view(1, -1, 1)).sum(dim=1, dtype=torch.int64) \
        .to(torch.int32).contiguous()                        # [K//8, N]


def pack_zp_checkpoint_format(zp_KG_N):
    """[K//G, N] int32 (0..15) -> [N//8, K//G] int32 packed along N; nibble j of
    word (n8, g) is column 8*n8 + j (bridge :212-223).

    Deliberately NOT the bridge's own `_pack_gptq_sequential` -- L0 has to
    cross-check the bridge, and sharing its packer makes the test circular.
    """
    Kg, N = zp_KG_N.shape
    z = zp_KG_N.t().contiguous().view(N // 8, 8, Kg).to(torch.int64)
    bits = torch.arange(8, device=zp_KG_N.device, dtype=torch.int64) * 4
    return (z << bits.view(1, -1, 1)).sum(dim=1, dtype=torch.int64) \
        .to(torch.int32).contiguous()                        # [N//8, K//G]


def _harness_param(data, input_dim, output_dim, packed_dim=None):
    """Minimal parameter carrying the layout attributes permute_param_layout_
    reads (input_dim / output_dim / packed_dim).

    BasevLLMParameter is NOT usable in a harness: its __init__ calls
    get_tensor_model_parallel_rank() and asserts that vLLM's distributed state
    exists (reproduced: "tensor model parallel group is not initialized"). The
    bridge path itself only needs a torch.nn.Parameter exposing those attributes;
    weight_loader is a weight-LOADING concern we never exercise.
    """

    class _HarnessParam(torch.nn.Parameter):
        def __new__(cls, t):
            return super().__new__(cls, t, requires_grad=False)

        def __init__(self, t):
            self._in, self._out, self._pk = input_dim, output_dim, packed_dim

        @property
        def input_dim(self):
            return self._in

        @property
        def output_dim(self):
            return self._out

        @property
        def packed_dim(self):
            return self._pk

    return _HarnessParam(data)


def build_layer(K, N, codes_KN, scales_KG_N, zp_KG_N):
    """A bare module holding the three params in their CHECKPOINT forms, wrapped
    in harness params that carry the layout attributes production provides:
    weight_packed [N, K//8] (packed along dim 1), weight_scale [N, K//G],
    weight_zero_point [N//8, K//G]. Feeding zp as [N, K//G] instead makes the
    bridge die inside its own permute/reshape (reproduced).
    """
    from torch import nn

    class L(nn.Module):
        pass

    layer = L()
    for name, t, idim, pk in (
            ("weight_packed", pack_checkpoint_format(codes_KN), 1, 1),
            ("weight_scale", scales_KG_N.t().contiguous(), 1, None),          # [N, K//G]
            ("weight_zero_point", pack_zp_checkpoint_format(zp_KG_N), 1, None)):
        layer.register_parameter(name, _harness_param(t, input_dim=idim,
                                                      output_dim=0, packed_dim=pk))
    return layer


def kernel_for(K, N):
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4, act_type=torch.float16,
        group_size=G, zero_points=True, has_g_idx=False)
    # w_zp param name is the checkpoint's; the bridge repacks in place.
    return WNA16ExllamaKernel(cfg, "weight_packed", "weight_scale",
                              "weight_zero_point")


def make_layer_and_kernel(K, N, seed=0):
    torch.manual_seed(seed)
    codes = codes_for(K, N)
    scales = (torch.rand(K // G, N, device=DEV, dtype=torch.float16) * 0.02 + 0.001)
    zp = torch.randint(0, 16, (K // G, N), device=DEV, dtype=torch.int32)
    kern = kernel_for(K, N)
    layer = build_layer(K, N, codes, scales, zp)
    kern.process_weights_after_loading(layer)
    return kern, layer, codes, scales, zp


def test_L0a(K=512, N=256):
    """Documented rule: k index i lands at bits (i//2)*4 + (i%2)*16."""
    i = torch.arange(8, device=DEV, dtype=torch.int64)
    dst_bits = (i // 2) * BITS + (i % 2) * 16
    kern, layer, codes, *_ = make_layer_and_kernel(K, N)
    want = expected_qweight(codes, dst_bits.to(DEV))
    got = layer.weight_packed.data
    assert got.shape == want.shape, "qweight shape %s want %s" % (tuple(got.shape), tuple(want.shape))
    diff = (got.to(torch.int64) != want.to(torch.int64))
    print("L0a K=%d N=%d words=%d mismatched_words=%d" % (K, N, got.numel(), int(diff.sum())))
    assert int(diff.sum()) == 0, "bridge layout != documented shuffle_4bit_8"
    # scales/zp must end up [K/G, N] contiguous
    assert tuple(layer.weight_scale.shape) == (K // G, N), layer.weight_scale.shape
    print("L0a PASS (scales %s dtype %s)" % (tuple(layer.weight_scale.shape),
                                             layer.weight_scale.dtype))
    return dst_bits, codes, layer, K, N


def dequant_analytic(code_int8, scale_f16, zp_int32):
    """The reference every variant must match: (q - zp) * scale, fp32 math."""
    return (code_int8.to(torch.float32) - zp_int32.to(torch.float32)) \
        * scale_f16.to(torch.float32)


def test_L0b(K=512, N=256):
    """x = e_k makes out[:, n] the dequantized weight at that single (k, n).
    That turns 'is the nibble mapping right' into a zero-tolerance structural
    readout instead of an error-budget argument."""
    dst_bits, codes, layer, K, N = test_L0a(K, N)
    w_q = layer.weight_packed.data.contiguous()
    w_s = layer.weight_scale.data.contiguous()
    # After the transform this IS the packed qzeros [K//G, N//8] int32 -- that is
    # the form gptq_gemm_out wants (torch_bindings schema; the kernel indexes
    # N/8 words and takes nibble n%8). Unpack it ONLY to build the reference.
    z_packed = layer.weight_zero_point.data.contiguous()
    assert z_packed.shape == (K // G, N // 8) and z_packed.dtype == torch.int32, \
        "unexpected qzeros form %s %s" % (tuple(z_packed.shape), z_packed.dtype)
    shift = torch.arange(32 // BITS, device=DEV, dtype=torch.int64) * BITS
    u = ((z_packed.to(torch.int64).unsqueeze(-1) >> shift.view(1, 1, -1)) & 0xF)
    u = u.reshape(K // G, N).to(torch.int32)               # [K/G, N], reference only
    print("L0b qzeros packed=%s unpacked=%s" % (tuple(z_packed.shape), tuple(u.shape)))

    x = torch.zeros((1, K), device=DEV, dtype=torch.float16)
    c_out = torch.zeros((1, N), device=DEV, dtype=torch.float16)
    # M=1 takes the fused branch, where tdq is nullptr and production passes an
    # empty tensor -- mirror that; a [K, N] fp16 buffer would be 178 MB for nothing.
    temp_dq = torch.empty(0, dtype=torch.float16, device=DEV)
    g_idx = layer.g_idx
    max_rel, worst = 0.0, None
    for k in range(0, K, max(1, K // 32)):               # 32 spot positions
        x.zero_(); x[0, k] = 1.0; c_out.zero_()
        ops.gptq_gemm_out(c_out, x, w_q, z_packed, w_s, g_idx,
                          True, True, BITS, temp_dq)
        g = k // G
        want = dequant_analytic(codes[k], w_s[g], u[g])
        err = (c_out[0].float() - want).abs()
        rel = float((err / want.abs().clamp(min=1e-6)).max())
        if rel > max_rel:
            max_rel, worst = rel, (k, int(err.argmax()))
    print("L0b positions=%d max_rel=%.3e worst=%s" % (len(range(0, K, max(1, K // 32))),
                                                      max_rel, worst))
    # One-hot removes cancellation entirely: anything above fp16 representable
    # error means the kernel read a different (k, n) than we addressed.
    assert max_rel < 1e-3, "kernel dequant does not match the mapping (rel %g)" % max_rel
    print("L0b PASS")


SHAPES = [(5120, 7168), (3072, 5120), (5120, 17408), (8704, 5120), (5120, 8192)]


def main():
    test_L0a()
    test_L0b()
    for K, N in SHAPES:                       # full size: proves the K/G group walk
        test_L0b(K, N)
    print("L0 ALL PASS")


if __name__ == "__main__":
    main()
