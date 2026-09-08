# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""compressed-tensors WNA16 -> ExLlama GPTQ kernel bridge for gfx906.

On gfx906 (MI50, no MFMA) the upstream ExLlama Q4 GEMM/HIP kernel
(`ops.gptq_gemm(use_exllama=True)`) is ~8-17x faster than
`TritonW4A16LinearKernel` at M=1 decode shapes (measured: 0.077-0.163ms
vs ~1.32ms average per projection; 70% of decode step time was W4A16 GEMM).

This kernel repacks compressed-tensors pack-quantized parameters into the
GPTQ checkpoint layout expected by `ops.gptq_gemm`:

  weight_packed     [N, K//8]   int32, packed along K, LSB-first  (checkpoint)
  -> qweight        [K//8, N]   int32, packed along K, LSB-first  (GPTQ)

  weight_scale      [N, K//G]   fp16
  -> scales         [K//G, N]   fp16

  weight_zero_point [N//8, K//G] int32, packed along N (asymmetric)
  -> qzeros         [K//G, N//8] int32, packed along N

Zero points use GPTQv2 encoding (`use_v2_format=True`: effective zero ==
stored nibble), which natively matches AWQ-style zero points in [0, 15]
and avoids the GPTQv1 "+1" checkpoint bug that cannot represent zp == 0.

Enable with env `VLLM_WNA16_EXLLAMA=1`; otherwise can_implement() returns
False and kernel selection falls through unchanged (default-safe).
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearLayerConfig
from .exllama import ExllamaLinearKernel


class WNA16ExllamaKernel(ExllamaLinearKernel):
    """ExLlama GPTQ kernel fed by repacked compressed-tensors WNA16 params."""

    # Asymmetric WNA16 uses `uint4` (true zero points), symmetric uses
    # `uint4b8` (implicit bias of 8). The parent only allows the biased
    # variants; we accept both and handle the zero encoding ourselves.
    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.uint8,
        scalar_types.uint8b128,
    ]

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if os.environ.get("VLLM_WNA16_EXLLAMA", "0") != "1":
            return False, "disabled (set VLLM_WNA16_EXLLAMA=1 to enable)"

        if c.weight_type.size_bits not in (4, 8):
            return False, "only 4/8-bit WNA16 is supported"

        if c.has_g_idx:
            return False, "actorder (g_idx) is not supported yet"

        return super().can_implement(c)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        K_dim, N_dim = c.partition_weight_shape
        group_size = c.group_size if c.group_size != -1 else K_dim
        device = getattr(layer, self.w_q_name).device

        # ---- w_q: [N, K//8] (K packed, LSB-first) -> [K//8, N] (GPTQ) ----
        def transform_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32
            N_dim_, K8 = w.shape
            K_dim_ = K8 * 8
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            codes = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim_, K_dim_)
            codes = codes.t().contiguous()  # [K, N] raw 4-bit codes
            qweight = torch.sum(
                (codes.view(K_dim_ // 8, 8, N_dim_) << shifts.view(1, 8, 1)),
                dim=1,
                dtype=torch.int32,
            ).contiguous()
            x.data = qweight  # [K//8, N] int32, GPTQ layout
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)

        # ---- w_s: [N, K//G] -> [K//G, N] fp16 ----
        def transform_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous().to(dtype=c.act_type)
            return x

        self._transform_param(layer, self.w_s_name, transform_w_s)

        # ---- zeros: GPTQv2 encoding (stored nibble == true zero point) ----
        groups = K_dim // group_size
        if not c.zero_points:
            # Symmetric: the implicit bias (8 for uint4b8) is the zero point.
            bias = c.weight_type.bias
            zeros = torch.full(
                (groups, N_dim), bias, dtype=torch.int32, device=device
            )
            shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
            qzeros = torch.sum(
                (zeros.view(groups, N_dim // 8, 8) << shifts.view(1, 1, 8)),
                dim=2,
                dtype=torch.int32,
            ).contiguous()
            self.w_zp_name = "qzeros"
            layer.register_parameter(
                "qzeros",
                torch.nn.Parameter(qzeros, requires_grad=False),
            )
        else:
            # Asymmetric: weight_zero_point [N//8, K//G] int32, N packed.
            # Nibble j of int32 (n8, g) is the zero point of output column
            # n = 8*n8 + j. Unpack carefully: [N//8, K//G, 8] must be
            # permuted to [N//8, 8, K//G] BEFORE flattening to [N, K//G],
            # otherwise the group and nibble dimensions get mixed up.
            zp = getattr(layer, self.w_zp_name)
            shifts = torch.arange(8, device=zp.device, dtype=torch.int32) * 4
            z = (zp.data.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
            z = z.permute(0, 2, 1).reshape(N_dim, groups)  # [N, K//G]
            z = z.t().contiguous()  # [K//G, N]
            qzeros = torch.sum(
                (z.view(groups, N_dim // 8, 8) << shifts.view(1, 1, 8)),
                dim=2,
                dtype=torch.int32,
            ).contiguous()
            self._transform_param(
                layer, self.w_zp_name,
                lambda x: _set_data(x, qzeros),
            )

        # ---- g_idx: group index per K row (value used directly as the
        # scales/zeros row by the reconstruct kernels; the decode GEMM
        # kernel ignores it and assumes sequential groups, which matches) ----
        self.w_gidx_name = "g_idx"
        layer.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                (torch.arange(K_dim, device=device, dtype=torch.int32)
                 // group_size).contiguous(),
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        c = self.config
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        # Ensure all weight params are contiguous before passing to kernel
        if not w_q.is_contiguous():
            w_q = w_q.contiguous()
        if not w_zp.is_contiguous():
            w_zp = w_zp.contiguous()
        if not w_s.is_contiguous():
            w_s = w_s.contiguous()
        if not w_g_idx.is_contiguous():
            w_g_idx = w_g_idx.contiguous()

        # GPTQv2 zero-point encoding: stored nibble == true zero point.
        # This is what makes AWQ-style zero points (which include 0) exact.
        output = ops.gptq_gemm(
            x_2d, w_q, w_zp, w_s, w_g_idx, True, True, c.weight_type.size_bits
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


def _set_data(x: BasevLLMParameter, data: torch.Tensor) -> BasevLLMParameter:
    x.data = data
    return x
