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
  -> qweight        [K//8, N]   int32, packed along K, nibble-interleaved

  weight_scale      [N, K//G]   fp16
  -> scales         [K//G, N]   fp16

  weight_zero_point [N//8, K//G] int32, packed along N (asymmetric)
  -> qzeros         [K//G, N//8] int32, packed along N, LSB-first

The qweight nibble interleave is the in-kernel equivalent of
`ops.gptq_shuffle` (csrc .../gptq/qdq_4.cuh::shuffle_4bit_8): the ExLlama
dequant reads consecutive k values from bits (i//2)*4 + (i%2)*16, not from
the sequential 4*i position a GPTQ checkpoint stores them in.

Zero points use GPTQv2 encoding (`use_v2_format=True`: effective zero ==
stored nibble), which natively matches AWQ-style zero points in [0, 15]
and avoids the GPTQv1 "+1" checkpoint bug that cannot represent zp == 0.

`b_g_idx` is consumed by the kernels as an act-order *permutation* (row
remap table), never as a group index, so for checkpoints without act-order
it must be an empty tensor (upstream ExllamaLinearKernel does the same).

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

# P0 attribution hook: records the (M, K, N) the kernel actually receives.
# The hot path may only do pure-python int/list work -- see apply_weights.
_SHAPE_LOG = os.environ.get("VLLM_WNA16_EXL_SHAPEDBG")
_shape_seen: list[tuple[int, int, int]] = []


def _dump_shapes(*_a):
    if not _SHAPE_LOG:
        return
    import csv
    # one file per rank: with TP2 each worker holds its own list, and sharing a
    # path would both race and silently lose a rank.
    path = "%s.%d" % (_SHAPE_LOG, os.getpid())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "k", "n"])
        w.writerows(_shape_seen)


if _SHAPE_LOG:
    import atexit
    import signal

    # atexit alone is not enough: vLLM hard-kills the TP workers on shutdown, so
    # only the APIServer's (empty) file ever appeared. SIGUSR1 dumps on demand,
    # and a signal handler runs outside the dynamo-traced region.
    def _on_sigusr1(*_a):
        _dump_shapes()

    try:
        if signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL:
            signal.signal(signal.SIGUSR1, _on_sigusr1)
    except (ValueError, OSError):
        pass
    atexit.register(_dump_shapes)


def _unpack_codes(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack LSB-first int32 codes -> int64 [.., 32//bits] along last dim."""
    shifts = torch.arange(32 // bits, device=w.device, dtype=torch.int64) * bits
    return (w.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)


def _pack_gptq_sequential(codes: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """Pack along `pack_dim` LSB-first (plain GPTQ/checkpoint encoding)."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=codes.device, dtype=torch.int64) * bits
    view_shape = list(codes.shape)
    view_shape[pack_dim] = codes.shape[pack_dim] // per_word
    view_shape.insert(pack_dim + 1, per_word)
    grouped = codes.view(view_shape).to(torch.int64)
    bcast = [1] * grouped.ndim
    bcast[pack_dim + 1] = -1
    packed = (grouped << shifts.view(bcast)).sum(dim=pack_dim + 1, dtype=torch.int64)
    return packed.to(torch.int32).contiguous()


_TDQ_POOL: dict[torch.device, torch.Tensor] = {}


def _get_tempdq(nelem: int, dtype, device) -> torch.Tensor:
    """Grow-only shared dequant scratch. Only the eager reconstruct path
    (large-M prefill) ever touches it, so first-touch can never happen
    inside a graph capture."""
    buf = _TDQ_POOL.get(device)
    if buf is None or buf.numel() < nelem or buf.dtype != dtype:
        buf = torch.empty(nelem, dtype=dtype, device=device)
        _TDQ_POOL[device] = buf
    return buf[:nelem]


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
        bits = c.weight_type.size_bits
        group_size = c.group_size if c.group_size != -1 else K_dim
        device = getattr(layer, self.w_q_name).device

        # ---- g_idx: ExLlama wants a *permutation*; no act-order => empty ----
        # Must be registered before transform_w_q in case a future variant
        # needs it to build a shuffled layout.
        self.w_gidx_name = "g_idx"
        layer.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int32, device=device),
                requires_grad=False,
            ),
        )

        # ---- w_q: [N, K//8] LSB-first -> [K//8, N] ExLlama nibble order ----
        def transform_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32
            N_dim_ = w.shape[0]
            codes = _unpack_codes(w, bits).reshape(N_dim_, -1)  # [N, K]
            codes = codes.t().contiguous()  # [K, N]

            per_word = 32 // bits
            idx = torch.arange(per_word, device=device, dtype=torch.int64)
            # Mirrors shuffle_4bit_8 / shuffle_8bit_4: even-indexed codes fill
            # the low half of the word, odd-indexed codes the high half.
            dst_bits = (idx // 2) * bits + (idx % 2) * 16
            grouped = codes.view(K_dim // per_word, per_word, N_dim_)
            packed = (
                grouped.to(torch.int64) << dst_bits.view(1, -1, 1)
            ).sum(dim=1, dtype=torch.int64)
            x.data = packed.to(torch.int32).contiguous()  # [K//pack, N]
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
            qzeros = _pack_gptq_sequential(zeros, bits, pack_dim=1)
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
            z = _unpack_codes(zp.data, bits)  # [N//8, K//G, per_word]
            z = z.permute(0, 2, 1).reshape(N_dim, groups)  # [N, K//G]
            z = z.t().contiguous()  # [K//G, N]
            # Zero points are NOT shuffled by gptq_shuffle -- the kernels
            # unpack them with the plain LSB-first MatrixView_q4_row.
            qzeros = _pack_gptq_sequential(z, bits, pack_dim=1)
            self._transform_param(
                layer, self.w_zp_name,
                lambda x: _set_data(x, qzeros),
            )

        # ---- capture-safe buffers (must exist before any graph capture) ----
        # ExLlama fused kernels run at M <= 50 (4-bit) / 24 (8-bit); size the
        # persistent output slab above that and keep every captured graph's
        # writes in this fixed buffer.
        self._mcap = int(os.environ.get("VLLM_WNA16_EXL_MCAP", "64"))
        self._cbuf = torch.empty(
            (self._mcap, N_dim), dtype=c.act_type, device=device
        )
        self._tdq_empty = torch.empty(0, dtype=c.act_type, device=device)

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

        # GPTQv2 zero-point encoding: stored nibble == true zero point.
        # This is what makes AWQ-style zero points (which include 0) exact.
        #
        # Capture-safety: for M <= mcap we write into the persistent slab, so
        # the op performs no allocations inside a cudagraph capture. Guards
        # against misuse at larger M during capture live in C++ (empty
        # temp_dq + reconstruct path throws); keep
        # cudagraph_capture_sizes <= mcap and <= 48 (4-bit fused-row cap).
        bits = c.weight_type.size_bits
        K_dim, N_dim = c.partition_weight_shape
        M = x_2d.shape[0]
        if _SHAPE_LOG:
            # ONLY a pure-python append here. apply_weights sits inside the
            # dynamo-traced region, and any torch call that returns a non-Tensor
            # (e.g. is_current_stream_capturing) kills engine init with
            # "torch.* op returned non-Tensor". Whether a row came from graph
            # capture is inferred from the server log, not queried here.
            _shape_seen.append((int(M), int(K_dim), int(N_dim)))
        if M <= self._mcap:
            c_out = self._cbuf[:M]
        else:
            c_out = torch.empty(
                (M, N_dim), dtype=c.act_type, device=x_2d.device)
        needs_tdq = M > (24 if bits == 8 else 50)
        if needs_tdq:
            # Only happens on eager paths (M > fused cap); allocating fresh
            # per call avoids the shared-scratch race with async kernels.
            temp_dq = torch.empty((K_dim, N_dim), dtype=c.act_type,
                                  device=x_2d.device)
        else:
            temp_dq = self._tdq_empty
        if os.environ.get("VLLM_WNA16_EXL_DEBUG") == "legacy":
            output = ops.gptq_gemm(
                x_2d, w_q, w_zp, w_s, w_g_idx, True, True, bits)
        else:
            # The fused kernel accumulates into c via atomicAdd; upstream
            # always passes fresh zeros. zero_() on our persistent slab is a
            # capture-legal fixed-address write.
            c_out.zero_()
            ops.gptq_gemm_out(
                c_out, x_2d, w_q, w_zp, w_s, w_g_idx, True, True, bits,
                temp_dq)
            output = c_out
            dbg = os.environ.get("VLLM_WNA16_EXL_DEBUG")
            self._dbg_n = getattr(self, "_dbg_n", 0)
            if dbg == "last" and self._dbg_n < 3:
                self._dbg_n += 1
                nz = output.float().abs().sum().item()
                print("[GDNDBG] K=%d M=%d N=%d abs_sum=%.4e finite=%s"
                      % (K_dim, M, N_dim, nz,
                         bool(torch.isfinite(output).all().item())),
                      flush=True)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


def _set_data(x: BasevLLMParameter, data: torch.Tensor) -> BasevLLMParameter:
    x.data = data
    return x
