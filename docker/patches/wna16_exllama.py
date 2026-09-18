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
from vllm.utils.torch_utils import direct_register_custom_op

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


# ---- P2: gfx906 M=1 GEMV for whitelisted shapes (P1 report v3) ----
# Shapes are POST-partition (K_dim, N_dim): TP2 sharded widths (same table that
# has 8704 = 17408/2 and 3072 = 6144/2). A wrong assumption here fails SILENTLY
# -- zero hits, zero error, zero gain -- so Task 2 asserts them against the
# engine shape log.
_P2_SHAPES = frozenset({(5120, 8192), (5120, 7168)})
# ONE source of truth, on the Python side: the same constant sizes the `part`
# buffer AND is passed to the kernel, so the two can never disagree (a splitk
# larger than part's row count = out-of-bounds write the .so cannot detect -- it
# only sees a raw pointer). 10 is Task 0's measured optimum (60.51 us at 8192 vs
# 68.56 at the design's 8). Constraint the .hip DOES enforce: splitk must divide
# K/GROUP = 40 -- 16 does not, which is why the P1/P2 "powers of two" grid missed
# the optimum entirely.
_P2_SPLITK = 10
# GROUP is compiled into the .so. It is also the ONLY layout fact the kernel
# cannot check: p2_gemv validates K%128 and N%64, then reads [K/128, N] of
# scales and [K/8, N] of qweight. A per-channel layer (group_size == -1) at a
# whitelisted shape has ONE scales row, so the warm-up call would read 39 rows
# past the end -- the .so only sees a raw pointer and cannot tell. Hence
# _p2_layer_ok below gates on group_size as well as on bits/act_type.
_P2_GROUP = 128
_P2_LIB = None
_P2_OK = False
try:
    import ctypes as _ctypes
    _P2_LIB = _ctypes.CDLL(
        "/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so")
    _P2_LIB.p2_gemv.argtypes = ([_ctypes.c_void_p] * 6
                                + [_ctypes.c_int] * 3 + [_ctypes.c_void_p])
    _P2_LIB.p2_gemv.restype = _ctypes.c_int
    # CDLL binds lazily: a successful load does NOT mean the symbol exists.
    # Without this probe the AttributeError surfaces on the first call --
    # inside the engine, during capture (P2 review P1).
    _P2_OK = callable(getattr(_P2_LIB, "p2_gemv", None))
except (OSError, AttributeError):
    _P2_OK = False       # default-safe: dispatch falls through to gptq_gemm_out
# Pure-python hit log: proves P2 actually fired, per shape. Appended INSIDE the
# op impl, so a hit is a real dispatch at execution time -- not a trace-time
# branch visit, which is what made G-P3's dispatch invisible from Python.
# Same pattern as _shape_seen -- list append only, dynamo-safe.
_P2_HITS: list = []


def _p2_layer_ok(c: MPLinearLayerConfig, K_dim: int, N_dim: int) -> bool:
    """Eligibility for BOTH the split-K scratch and the routing branch.

    Two predicates would be a correctness hazard: allocate-but-not-route wastes
    memory, route-but-not-allocate crashes on a None buffer, and a group_size
    mismatch is silently WRONG, not loud. One function, so they cannot drift.
    """
    return (_P2_OK
            and (int(K_dim), int(N_dim)) in _P2_SHAPES
            and c.weight_type.size_bits == 4
            and not c.has_g_idx
            and c.group_size == _P2_GROUP
            and c.act_type is torch.float16)


def _p2_w4a16_gemm_impl(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                        w_zp: torch.Tensor, part: torch.Tensor,
                        out: torch.Tensor) -> torch.Tensor:
    rc = _P2_LIB.p2_gemv(
        x.data_ptr(), w_q.data_ptr(), w_s.data_ptr(), w_zp.data_ptr(),
        part.data_ptr(), out.data_ptr(),
        int(x.shape[1]), int(out.shape[1]), _P2_SPLITK,
        torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"p2_gemv rc={rc}")   # loud beats silently-wrong out
    return out


def _p2_w4a16_gemm_fake(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                        w_zp: torch.Tensor, part: torch.Tensor,
                        out: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


if _P2_OK and not hasattr(torch.ops.vllm, "p2_w4a16_gemm"):
    direct_register_custom_op(
        op_name="p2_w4a16_gemm",
        op_func=_p2_w4a16_gemm_impl,
        mutates_args=["part", "out"],
        fake_impl=_p2_w4a16_gemm_fake,
    )


def _p2_gemv_or_fused_impl(x: torch.Tensor, w_q: torch.Tensor,
                           w_s: torch.Tensor, w_zp: torch.Tensor,
                           w_g_idx: torch.Tensor, part: torch.Tensor,
                           out: torch.Tensor, temp_dq: torch.Tensor,
                           bits: int) -> None:
    """M-based dispatch that SURVIVES torch.compile.

    This `if` cannot live in apply_weights: that function is inside the
    dynamo-traced region, so `M == 1` is resolved at TRACE time against the
    shape the region was first traced with (the prefill M) and the False branch
    is baked into the artifact -- every captured decode graph then replays the
    incumbent. Measured in G-P3 run 1: 0 p2_gemv launches compiled vs
    64/step/rank under enforce_eager. Dynamo does not trace inside a custom op,
    so here M is the real runtime shape, including during cudagraph capture --
    which is exactly how vLLM's own M=1 skinny-GEMV dispatch
    (rocm_unquantized_gemm_impl) stays live in the compiled engine.

    Returns nothing on purpose: `out` is also an input, and an op that returns
    an aliased input makes dynamo warn ("must not also be an input to this
    custom operator ... may not alias any inputs", seen once per rank in G-P3
    run 1) and that is slated to become an error. Declaring `out` mutated is
    what keeps the write visible -- the same shape as ops.gptq_gemm_out and
    prefetch_ops' start_prefetch.
    """
    if int(x.shape[0]) == 1 and os.environ.get("VLLM_WNA16_P2_GEMV", "1") == "1":
        # p2_reduce overwrites out[n] for every n, so unlike the fused kernel
        # (atomicAdd into c) this needs no out.zero_() first.
        _P2_HITS.append((int(x.shape[1]), int(out.shape[1])))
        _p2_w4a16_gemm_impl(x, w_q, w_s, w_zp, part, out)
        return
    out.zero_()
    ops.gptq_gemm_out(out, x, w_q, w_zp, w_s, w_g_idx, True, True, bits,
                      temp_dq)


def _p2_gemv_or_fused_fake(x: torch.Tensor, w_q: torch.Tensor,
                           w_s: torch.Tensor, w_zp: torch.Tensor,
                           w_g_idx: torch.Tensor, part: torch.Tensor,
                           out: torch.Tensor, temp_dq: torch.Tensor,
                           bits: int) -> None:
    return


if _P2_OK and not hasattr(torch.ops.vllm, "p2_gemv_or_fused"):
    direct_register_custom_op(
        op_name="p2_gemv_or_fused",
        op_func=_p2_gemv_or_fused_impl,
        mutates_args=["part", "out"],
        fake_impl=_p2_gemv_or_fused_fake,
    )


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

        # ---- P2: split-K scratch, only for the whitelisted families ----
        # Unconditional allocation would hang (splitk, N) fp32 off EVERY WNA16
        # layer -- ~91 MB/card at the model's N distribution, of which only two
        # shape families use any of it (P2 review P1). Fixed address, allocated
        # here in the capture-free zone; the warm-up call below forces the HIP
        # code object to load OUTSIDE graph capture, so a first-call lazy load
        # cannot become a capture failure (P2 review missing-2).
        self._p2_part = None
        self._p2_route = False
        if _p2_layer_ok(c, K_dim, N_dim):
            self._p2_part = torch.empty((_P2_SPLITK, N_dim),
                                         dtype=torch.float32, device=device)
            self._p2_route = True
            # (1, N) not (N,): the op reads out.shape[1] for N.
            _warm = torch.zeros(1, K_dim, dtype=torch.float16, device=device)
            _sink = torch.empty(1, N_dim, dtype=torch.float16, device=device)
            w_q, w_s, w_zp, _w_g_idx = self._get_weight_params(layer)
            _p2_w4a16_gemm_impl(_warm, w_q, w_s, w_zp, self._p2_part, _sink)
            torch.cuda.synchronize()

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
        if self._p2_route:
            # No M here on purpose: a shape test in this function is folded at
            # trace time (G-P3). The op decides, at runtime, between the P2 GEMV
            # and the fused incumbent, and writes the result into c_out.
            torch.ops.vllm.p2_gemv_or_fused(
                x_2d, w_q, w_s, w_zp, w_g_idx, self._p2_part, c_out, temp_dq,
                bits)
            output = c_out
        elif os.environ.get("VLLM_WNA16_EXL_DEBUG") == "legacy":
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
