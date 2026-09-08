#!/usr/bin/env bash
# gfx906 (Radeon Pro VII / MI50 / MI60) vLLM entry point.
#
# Every default below is a setting verified on gfx906 hardware. Override by env:
#   TP, DTYPE, MAX_LEN, GPU_MEM_UTIL, MAX_NUM_SEQS, PORT, HOST, SERVED_NAME,
#   QUANTIZATION, MM, EAGER, CUDAGRAPH_MODE, CAPTURE_SIZES, CUSTOM_ALL_REDUCE,
#   MODELS_ROOT
#
# Usage: gfx906-serve [MODEL_DIR] [extra args, forwarded to `vllm serve` verbatim]
#   MODEL_DIR may also come as $MODEL_DIR; otherwise a single model under
#   $MODELS_ROOT (default /models) is picked automatically.
set -euo pipefail

die() { printf 'serve.sh: %s\n' "$*" >&2; exit 2; }

# Backstop the gfx906 runtime env. The image sets these through Dockerfile ENV,
# but a container created from an earlier snapshot -- or a stray `-e` override --
# would otherwise silently drop W4A16 onto the Triton kernel (measured 3.9 tok/s
# vs 22 tok/s on the ExLlama bridge) or fail to load at all.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.0.6}"
export VLLM_TRITON_ATTN_NUM_SEGMENTS="${VLLM_TRITON_ATTN_NUM_SEGMENTS:-64}"
export VLLM_WNA16_EXLLAMA="${VLLM_WNA16_EXLLAMA:-1}"

# vLLM's compile-cache key covers the model/parallel/graph config but not the
# W4A16 kernel choice, while the two kernels repack weights into different
# layouts that get baked into the cached graphs as static inputs. Loading the
# other mode's cache dies at profile time with "expected size 640==5120,
# stride 8192==1024", so give each selection its own cache root.
if [ "$VLLM_WNA16_EXLLAMA" = 1 ]; then W4A16=exllama; else W4A16=triton; fi
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME:-/root}/.cache/vllm}-w4a16-${W4A16}"

MODEL="${1:-${MODEL_DIR:-}}"
if [ $# -gt 0 ]; then shift; fi

if [ -z "$MODEL" ]; then
  ROOT="${MODELS_ROOT:-/models}"
  if [ -f "$ROOT/config.json" ]; then
    MODEL="$ROOT"
  else
    cands=()
    for d in "$ROOT"/*/; do [ -f "${d}config.json" ] && cands+=("${d%/}"); done
    if [ "${#cands[@]}" -eq 1 ]; then
      MODEL="${cands[0]}"
    elif [ "${#cands[@]}" -eq 0 ]; then
      die "no model under $ROOT. Mount weights read-only (-v /path/to/models:$ROOT:ro) or pass a path: gfx906-serve $ROOT/<model>"
    else
      die "several models under $ROOT, choose one: ${cands[*]}"
    fi
  fi
fi
CFG="$MODEL/config.json"
[ -f "$CFG" ] || die "$MODEL has no config.json (weights are not shipped inside this image)"

GPUS=$(ls -d /dev/dri/renderD* 2>/dev/null | wc -l)
if [ "$GPUS" -lt 1 ]; then
  die "no GPU render node in /dev/dri. Run the container with --device=/dev/kfd --device=/dev/dri --group-add video (see QUICKSTART.md)"
fi
TP="${TP:-$GPUS}"

ARGS=(
  "$MODEL"
  --served-model-name "${SERVED_NAME:-$(basename "$MODEL")}"
  --dtype "${DTYPE:-float16}"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
  --max-model-len "${MAX_LEN:-16384}"
  --max-num-seqs "${MAX_NUM_SEQS:-48}"
  --attention-backend TRITON_ATTN
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
)

# Quantization is read from the model's config.json by default. Set
# QUANTIZATION only if auto-detection is not what you want.
if [ -n "${QUANTIZATION:-}" ]; then
  ARGS+=( --quantization "$QUANTIZATION" )
fi

# Custom all-reduce is a wash on PCIe P2P (RCCL already lands at similar
# latency without gfx906-tuned xGMI) and adds a stale-data risk surface.
if [ "${CUSTOM_ALL_REDUCE:-0}" != 1 ]; then
  ARGS+=( --disable-custom-all-reduce )
fi

# CUDA graphs: decode +50% over eager. 4-bit fused GEMM is only used up to
# M<=50, hence capture sizes and max-num-seqs stay at or below 48.
if [ "${EAGER:-0}" = 1 ]; then
  GRAPH=off
  ARGS+=( --enforce-eager )
else
  GRAPH="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
  ARGS+=( --compilation-config "{\"cudagraph_mode\":\"$GRAPH\",\"cudagraph_capture_sizes\":${CAPTURE_SIZES:-[1,2,4,8,16,24,32,48]}}" )
fi

# Vision encoders need a non-MFMA attention backend -- gfx906 has no MFMA.
# Both image and video are verified end to end on this hardware (16-frame clip
# ~1k prompt tokens); the dummy profile sizes off the max image feature, so no
# extra video budget cap is needed to fit 16GB.
MM="${MM:-auto}"
if [ "$MM" = auto ]; then
  if grep -q '"vision_config"' "$CFG"; then
    MM='{"image":1,"video":1}'
  else
    MM=off
  fi
fi
if [ "$MM" != off ]; then
  ARGS+=( --limit-mm-per-prompt "$MM" --mm-encoder-attn-backend TRITON_ATTN )
fi

printf '[serve] model=%s tp=%s dtype=%s max_len=%s max_seqs=%s graph=%s mm=%s custom_ar=%s w4a16=%s cache=%s\n' \
  "$MODEL" "$TP" "${DTYPE:-float16}" "${MAX_LEN:-16384}" "${MAX_NUM_SEQS:-48}" \
  "$GRAPH" "$MM" "${CUSTOM_ALL_REDUCE:-0}" "$W4A16" "$VLLM_CACHE_ROOT" >&2

exec vllm serve "${ARGS[@]}" "$@"
