#!/usr/bin/env bash
# 收摊清理（用户已确认：直接删）
#   保留：vllm-gfx906:latest + images/*.tar + llama.cpp 全套 + comfyui + Qwen3-0.6B-real
#   删除：vLLM 测试脚本/payload/工作目录/旧镜像/构建缓存
# 纪律：分批（每批 ≤10）、每批校验、遇错即停；禁区名单硬保护；不用通配符。
set -uo pipefail
H=/home/x
FAIL=0

hr(){ printf '\n########## %s ##########\n' "$1"; }
df_before=$(df -B1 --output=used / | tail -1)

safe_rm() {
  local p="$1"
  case "$p" in
    /|/home|/home/x|"$H"/models|"$H"/logs|"$H"/bin|"$H"/llama-swap|"$H"/llamacpp-gfx906-slot-save|"$H"/comfyui|"$H"/images|"$H"/models/vllm|"$H"/models/vllm/Qwen3-0.6B-real)
      echo "  🛑 REFUSE(禁区): $p"; return 1;;
  esac
  case "$p" in
    *'*'*|*'?'*) echo "  🛑 REFUSE(通配符): $p"; return 1;;
  esac
  if [ ! -e "$p" ]; then echo "  – skip(不存在): $p"; return 0; fi
  if rm -rf -- "$p"; then echo "  ✓ rm  $p"; else echo "  ✗ FAIL $p"; return 1; fi
}

batch() {
  local name="$1"; shift
  hr "$name"
  local n=0
  for p in "$@"; do
    safe_rm "$p" || FAIL=1
    n=$((n+1))
  done
  echo "  批次条目: $n"
  if [ "$FAIL" = 1 ]; then echo "  ⚠ 本批次有失败 ⇒ 停止后续删除"; return 1; fi
  return 0
}

# ---------------------------------------------------------------- 文件批次
batch "批次 1/10：顶层启动/运维脚本 (5)" \
  "$H/start-vllm.sh" "$H/stop-vllm.sh" "$H/start-27b.sh" "$H/start-35b.sh" \
  "$H/build_image_cdnafix.sh" || exit 1

batch "批次 2/10：探针脚本 (9)" \
  "$H/longctx_probe.sh" "$H/longctx_probe2.sh" "$H/prefix_hit_probe.sh" \
  "$H/prefix_hit_verify.sh" "$H/kv-reuse-probe.sh" "$H/probe_enc_attn_inner.sh" \
  "$H/probe_enc_attn2_inner.sh" "$H/probe_pth_offload_inner.sh" "$H/probe_gate2_inner.sh" || exit 1

batch "批次 3/10：请求 payload (10)" \
  "$H/lc_125k.json" "$H/lc_64k.json" "$H/lc2_32k.json" "$H/lc2_64k.json" \
  "$H/mm_image.json" "$H/mm_text.json" "$H/p_long_a.json" "$H/p_long_a2.json" \
  "$H/p_short_a.json" "$H/p_short_a2.json" || exit 1

batch "批次 4/10：请求 payload (10)" \
  "$H/pv_end_1121_1.json" "$H/pv_end_1121_2.json" "$H/pv_end_4465_1.json" "$H/pv_end_4465_2.json" \
  "$H/pv_mid_1121_1.json" "$H/pv_mid_1121_2.json" "$H/pv_mid_4465_1.json" "$H/pv_mid_4465_2.json" \
  "$H/ph_A_tiny_1.json" "$H/ph_A_tiny_2.json" || exit 1

batch "批次 5/10：payload 尾 + 补丁 + 本次审计临时脚本 (8)" \
  "$H/ph_B_just_under_blk_1.json" "$H/ph_B_just_under_blk_2.json" \
  "$H/ph_C_multi_blk_1.json" "$H/ph_C_multi_blk_2.json" \
  "$H/patch_rocm_cdna.py" "$H/verify_keep_image.sh" \
  "$H/check_rebuildability.sh" "$H/extract_missing_build_inputs.sh" || exit 1

batch "批次 6/10：临时脚本 + 旧镜像归档 tar (3)" \
  "$H/test_supplement_fallback.sh" "$H/verify_tar_structure.sh" \
  "$H/vllm-gfx906-c31a8b65.tar" || exit 1

# ---------------------------------------------------------------- 目录批次
batch "批次 7/10：vLLM 工作/缓存/归档目录 (10)" \
  "$H/vllm-work" "$H/vllm-cache" "$H/vllm-src" "$H/logs/kv-tier" \
  "$H/archive_old_work" "$H/backup-rocm-tensors" "$H/rccl-gfx906-2.27.7-rocm7.2.4" \
  "$H/llm-cache" "$H/model_benchmark" "$H/_rebuild_check" || exit 1

batch "批次 8/10：模型残留 + 空壳目录 (5)" \
  "$H/models/vllm-cache" "$H/image-export" "$H/ccache" \
  "$H/models/vllm/Qwen3-0.6B" "$H/models/vllm/MiniCPM-o-4.5-awq" || exit 1

hr "批次 9/10：vLLM 日志（保留 llama.cpp 日志）"
for f in serve.log serve-16k.log fp8_reject_20260917.log image_build_20260917.log \
         furnace-build.log furnace-test.err furnace-test.out longtest.out \
         vllm-custom.log vllm-image-pull.log vllm-pull-ghcr2.log vllm-pull-ghcr-loop.log \
         vllm-pull-ghcr-loop.out vllm-pull.log vllm-pull-mirror.log \
         dl-minicpm.log dl-model.log pull-0191.log pull-0231.log pull-jackd.log; do
  safe_rm "$H/logs/$f" || true
done
echo "  logs/ 剩余:"; ls -A "$H/logs" | sed 's/^/    /'

hr "批次 10/10：仅清 /home/x 顶层空目录（不伸进隐藏目录）"
echo "  清理前顶层空目录："
find "$H" -maxdepth 1 -mindepth 1 -type d -empty 2>/dev/null | sed 's/^/    /'
find "$H" -maxdepth 1 -mindepth 1 -type d -empty -exec rm -rf {} + 2>/dev/null
echo "  清理后顶层空目录：$(find "$H" -maxdepth 1 -mindepth 1 -type d -empty 2>/dev/null | wc -l)"
echo "  ℹ 有意保留：.config/.local/.cache/.lmstudio/snap 等隐藏目录内的空目录（应用可能依赖）"
echo "  ℹ 有意保留：*.pyc / __pycache__（约 93 MB，删了收益极小，动 .local 风险更大）"

# ---------------------------------------------------------------- docker
hr "Docker：删容器 + 旧镜像 + 构建缓存（保留 vllm-gfx906:latest）"
docker rm vllm 2>&1 | sed 's/^/  /'
docker rmi vllm-gfx906:pre-cdnafix-20260917 vllm-gfx906:rebuilt 2>&1 | tail -4 | sed 's/^/  /'
docker rmi rocm/dev-ubuntu-24.04:7.2.4-complete 2>&1 | tail -3 | sed 's/^/  /'
docker builder prune -af 2>&1 | tail -3 | sed 's/^/  /'

hr "结果"
echo "--- docker images ---"; docker images --format '  {{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'
echo "--- docker system df ---"; docker system df | sed 's/^/  /'
echo "--- 保留物核对 ---"
for p in "$H/images/vllm-gfx906-latest-2f6d5bf6.tar" "$H/bin/llama-swap" "$H/llama-swap/config.yaml" \
         "$H/llamacpp-gfx906-slot-save/build-rocm724-fix/bin/llama-server" \
         "$H/start-llama-swap.sh" "$H/stop-llama-swap.sh" "$H/comfyui/models" \
         "$H/models/vllm/Qwen3-0.6B-real/model.safetensors" "$H/.vllm-api-key"; do
  printf '  %s  %s\n' "$([ -e "$p" ] && echo OK || echo MISSING)" "$p"
done
df_after=$(df -B1 --output=used / | tail -1)
echo "--- 磁盘 ---"
echo "  释放: $(( (df_before - df_after) / 1073741824 )) GiB"
df -h / | tail -1 | sed 's/^/  /'
printf '\n########## 收摊清理结束 ##########\n'
