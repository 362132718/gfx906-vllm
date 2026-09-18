#!/usr/bin/env bash
# 收摊清理 · 续（剩余目录属主含 root ⇒ 用 sudo）
set -uo pipefail
H=/home/x
FAIL=0
hr(){ printf '\n########## %s ##########\n' "$1"; }

safe_sudo_rm() {
  local p="$1"
  case "$p" in
    /|/home|/home/x|"$H"/models|"$H"/logs|"$H"/bin|"$H"/llama-swap|"$H"/llamacpp-gfx906-slot-save|\
"$H"/comfyui|"$H"/images|"$H"/models/vllm|"$H"/models/vllm/Qwen3-0.6B-real)
      echo "  🛑 REFUSE(禁区): $p"; return 1;;
  esac
  case "$p" in *'*'*|*'?'*) echo "  🛑 REFUSE(通配符): $p"; return 1;; esac
  if [ ! -e "$p" ]; then echo "  – skip: $p"; return 0; fi
  if sudo -n rm -rf -- "$p"; then echo "  ✓ rm  $p"; else echo "  ✗ FAIL $p"; return 1; fi
}

batch(){
  local name="$1"; shift
  hr "$name"
  for p in "$@"; do safe_sudo_rm "$p" || FAIL=1; done
  [ "$FAIL" = 1 ] && { echo "  ⚠ 有失败 ⇒ 停止"; return 1; }
  return 0
}

batch "续 7/10：被 root 占有的目录 (7)" \
  "$H/vllm-work" "$H/vllm-cache" "$H/vllm-src" "$H/archive_old_work" \
  "$H/llm-cache" "$H/image-export" "$H/ccache" || exit 1

batch "续 8/10：模型残留 (3)" \
  "$H/models/vllm-cache" "$H/models/vllm/Qwen3-0.6B" "$H/models/vllm/MiniCPM-o-4.5-awq" || exit 1

hr "续 9/10：vLLM 日志"
for f in serve.log serve-16k.log fp8_reject_20260917.log image_build_20260917.log \
         furnace-build.log furnace-test.err furnace-test.out longtest.out \
         vllm-custom.log vllm-image-pull.log vllm-pull-ghcr2.log vllm-pull-ghcr-loop.log \
         vllm-pull-ghcr-loop.out vllm-pull.log vllm-pull-mirror.log \
         dl-minicpm.log dl-model.log pull-0191.log pull-0231.log pull-jackd.log; do
  safe_sudo_rm "$H/logs/$f"
done
echo "  logs/ 剩余:"; ls -A "$H/logs" | sed 's/^/    /'

hr "顶层空目录"
find "$H" -maxdepth 1 -mindepth 1 -type d -empty 2>/dev/null | sed 's/^/    /'
find "$H" -maxdepth 1 -mindepth 1 -type d -empty -exec sudo -n rm -rf {} + 2>/dev/null
safe_sudo_rm "$H/wind_down_cleanup.sh"

hr "Docker：删容器 + 旧镜像 + 构建缓存（保留 vllm-gfx906:latest）"
docker rm vllm 2>&1 | sed 's/^/  /'
docker rmi vllm-gfx906:pre-cdnafix-20260917 2>&1 | tail -3 | sed 's/^/  /'
docker rmi vllm-gfx906:rebuilt 2>&1 | tail -3 | sed 's/^/  /'
docker rmi rocm/dev-ubuntu-24.04:7.2.4-complete 2>&1 | tail -3 | sed 's/^/  /'
docker builder prune -af 2>&1 | tail -3 | sed 's/^/  /'

hr "最终状态"
echo "--- docker images ---"; docker images --format '  {{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'
echo "--- docker system df ---"; docker system df | sed 's/^/  /'
echo "--- /home/x 顶层 ---"; ls -A "$H" | sed 's/^/  /'
echo "--- 保留物核对 ---"
for p in "$H/images/vllm-gfx906-latest-2f6d5bf6.tar" "$H/bin/llama-swap" "$H/llama-swap/config.yaml" \
         "$H/llama-swap/Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-LynnStyle-Q4-imatrix.sh" \
         "$H/llamacpp-gfx906-slot-save/build-rocm724-fix/bin/llama-server" \
         "$H/start-llama-swap.sh" "$H/stop-llama-swap.sh" "$H/comfyui/models" \
         "$H/models/vllm/Qwen3-0.6B-real/model.safetensors" "$H/.vllm-api-key" \
         /etc/cron.d/amdgpu-fancontrol /etc/cron.d/amdgpu-powercap /etc/cron.d/amdgpu-pp-fan-patch; do
  printf '  %-8s %s\n' "$([ -e "$p" ] && echo OK || echo MISSING)" "$p"
done
echo "--- 磁盘 ---"; df -h / | tail -1 | sed 's/^/  /'
printf '\n########## 收摊清理结束 ##########\n'
