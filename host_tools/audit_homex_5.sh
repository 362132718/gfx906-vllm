#!/usr/bin/env bash
# 只读盘点 第五轮：报告所需的收尾信息
set -u
H=/home/x
hr() { printf '\n===== %s =====\n' "$1"; }

hr "C. llama-swap 配置与后端脚本"
cat "$H/llama-swap/config.yaml" 2>/dev/null
echo "--- 后端脚本引用的模型路径 ---"
grep -hE '\-\-model|model=' "$H/llama-swap"/*.sh 2>/dev/null | sed 's/^[[:space:]]*//'

hr "D. llama.cpp 二进制"
for b in build-rocm724 build-rocm724-fix build-rocm724-rccl; do
  d="$H/llamacpp-gfx906-slot-save/$b/bin"
  echo "--- $b: $(du -sh "$H/llamacpp-gfx906-slot-save/$b" 2>/dev/null | cut -f1) ---"
  ls -lh "$d"/llama-server "$d"/llama-cli "$d"/libggml* 2>/dev/null | head -6
done
[ -d "$H/llamacpp-gfx906-slot-save/build-rocm724-fix" ] && ls -d "$H/llamacpp-gfx906-slot-save"/build-rocm724-fix/*/ 2>/dev/null | head
echo "--- 哪个 build 是最新的 ---"
ls -ldt "$H/llamacpp-gfx906-slot-save"/build-rocm724* 2>/dev/null

hr "E. vllm-work/docker"
du -h -d 2 "$H/vllm-work/docker" 2>/dev/null | sort -h | tail -10
echo "--- build-context 顶层 ---"; ls -lah "$H/vllm-work/docker/build-context" 2>/dev/null | head -20
echo "--- docker/ 目录 ---"; ls -lah "$H/vllm-work/docker" 2>/dev/null | head -25

hr "E2. vllm 源码树们"
for d in "$H/vllm-work/vllm" "$H/vllm-work/vllm-official" "$H/vllm-src" "$H/vllm-work/gfx906-supplements"; do
  printf '%-44s %s' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
  if [ -d "$d/.git" ]; then printf '  [git] %s' "$(git -C "$d" log -1 --format='%h %ad' --date=short 2>/dev/null | head -c 40)"; fi
  echo
done

hr "F. archive_old_work"
ls -la "$H/archive_old_work" 2>/dev/null
du -h -d 1 "$H/archive_old_work" 2>/dev/null | sort -h

hr "G. comfyui 顶层"
du -h -d 1 "$H/comfyui" 2>/dev/null | sort -h | tail -10

hr "H. ~/.cache / ~/.local / .config / snap 明细"
du -h -d 1 "$H/.cache" 2>/dev/null | sort -h | tail -8
du -h -d 1 "$H/.local" 2>/dev/null | sort -h | tail -6

hr "I. 下载目录"
ls -lah "$H/下载" 2>/dev/null

hr "J. 容器与端口"
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null
echo "--- 容器内进程 ---"; docker top vllm 2>/dev/null | head -6
echo "--- /dev/shm ---"; ls -lah /dev/shm
echo "--- docker images ---"; docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' 2>/dev/null
echo "--- docker system df -v（TOP 项）---"; docker system df -v 2>/dev/null | head -30

hr "L. vllm-work 大文件 TOP 10"
find "$H/vllm-work" -xdev -type f -size +30M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -10 | awk -F'\t' '{printf "%8.2f MB  %s  %s\n", $1/1048576, $2, $3}'

hr "M. 小目录内容"
for d in "$H/vllm-work/fp8kv" "$H/vllm-work/moe_tuning" "$H/vllm-work/lcpp-ref" "$H/vllm-work/triton_kernels_pkg" "$H/reinstinct" "$H/model_benchmark" "$H/backup-rocm-tensors" "$H/rccl-gfx906-2.27.7-rocm7.2.4" "$H/image-export" "$H/ccache" "$H/llm-cache"; do
  echo "--- $d ($(du -sh "$d" 2>/dev/null | cut -f1)) ---"
  ls -A "$d" 2>/dev/null | head -8 | tr '\n' ' '; echo
done

hr "N. /home/x 下所有 git 仓库及体积"
find "$H" -maxdepth 3 -type d -name '.git' 2>/dev/null | while read g; do
  d=$(dirname "$g"); printf '%-48s git=%-8s %s\n' "$d" "$(du -sh "$g" 2>/dev/null | cut -f1)" "$(git -C "$d" log -1 --format='%h %ad' --date=short 2>/dev/null | head -c 30)"
done

echo
echo "===== 第五轮盘点结束（未修改任何文件） ====="
