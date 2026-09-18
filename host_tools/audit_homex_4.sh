#!/usr/bin/env bash
# 只读盘点 第四轮：补齐报告所需细节
set -u
H=/home/x
hr() { printf '\n===== %s =====\n' "$1"; }

hr "A. 顶层散落文件全清单（按大小）"
find "$H" -maxdepth 1 -type f -printf '%10s  %TY-%Tm-%Td  %f\n' 2>/dev/null | sort -rn

hr "B. models/vllm 明细"
ls -laR "$H/models/vllm/MiniCPM-o-4.5-awq" 2>/dev/null | head -20
echo "--- Qwen3-0.6B ---"; ls -la "$H/models/vllm/Qwen3-0.6B" 2>/dev/null
echo "--- Qwen3-0.6B-real ---"; ls -la "$H/models/vllm/Qwen3-0.6B-real" 2>/dev/null
echo "--- models/vllm-cache ---"; du -h -d 2 "$H/models/vllm-cache" 2>/dev/null

hr "C. llama-swap 配置（看它引用的 GGUF 是否已不存在）"
cat "$H/llama-swap/config.yaml" 2>/dev/null
echo "--- 各后端脚本 ---"
for f in "$H/llama-swap"/*.sh; do echo "### $f"; cat "$f"; echo; done 2>/dev/null

hr "D. llama.cpp 二进制是否还在"
ls -la "$H/llamacpp-gfx906-slot-save/build-rocm724/bin/" 2>/dev/null | grep -E 'llama|ggml' | head -12
ls -la "$H/llamacpp-gfx906-slot-save/build-rocm724-fix/bin/" 2>/dev/null | grep -E 'llama-server|llama-cli' | head
which llama-server llama-cli 2>/dev/null || echo "(PATH 里没有 llama-server)"
echo "--- 三个 build 目录体积 ---"
du -sh "$H/llamacpp-gfx906-slot-save"/build-rocm724* 2>/dev/null

hr "E. vllm-work/docker 与 build-context"
du -h -d 2 "$H/vllm-work/docker" 2>/dev/null | sort -h | tail -12
echo "--- build-context 顶层 ---"; ls -lah "$H/vllm-work/docker/build-context" 2>/dev/null | head -15
echo "--- git 仓库们（哪些是源码树）---"
for d in "$H/vllm-work" "$H/vllm-work/vllm" "$H/vllm-work/vllm-official" "$H/vllm-src" "$H/archive_old_work" "$H/llamacpp-gfx906-slot-save" "$H/comfyui"; do
  g="$d/.git"
  if [ -d "$g" ]; then
    printf '%-40s %s  最后一个提交: %s\n' "$d" "$(du -sh "$g" 2>/dev/null | cut -f1)" "$(git -C "$d" log -1 --format='%h %ad %s' --date=short 2>/dev/null | head -c 70)"
  fi
done

hr "F. archive_old_work 明细"
ls -la "$H/archive_old_work" 2>/dev/null
du -h -d 1 "$H/archive_old_work" 2>/dev/null | sort -h

hr "G. comfyui 顶层"
du -h -d 1 "$H/comfyui" 2>/dev/null | sort -h | tail -12

hr "H. ~/.cache 与 ~/.local 明细"
du -h -d 1 "$H/.cache" 2>/dev/null | sort -h | tail -12
du -h -d 1 "$H/.local" 2>/dev/null | sort -h | tail -8

hr "I. 下载目录"
ls -lah "$H/下载" 2>/dev/null

hr "J. 容器运行状态"
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
echo "--- 容器内进程 ---"
docker top vllm 2>/dev/null | head -8 || echo "(docker top 失败)"
echo "--- 端口 8000 归属 ---"
sudo -n ss -ltnp 2>/dev/null | grep 8000 || ss -ltnp 2>/dev/null | grep 8000 || echo "(无)"
echo "--- /dev/shm ---"; ls -lah /dev/shm

hr "K. docker 存储真实位置"
sudo -n docker info 2>/dev/null | grep -E 'Docker Root Dir|Storage Driver' || echo "(docker info 需权限)"

hr "L. vllm-work 各 p* 目录里最大的文件 TOP 12"
find "$H/vllm-work" -xdev -type f -size +50M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -12 | awk -F'\t' '{printf "%8.2f MB  %s  %s\n", $1/1048576, $2, $3}'

hr "M. fp8kv / moe_tuning / gfx906-supplements 内容"
for d in "$H/vllm-work/fp8kv" "$H/vllm-work/moe_tuning" "$H/vllm-work/gfx906-supplements" "$H/vllm-work/triton_kernels_pkg" "$H/vllm-work/lcpp-ref"; do
  echo "--- $d ($(du -sh "$d" 2>/dev/null | cut -f1)) ---"; ls -la "$d" 2>/dev/null | head -12
done

echo
echo "===== 第四轮盘点结束（未修改任何文件） ====="
