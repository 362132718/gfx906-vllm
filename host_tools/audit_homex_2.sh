#!/usr/bin/env bash
# 只读盘点 第二轮 —— 不移动、不删除、不修改任何文件
set -u
H=/home/x
hr() { printf '\n===== %s =====\n' "$1"; }

hr "A. models/ 现状（上次 138G，现在 1.6G —— 发生了什么）"
echo "--- models 顶层 ---"; ls -lah "$H/models"
echo "--- models 递归深度 2 ---"; du -h -d 2 "$H/models" 2>/dev/null | sort -h
echo "--- models 下所有 >100M 的文件 ---"
find "$H/models" -xdev -type f -size +100M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'
echo "--- models 下文件数/总数 ---"; find "$H/models" -type f 2>/dev/null | wc -l

hr "B. 全盘找 GGUF（确认 llama.cpp 资产是否还在）"
find / -xdev -type f -name '*.gguf' -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'
echo "(以上为全部 .gguf)"

hr "C. llama.cpp 相关资产"
for d in "$H/bin" "$H/llama-swap" "$H/llamacpp-gfx906-slot-save" "$H/llm-cache"; do
  echo "--- $d ---"; ls -lah "$d" 2>/dev/null | head -20; echo "   体积: $(du -sh "$d" 2>/dev/null | cut -f1)"
done

hr "D. vllm-work/ 一级"
du -h -d 1 "$H/vllm-work" 2>/dev/null | sort -h
echo "--- 子目录清单 ---"; ls -d "$H/vllm-work"/*/ 2>/dev/null

hr "E. logs/ 明细"
du -h -d 1 "$H/logs" 2>/dev/null | sort -h
echo "--- logs 顶层文件 ---"; ls -lah "$H/logs" 2>/dev/null
echo "--- kv-tier 落盘情况（腿0 实测产物）---"
if [ -d "$H/logs/kv-tier" ]; then
  echo "体积: $(du -sh "$H/logs/kv-tier" 2>/dev/null | cut -f1)  文件数: $(find "$H/logs/kv-tier" -type f 2>/dev/null | wc -l)"
  ls -d "$H/logs/kv-tier"/*/ 2>/dev/null | head -5
fi

hr "F. vllm-cache/ 明细（AOT/torch compile 缓存，可重建）"
du -h -d 3 "$H/vllm-cache" 2>/dev/null | sort -h | tail -15

hr "G. archive_old_work/ 明细"
du -h -d 1 "$H/archive_old_work" 2>/dev/null | sort -h
ls -la "$H/archive_old_work" 2>/dev/null

hr "H. 其它候选目录体积"
for d in "$H/comfyui" "$H/.cache" "$H/snap" "$H/.rustup" "$H/.cargo" "$H/.lmstudio" "$H/下载" "$H/backup-rocm-tensors" "$H/vllm-src" "$H/reinstinct" "$H/model_benchmark" "$H/rccl-gfx906-2.27.7-rocm7.2.4" "$H/.local" "$H/.config" "$H/image-export" "$H/ccache"; do
  [ -e "$d" ] && printf '%-46s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done

hr "I. 顶层散落文件合计（json/sh/py/其他）"
cnt=$(find "$H" -maxdepth 1 -type f | wc -l)
tot=$(find "$H" -maxdepth 1 -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%.2f MB", s/1048576}')
echo "顶层散落文件数: $cnt  合计: $tot"
echo "--- 按类型分组 ---"
find "$H" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sed 's/.*\.//' | sort | uniq -c | sort -rn

hr "J. docker 真实根目录与占用"
docker info 2>/dev/null | grep -E 'Docker Root Dir|Storage Driver|Total Memory|Name:'
echo "--- 各镜像 ID / 大小 / 是否被容器使用 ---"
docker images --format '{{.ID}}\t{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' 2>/dev/null
echo "--- 容器用的镜像 ID ---"
docker inspect vllm --format '{{.Image}}' 2>/dev/null
echo "--- 容器状态 ---"
docker inspect vllm --format 'Cmd={{.Config.Cmd}} | State={{.State.Status}} | Started={{.State.StartedAt}}' 2>/dev/null
echo "--- 容器挂载 ---"
docker inspect vllm --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{println}}{{end}}' 2>/dev/null

hr "K. 磁盘占用全景"
df -h | grep -vE '^tmpfs|^devtmpfs|^overlay'
echo "--- / 下各一级目录 ---"
sudo -n du -sh /* 2>/dev/null | sort -h | tail -15 || du -sh /home /usr /var /opt 2>/dev/null

hr "L. 回收站/残留"
ls -la "$H/.local/share/Trash" 2>/dev/null || echo "(无 Trash)"
find "$H" -maxdepth 1 -type d -name '*delete*' -o -maxdepth 1 -type d -name '*trash*' 2>/dev/null

echo
echo "===== 第二轮盘点结束（未修改任何文件） ====="
