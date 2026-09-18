#!/usr/bin/env bash
# 只读盘点 第三轮：模型去向 + 挂载 + 历史操作线索
set -u
H=/home/x
hr() { printf '\n===== %s =====\n' "$1"; }

hr "A. 所有文件系统与挂载点"
df -h
echo "--- 块设备 ---"; lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT 2>/dev/null
echo "--- mount 里的非常规挂载 ---"; mount | grep -vE 'proc|sysfs|cgroup|devpts|tmpfs|securityfs|debugfs|tracefs|pstore|bpf|configfs|fusectl|hugetlbfs|mqueue|ramfs|autofs|binfmt' | head -20

hr "B. 全盘（跨文件系统）找大模型文件"
echo "--- .gguf（含非 xdev）---"
find / -type f -name '*.gguf' -size +100M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'
echo "--- .safetensors >500M（跨文件系统）---"
find / -type f -name '*.safetensors' -size +500M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'
echo "--- 所有 >3G 的文件（跨文件系统）---"
find / -type f -size +3G -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -25 | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'

hr "C. 容器挂载与可见的模型目录"
docker inspect vllm --format '{{range .Mounts}}{{.Type}}  {{.Source}}  ->  {{.Destination}}{{"\n"}}{{end}}' 2>/dev/null
echo "--- 容器内 /models ---"
docker exec vllm ls -lah /models 2>/dev/null | head -20
echo "--- 容器内 /models/vllm ---"
docker exec vllm ls -lah /models/vllm 2>/dev/null | head -20

hr "D. 主机 /home/x/models 是否独立挂载"
findmnt -T /home/x/models 2>/dev/null || echo "(findmnt 不可用)"
stat -f -c '%T  %b blocks * %S' /home/x/models 2>/dev/null

hr "E. bash 历史里的删除/移动线索（最近 60 条命令）"
tail -60 "$H/.bash_history" 2>/dev/null | grep -nE 'rm |mv |dd |shred|find .*-delete|rsync|trash' || echo "(未见删除/移动类命令)"
echo "--- 历史最后 30 行原样 ---"
tail -30 "$H/.bash_history" 2>/dev/null

hr "F. 回收站 / 快照 / 备份目录"
ls -la "$H/.local/share/Trash/files" 2>/dev/null | head || echo "(无 Trash/files)"
ls -la "$H/.local/share/Trash/info" 2>/dev/null | head || echo "(无 Trash/info)"
echo "--- btrfs/zfs/lvm 快照 ---"
which btrfs >/dev/null 2>&1 && btrfs subvolume list / 2>/dev/null | head || echo "(无 btrfs)"
lvs 2>/dev/null | head || echo "(无 LVM)"
zfs list 2>/dev/null | head || echo "(无 ZFS)"

hr "G. 最近 48 小时被修改的目录（按时间倒序，深度 2）"
find "$H" -maxdepth 2 -type d -mtime -2 -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head -40

hr "H. /opt 与其它可能存放模型的位置"
ls -la /opt 2>/dev/null
for d in /data /mnt /media /srv /ai /models /raid /storage /home/x/models; do
  [ -e "$d" ] && printf '%-18s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)" || true
done

hr "I. 当前 docker 容器 c31a8b65 与镜像"
docker ps -a --no-trunc --format '{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}'
echo "--- 镜像 ID 对照 ---"
docker images --no-trunc --format '{{.ID}}\t{{.Repository}}:{{.Tag}}' 2>/dev/null
echo "--- 运行中容器的镜像 vs :latest ---"
echo "container uses: $(docker inspect vllm --format '{{.Image}}' 2>/dev/null)"
echo "latest        : $(docker inspect vllm-gfx906:latest --format '{{.Id}}' 2>/dev/null)"

hr "J. logs/serve.log 尾部（看最后一次服务状态）"
tail -25 "$H/logs/serve.log" 2>/dev/null

echo
echo "===== 第三轮盘点结束（未修改任何文件） ====="
