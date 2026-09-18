#!/usr/bin/env bash
# 只读盘点 /home/x —— 不移动、不删除、不修改任何文件
set -u
H=/home/x

hr() { printf '\n===== %s =====\n' "$1"; }

hr "0. 磁盘总况"
df -h / /home 2>/dev/null | grep -vE '^tmpfs|^devtmpfs'
echo "--- inode ---"
df -i / | tail -1

hr "1. /home/x 顶层（含隐藏）"
ls -lah "$H"

hr "2. /home/x 一级体积（降序）"
du -h -d 1 "$H" 2>/dev/null | sort -h

hr "3. /home/x 二级体积（降序，只列 >50M）"
du -h -d 2 "$H" 2>/dev/null | sort -h | awk '{v=$1; gsub(/[0-9.]/,"",v); n=$1+0;
  if (v ~ /M/ && n<50) next; if (v ~ /K/ || v ~ /B/) next; print}'

hr "4. 目录数 / 文件数（按分支）"
for d in "$H"/*/ "$H"/.[!.]*/; do
  [ -d "$d" ] || continue
  nf=$(find "$d" -type f 2>/dev/null | wc -l)
  nd=$(find "$d" -type d 2>/dev/null | wc -l)
  printf '%-42s files=%-8s dirs=%s\n' "$(basename "$d")" "$nf" "$nd"
done

hr "5. 顶层散落的文件（非目录）"
find "$H" -maxdepth 1 -type f -printf '%10s  %TY-%Tm-%Td  %p\n' 2>/dev/null | sort -k1 -n -r

hr "6. 所有 shell / python / json 脚本（顶层及一级）"
find "$H" -maxdepth 2 -type f \( -name '*.sh' -o -name '*.py' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) -printf '%10s  %TY-%Tm-%Td  %p\n' 2>/dev/null | sort -k3

hr "7. 最大文件 TOP 30（>100M）"
find "$H" -xdev -type f -size +100M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -30 | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'

hr "8. 空目录"
find "$H" -xdev -type d -empty 2>/dev/null | head -40
echo "空目录总数: $(find "$H" -xdev -type d -empty 2>/dev/null | wc -l)"

hr "9. __pycache__ / *.pyc"
echo "pycache 目录: $(find "$H" -xdev -type d -name '__pycache__' 2>/dev/null | wc -l)"
echo "pyc 文件     : $(find "$H" -xdev -type f -name '*.pyc' 2>/dev/null | wc -l)"
du -sh $(find "$H" -xdev -type d -name '__pycache__' 2>/dev/null | head -200) 2>/dev/null | awk '{s+=$1} END{}' >/dev/null
find "$H" -xdev -type f -name '*.pyc' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "pyc 合计: %.1f MB\n", s/1048576}'

hr "10. 备份/临时/中间产物可疑项"
find "$H" -xdev \( -name '*.orig' -o -name '*.bak' -o -name '*.rej' -o -name '*~' -o -name '*.tmp' -o -name '*.old' -o -name 'core.*' -o -name '*.swp' \) -printf '%10s  %TY-%Tm-%Td  %p\n' 2>/dev/null | head -40
echo "小计: $(find "$H" -xdev \( -name '*.orig' -o -name '*.bak' -o -name '*.rej' -o -name '*~' -o -name '*.tmp' -o -name '*.old' -o -name 'core.*' -o -name '*.swp' \) 2>/dev/null | wc -l)"

hr "11. 日志文件"
find "$H" -xdev -type f \( -name '*.log' -o -name '*.log.*' -o -name '*.out' \) -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -25 | awk -F'\t' '{printf "%8.2f MB  %s  %s\n", $1/1048576, $2, $3}'
echo "--- 日志合计 ---"
find "$H" -xdev -type f \( -name '*.log' -o -name '*.log.*' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1;n++} END{printf "%d 个文件, %.2f GB\n", n, s/1073741824}'

hr "12. tar / 压缩包"
find "$H" -xdev -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' -o -name '*.gz' -o -name '*.xz' -o -name '*.zst' \) -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -20 | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'

hr "13. 按扩展名统计占用（TOP 15）"
find "$H" -xdev -type f -printf '%s\t%f\n' 2>/dev/null \
 | awk -F'\t' '{n=$2; if (n !~ /\./) e="(无扩展名)"; else {sub(/.*\./,"",n); e=tolower(n)}; s[e]+=$1; c[e]++} END{for (k in s) printf "%12.2f MB  %6d 个  .%s\n", s[k]/1048576, c[k], k}' \
 | sort -rn | head -15

hr "14. 最近 3 天修改过的文件（>1M）"
find "$H" -xdev -type f -size +1M -mtime -3 -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' 2>/dev/null | sort -rn | head -25 | awk -F'\t' '{printf "%8.2f MB  %s  %s\n", $1/1048576, $2, $3}'

hr "15. 最老的 20 个大文件（>500M）"
find "$H" -xdev -type f -size +500M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -k2 | head -20 | awk -F'\t' '{printf "%8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'

hr "16. docker 相关（镜像/容器/卷/构建缓存）"
echo "--- images ---"; docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' 2>/dev/null
echo "--- containers ---"; docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Size}}' 2>/dev/null
echo "--- volumes ---"; docker volume ls 2>/dev/null | head -20
echo "--- docker 根占用 ---"; du -sh /var/lib/docker 2>/dev/null
echo "--- 可回收（build cache / dangling）---"; docker system df 2>/dev/null

hr "17. 计划任务 / 自启动"
echo "--- /etc/cron.d ---"; ls -l /etc/cron.d/ 2>/dev/null
echo "--- crontab (x) ---"; crontab -l -u x 2>/dev/null || echo "(无)"
echo "--- root crontab ---"; sudo -n crontab -l 2>/dev/null || echo "(需权限或为空)"
echo "--- systemd user units ---"; ls -l ~/.config/systemd/user/ 2>/dev/null || echo "(无)"

hr "18. 监听端口与在跑进程"
ss -ltnp 2>/dev/null | grep -E 'LISTEN' | head -20
echo "--- 与 vllm/llama 相关进程 ---"
ps -eo pid,etime,rss,args 2>/dev/null | grep -E '[v]llm|[l]lama|[r]ocm' | head -15 || echo "(无)"

hr "19. 环境变量里的 vLLM 相关（当前 shell）"
env | grep -iE 'VLLM|LLAMA|ROCM|HSA|HF_' | sort || echo "(无)"

hr "20. /home/x 之外的关联位置"
for d in /opt/gfx906 /logs /models /dev/shm /tmp; do
  if [ -d "$d" ]; then printf '%-14s ' "$d"; du -sh "$d" 2>/dev/null | cut -f1; else echo "$d 不存在"; fi
done
echo "--- /dev/shm 明细 ---"; ls -lah /dev/shm 2>/dev/null | head -15

echo
echo "===== 盘点结束（未修改任何文件） ====="
