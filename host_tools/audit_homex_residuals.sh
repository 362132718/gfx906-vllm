#!/usr/bin/env bash
# 只读：收摊后 /home/x 残留盘点（找还能清的东西）
set -u
H=/home/x
hr(){ printf '\n===== %s =====\n' "$1"; }

hr "0. 磁盘"
df -h / | tail -1
echo "/home/x 合计: $(du -sh $H 2>/dev/null | cut -f1)"

hr "1. 顶层全部（含隐藏）体积降序"
du -h -d 1 "$H" 2>/dev/null | sort -h

hr "2. 顶层散落文件"
find "$H" -maxdepth 1 -type f -printf '%10s  %TY-%Tm-%Td  %f\n' 2>/dev/null | sort -rn

hr "3. 还有没有 vLLM 残留（全盘 /home/x 内）"
echo "--- 名字含 vllm/kv-tier/triton 的目录 ---"
find "$H" -maxdepth 4 -type d \( -iname '*vllm*' -o -iname '*kv-tier*' -o -iname '*triton*' \) 2>/dev/null | sed 's/^/  /'
echo "--- 名字含 vllm 的文件 ---"
find "$H" -maxdepth 4 -type f -iname '*vllm*' 2>/dev/null | head -20 | sed 's/^/  /'
echo "--- /dev/shm 与 /tmp 残留 ---"
ls -la /dev/shm 2>/dev/null | sed 's/^/  /'
ls -la /tmp 2>/dev/null | head -12 | sed 's/^/  /'

hr "4. 隐藏目录明细（>1M）"
for d in "$H"/.[!.]*/; do
  s=$(du -s -m "$d" 2>/dev/null | cut -f1)
  [ "${s:-0}" -ge 1 ] && printf '  %6s MB  %s\n' "$s" "$d"
done

hr "5. .cache 明细"
du -h -d 1 "$H/.cache" 2>/dev/null | sort -h | tail -14
echo "--- huggingface hub 里还剩什么 ---"
ls -la "$H/.cache/huggingface/hub" 2>/dev/null | sed 's/^/  /'

hr "6. .local / .config / .modelscope 明细"
du -h -d 2 "$H/.local" 2>/dev/null | sort -h | tail -8
du -h -d 1 "$H/.config" 2>/dev/null | sort -h | tail -6
ls -la "$H/.modelscope" 2>/dev/null | sed 's/^/  /'

hr "7. llama.cpp 资产：3 个 build 哪个在用"
du -sh "$H"/llamacpp-gfx906-slot-save/build-* 2>/dev/null | sed 's/^/  /'
echo "--- 后端脚本实际用的是哪个 ---"
grep -h 'build-rocm' "$H"/llama-swap/*.sh 2>/dev/null | sort -u | sed 's/^/  /'
echo "--- llama.cpp .git 体积 ---"
du -sh "$H/llamacpp-gfx906-slot-save/.git" 2>/dev/null | sed 's/^/  /'
echo "--- 该仓库未跟踪/垃圾 ---"
git -C "$H/llamacpp-gfx906-slot-save" status --short 2>/dev/null | head -5 | sed 's/^/  /'
echo "--- models/ 里的 vocab gguf 合计 ---"
du -sh "$H/llamacpp-gfx906-slot-save/models" 2>/dev/null | sed 's/^/  /'

hr "8. models/ 现状"
du -h -d 2 "$H/models" 2>/dev/null | sort -h | sed 's/^/  /'
ls -la "$H/models/vllm" 2>/dev/null | sed 's/^/  /'

hr "9. logs/ 现状"
du -sh "$H/logs" 2>/dev/null | sed 's/^/  /'
ls -la "$H/logs" 2>/dev/null | sed 's/^/  /'

hr "10. images/（交付 tar，保留）"
ls -la "$H/images" 2>/dev/null | sed 's/^/  /'

hr "11. 其它与本项目无关但占地方的"
for d in "$H/.rustup" "$H/.cargo" "$H/snap" "$H/.lmstudio" "$H/下载" "$H/reinstinct" \
         "$H/comfyui" "$H/.xterminal" "$H/.docker" "$H/.modelscope" "$H/.pki"; do
  [ -e "$d" ] && printf '  %-24s %s\n' "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done

hr "12. 下载/ 内容（ROcm 旧包）"
ls -lah "$H/下载" 2>/dev/null | sed 's/^/  /'

hr "13. comfyui 明细（你的另一条线，仅列不删）"
du -h -d 1 "$H/comfyui" 2>/dev/null | sort -h | tail -8 | sed 's/^/  /'

hr "14. 残留的备份/临时/日志类"
find "$H" -maxdepth 3 \( -name '*.orig' -o -name '*.bak' -o -name '*.rej' -o -name '*~' \
     -o -name '*.tmp' -o -name '*.old' -o -name 'core.*' \) 2>/dev/null | head -20 | sed 's/^/  /'
echo "--- 空目录 ---"
find "$H" -maxdepth 2 -type d -empty 2>/dev/null | sed 's/^/  /'
echo "--- 顶层之外的大文件 TOP 10 ---"
find "$H" -xdev -type f -size +100M -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | head -10 \
 | awk -F'\t' '{printf "  %8.2f GB  %s  %s\n", $1/1073741824, $2, $3}'

echo
echo "===== 盘点结束（未修改任何文件） ====="
