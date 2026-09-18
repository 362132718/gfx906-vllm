#!/usr/bin/env bash
# 收摊清理 · 第三轮：按"建议档"删（第一档 + 第二档）
#   第一档：确属我方残留（~155 M）
#   第二档：vLLM 冒烟模型（1.5 G）
# 不动：第三档（.rustup/.cargo/reinstinct/.lmstudio/下载/snap）
set -uo pipefail
H=/home/x
FAIL=0
hr(){ printf '\n########## %s ##########\n' "$1"; }

safe_rm(){  # 一律 sudo（多处属主是 root）
  local p="$1"
  case "$p" in
    /|/home|/home/x|"$H"/models|"$H"/logs|"$H"/bin|"$H"/llama-swap|"$H"/llamacpp-gfx906-slot-save|\
"$H"/comfyui|"$H"/images|"$H"/images/vllm-gfx906-latest-2f6d5bf6.tar)
      echo "  🛑 REFUSE(禁区): $p"; return 1;;
  esac
  case "$p" in *'*'*|*'?'*) echo "  🛑 REFUSE(通配符): $p"; return 1;; esac
  if [ ! -e "$p" ]; then echo "  – skip(不存在): $p"; return 0; fi
  if sudo -n rm -rf -- "$p"; then echo "  ✓ rm  $p"; else echo "  ✗ FAIL $p"; return 1; fi
}

batch(){
  local name="$1"; shift
  hr "$name"; local n=0
  for p in "$@"; do safe_rm "$p" || FAIL=1; n=$((n+1)); done
  echo "  条目: $n"
  [ "$FAIL" = 1 ] && { echo "  ⚠ 有失败 ⇒ 停止"; return 1; }
  return 0
}

df_before=$(df -B1 --output=used / | tail -1)

batch "第一档：确属我方残留 (5)" \
  "$H/models/manifests" \
  "$H/.cache/pip" \
  "$H/.cache/huggingface" \
  "$H/images/docker_save.log" \
  "$H/llamacpp-gfx906-slot-save/build-rocm724-rccl" || exit 1

batch "第二档：vLLM 冒烟模型 (2)" \
  "$H/models/vllm/Qwen3-0.6B-real" \
  "$H/models/vllm" || exit 1

hr "终态核对"
echo "--- /home/x 顶层 ---"; ls -A "$H" | sed 's/^/  /'
echo "--- /home/x 一级体积 ---"; du -h -d 1 "$H" 2>/dev/null | sort -h | tail -14 | sed 's/^/  /'
echo "--- models/ 与 logs/（bind mount 目标，保留目录本身）---"
ls -la "$H/models" | sed 's/^/  /'; ls -A "$H/logs" | sed 's/^/  /'
echo "--- 保留物终检 ---"
for p in "$H/images/vllm-gfx906-latest-2f6d5bf6.tar" "$H/bin/llama-swap" "$H/llama-swap/config.yaml" \
         "$H/llamacpp-gfx906-slot-save/build-rocm724-fix/bin/llama-server" \
         "$H/llamacpp-gfx906-slot-save/build-rocm724/bin/llama-server" \
         "$H/llamacpp-gfx906-slot-save/.git" "$H/llamacpp-gfx906-slot-save/src/llama-context.cpp.orig" \
         "$H/start-llama-swap.sh" "$H/stop-llama-swap.sh" "$H/comfyui/models" "$H/.vllm-api-key"; do
  printf '  %-8s %s\n' "$([ -e "$p" ] && echo OK || echo MISSING)" "$p"
done
echo "--- 未动的第三档 ---"
for d in "$H/.rustup" "$H/.cargo" "$H/reinstinct" "$H/.lmstudio" "$H/下载" "$H/snap"; do
  [ -e "$d" ] && printf '  %-14s %s\n' "$(basename $d)" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo "--- docker ---"; docker images --format '  {{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'
echo "--- 磁盘 ---"
df_after=$(df -B1 --output=used / | tail -1)
echo "  本次释放: $(( (df_before - df_after) / 1048576 )) MiB"
df -h / | tail -1 | sed 's/^/  /'
printf '\n########## 结束 ##########\n'
