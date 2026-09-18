#!/bin/bash
# start-27b.sh -- 起 Qwen3.8-27B（dense，AWQ W4A16）· 长上下文规格
#
# 【定位】薄封装，只做三件事：
#   1. 钉住要哪个模型（MODEL + SERVED_NAME）
#   2. 换模型时先把旧的停掉，并等显存真正释放
#   3. 其余全部交给 start-vllm.sh（API key / 日志 / 端口占用检查 / 等就绪 / 参数透传）
#   ⇒ 这里**不复制** start-vllm.sh 的任何逻辑，那边改一次这边自动跟上。
#
# 【参数档案】本模型直接吃 start-vllm.sh 的默认档 —— 那套值本来就是按 27B
# 固化的（MAX_LEN 131072 / gmu 0.97 / max_seqs 48 / int8_per_token_head /
# prefix-match-unit 16 / kv-cache-memory-bytes 3389653545）。所以这里**故意不
# 重复写一遍**，免得将来默认档升级了、这里还钉着旧数。
#   实测：188,686 KV tokens、128K 并发 1.44x、就绪 427 s（AOT 命中）。
#   要看当前实际生效的参数，跑 DRY_RUN=1 或看启动日志头几行。
#
# 【用法】
#   ./start-27b.sh                    # 起 27B（会先把当前跑着的模型停掉）
#   NO_STOP=1 ./start-27b.sh          # 不停旧的；端口被占就让 start-vllm.sh 报错退出
#   DRY_RUN=1 ./start-27b.sh          # 只打印命令行，不启动
#   ./start-27b.sh --max-num-seqs 8   # 额外参数原样转发给 vllm serve
#
# 【为什么同一时刻只能有一个】2×16 GB 装不下两个模型（27B ≈9.5 GB/卡，
# 35B ≈12.5 GB/卡），所以"选择启动"在物理上就等于"先下再上"。
set -uo pipefail
cd "$(dirname "$0")" || exit 1

MODEL=/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
SERVED_NAME=Qwen3.8-27B-OBLITERATED-AWQ-W4A16

# ---- 换模型：先停旧的 ----
# ⚠ DRY_RUN 时必须跳过：否则"只想看命令行"会把正在跑的服务停掉且不起新的。
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[dry-run] 跳过 stop-vllm.sh 与显存等待"
elif [ "${NO_STOP:-0}" != 1 ]; then
  ./stop-vllm.sh || true      # 幂等：没在跑时它自己说"没有正在运行的 vLLM 服务"
fi

# ---- 等显存真正回收 ----
# stop-vllm.sh 返回 ≠ 显存已归还：API server 退出后 worker 的 HSA 内存还要几秒
# 才释放，这时立刻起新模型会偶发 OOM。等 GPU0 用量降到 1.5 GB 以下再走。
read_vram() {
  rocm-smi --showmeminfo vram 2>/dev/null \
    | grep 'VRAM Total Used Memory' | head -1 | awk '{print $NF}'
}
if [ "${DRY_RUN:-0}" != 1 ]; then
  u=""
  for _ in $(seq 1 30); do        # 最多等 60 s
    u=$(read_vram)
    case "${u:-}" in ''|*[!0-9]*) u=0 ;; esac
    [ "$u" -lt 1500000000 ] && break
    sleep 2
  done
  printf '[prep] GPU0 VRAM used = %s B\n' "${u:-?}"
  [ "${u:-0}" -ge 1500000000 ] && echo "[prep] ⚠ 显存仍未明显释放，仍继续启动（失败就查 stop-vllm.sh --status）"
fi

export MODEL SERVED_NAME
exec ./start-vllm.sh "$@"
