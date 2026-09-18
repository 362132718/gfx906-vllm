#!/bin/bash
# start-35b.sh -- 起 Qwen3.6-35B-A3B（MoE，AWQ 4bit）· 短上下文规格
#
# 【定位】与 start-27b.sh 同构的薄封装：钉模型 + 换模型前先停旧的 + 把参数档案
# 交给 start-vllm.sh。启动机制不复制（见 start-27b.sh 头部说明）。
#
# 【为什么这里必须显式写参数】start-vllm.sh 的默认档是**按 27B 固化**的
# （131072 / int8 / prefix16 / kv-cache-memory-bytes 3389653545）。后者的
# 3389653545 是从「27B 权重 + 本机 free 15.63 GiB」点定的**固定字节数**，
# 换 35B 会算错账（35B 权重 25 GB/双卡 ≈12.5 GB/卡，27B ≈9.5 GB/卡）。
# ⇒ 35B 必须退回 gfx906-serve 的原生默认，也就是下面这 6 个值。
#
# 【参数档案 = 2026-09-11 验收口径】实测 46.3 tok/s 单流 / 84.2 tok/s 8 路聚合
# / 图像 6/6 / 视频 3/3 / KV 57,344 tokens（`p8_accept_b.sh` 用的就是这套）。
#   MAX_LEN=16384（serve.sh 原生默认）
#   GPU_MEM_UTIL=0.90
#   MAX_NUM_SEQS=48（必须 ≤ 可用 Mamba block 数：16K+gmu0.90 时只有 53 个）
#   KV_CACHE_DTYPE=none / PREFIX_MATCH_UNIT=none（serve.sh 原生默认，无这两项）
#   KV_CACHE_MEM_BYTES=none（关键：不能用 27B 的定点池）
# MoE 的调优 kernel config 已经烘进镜像（fused_moe/configs/
# E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json），
# vLLM 按设备名自动加载 ⇒ **不需要** VLLM_TUNED_CONFIG_FOLDER，46.3 tok/s 默认就开。
#
# 【想抬 MAX_LEN】先起一次，看日志里的 `GPU KV cache size` 和它建议的池字节，
# 再按那个值改 MAX_LEN；别照搬 27B 的 131072。
#
# 【用法】
#   ./start-35b.sh                    # 起 35B（会先把 27B 停掉）
#   NO_STOP=1 ./start-35b.sh          # 不停旧的
#   DRY_RUN=1 ./start-35b.sh          # 只打印命令行
#   MAX_LEN=32768 ./start-35b.sh      # 覆盖档案里的单个值（env 优先）
set -uo pipefail
cd "$(dirname "$0")" || exit 1

MODEL=/models/vllm/Qwen3.6-35B-A3B-AWQ-4bit
SERVED_NAME=Qwen3.6-35B-A3B-AWQ-4bit

# 参数档案（env 优先，但留神：改完是"没验过"的配置）
export MAX_LEN="${MAX_LEN:-16384}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-none}"
export PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-none}"
export KV_CACHE_MEM_BYTES="${KV_CACHE_MEM_BYTES:-none}"

# ---- 换模型：先停旧的 ----
# ⚠ DRY_RUN 时必须跳过：否则"只想看命令行"会把正在跑的服务停掉且不起新的。
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[dry-run] 跳过 stop-vllm.sh 与显存等待"
elif [ "${NO_STOP:-0}" != 1 ]; then
  ./stop-vllm.sh || true      # 幂等：没在跑时它自己说"没有正在运行的 vLLM 服务"
fi

# ---- 等显存真正回收（同 start-27b.sh）----
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
