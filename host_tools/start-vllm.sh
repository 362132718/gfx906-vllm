#!/bin/bash
# start-vllm.sh -- 启动 vLLM OpenAI 兼容 API 服务（gfx906 / 双卡 TP2）
#
# 用法:
#   ./start-vllm.sh                       # 默认模型 + 端口 8000
#   PORT=8001 ./start-vllm.sh             # 换端口
#   MODEL=/models/vllm/<其他模型>/ ./start-vllm.sh
#   MAX_LEN=32768 MAX_NUM_SEQS=16 ./start-vllm.sh
#   KV_CACHE_DTYPE=int8_per_token_head ./start-vllm.sh   # KV 量化：容量 ~1.95x，可上 128K
#   KV_CACHE_DTYPE=none ./start-vllm.sh   # 关掉 KV 量化（回到 fp16）
#   PREFIX_MATCH_UNIT=16 ./start-vllm.sh  # 前缀命中粒度细化（默认=整块 1552 ⇒ 短 prompt 命中率为 0）
#   KV_CACHE_MEM_BYTES=none ./start-vllm.sh  # 退回 gmu 自动定 KV 池（默认点定 3389653545 字节/卡）
#   KV_OFFLOAD_GB=16 ./start-vllm.sh     # KV 额外常驻 CPU 内存，复用前缀少做 prefill
#   KV_OFFLOAD_GB=16 KV_TIER_DISK_DIR=/logs/kv-tier ./start-vllm.sh   # 再加 SSD 一层（推荐）
#   DRY_RUN=1 KV_OFFLOAD_GB=16 ... ./start-vllm.sh                    # 只打印命令行，不启动
#   ./start-vllm.sh --max-num-seqs 8      # 额外参数原样转发给 vllm serve
#
# 环境变量: MODEL PORT HOST_BIND MAX_LEN MAX_NUM_SEQS GPU_MEM_UTIL TP DTYPE EAGER API_KEY CONTAINER LOG_DIR
# 可选透传: CAPTURE_SIZES CUDAGRAPH_MODE MM SERVED_NAME CUSTOM_ALL_REDUCE
# KV 量化: KV_CACHE_DTYPE（默认 int8_per_token_head；none/auto/空 = 不传该参数，用 fp16）
# 前缀粒度: PREFIX_MATCH_UNIT（默认 16；none/off/auto/default = 退回不传，用各 group 块大小的 GCD）
# KV 池: KV_CACHE_MEM_BYTES（默认 3389653545 字节/卡；none/off/auto/default = 不传，由 gmu 自动定）
#         设了就忽略 GPU_MEM_UTIL 且跳过 memory profiling ⇒ 见正文注释的取值来源与注意
# KV 复用: KV_OFFLOAD_GB KV_OFFLOAD_BACKEND(cpu|disk) KV_OFFLOAD_DISK_PATH KV_OFFLOAD_DISK_GB
#          KV_OFFLOAD_LAZY KV_OFFLOAD_POLICY(lru|arc) KV_TIER_DISK_DIR KV_TIER_IO_THREADS
# API key: 优先用 $API_KEY；否则读 /home/x/.vllm-api-key；再否则自动生成并存盘（600 权限）
set -euo pipefail

CONTAINER="${CONTAINER:-vllm}"
LOG_DIR="${LOG_DIR:-/home/x/logs}"
LOG_HOST="$LOG_DIR/serve.log"
LOG_CTR="/logs/serve.log"
KEY_FILE="${KEY_FILE:-/home/x/.vllm-api-key}"

# 默认档位 = 128K 长上下文（2026-09-16 实测固化）：
#   int8 KV 让 KV 池装下 188,686 tokens（fp16 同配置只有 116,032），
#   于是 max_model_len 131,072 能跑，并发度 1.44x，多模态可同时开。
#   KV 池按字节点定为 3389653545/卡（3.16 GiB，= gmu 0.97 的 "fit into requested memory" 值），
#   取代 profiler 给的超预算 3.67 GiB，把 activation 余量从 1.44 → 1.95 GiB（profile 峰值 1.68）。
#   短上下文/高并发场景可覆盖回去：MAX_LEN=16384 GPU_MEM_UTIL=0.90 KV_CACHE_DTYPE=none KV_CACHE_MEM_BYTES=none
MODEL="${MODEL:-/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16}"
PORT="${PORT:-8000}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"
TP="${TP:-2}"
DTYPE="${DTYPE:-float16}"
EAGER="${EAGER:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"

# ---------- 可选透传 ----------------------------------------------------------
# 空值一律不传，让 gfx906-serve 用它自己的默认值。
#   CAPTURE_SIZES     图捕获尺寸，如 '[1]'。单流长上下文时把大尺寸去掉可省下大量
#                     activation 显存（实测 size=48 时峰值 activation 1.68 GiB）。
#   CUDAGRAPH_MODE    FULL_DECODE_ONLY（默认）/ PIECEWISE / NONE
#   MM                每 prompt 多模态上限；off 关闭（省 vision 相关显存）
#   SERVED_NAME       API 里的模型名，默认取目录名
#   CUSTOM_ALL_REDUCE 1 = 启用（PCIe P2P 上无收益且有 stale-data 风险，默认关）
OPT_ENVS=""
add_opt() { if [ -n "${2:-}" ]; then OPT_ENVS="$OPT_ENVS $1=$(printf '%q' "$2")"; fi; return 0; }
add_opt CAPTURE_SIZES     "${CAPTURE_SIZES:-}"
add_opt CUDAGRAPH_MODE    "${CUDAGRAPH_MODE:-}"
add_opt MM                "${MM:-}"
add_opt SERVED_NAME       "${SERVED_NAME:-}"
add_opt CUSTOM_ALL_REDUCE "${CUSTOM_ALL_REDUCE:-}"

# ---------- KV 量化（可选：把 attention KV cache 压成 int8，换取更大上下文）-------
# 只认 int8_per_token_head：Triton 实现（写 triton_reshape_and_cache_flash、
# 读 triton_unified_attention 的 KV_QUANT_MODE=2），0.28.0 在运行时动态算
# per-token-head scale ⇒ **零校准**；gfx906 有 int8 算术 ⇒ 实测可用。
#   ⚠ fp8 千万别开：Vega20 没有 FP8 单元，而 vLLM 的 fp8 KV 支持范围是
#     "CUDA SM 8.9+ / ROCm gfx94+"，gfx906 不在内（叠加 on_cdna() 子串匹配 bug
#     还会让它"静默放行"，失败点后移到 kernel 里，极难排查）。
# 实测收益（gmu 0.97 / max_seqs 1 / capture[1]，同 3.71 GiB KV 池）：
#   fp16 → 116,032 tokens；int8 → **221,814 tokens（1.91x）**，128K 可跑。
#   机制：attention block size 自动从 784 翻倍到 1552（mamba 页是每块固定大小，
#   注意力每 token 字节减半 ⇒ 块长翻倍 ⇒ 每块改存 2x token 而页成本几乎不变）。
# 代价：KV 精度降为 int8（公开数据损失 ~1-2%，且集中在长上下文检索/计数任务）。
#   ⚠ 与多模态**不冲突**（ViT 编码器走独立的 MMEncoderAttention，不用 KV cache）。
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-int8_per_token_head}"
KV_QUANT_ARGS=""
case "$KV_CACHE_DTYPE" in
  ""|none|off|auto|float16|fp16|bfloat16) KV_CACHE_DTYPE="" ;;
  *) KV_QUANT_ARGS=" --kv-cache-dtype $(printf '%q' "$KV_CACHE_DTYPE")" ;;
esac

# ---------- 前缀缓存匹配粒度（hybrid / Mamba 模型必看）---------------------------
# 本模型 attention block = 1552 tokens，而**命中粒度默认就等于整块** ⇒
# 实测（host_tools/prefix_hit_probe.sh）：prompt < 1552 tokens 时命中率**恒为 0**；
# 长 prompt 也只复用 floor(len/1552)×1552，尾部余数永远重算。
# 设成更细的因子可让命中落到物理块**内部**的边界（0.28.0 的 config/cache.py:57 明确支持，
# 原文举例 "32 vs a 1024-token hybrid-model block"）。
#   ⚠ 必须整除**每个** KV cache group 的 block_size，否则启动直接 raise
#     （报错信息会打印各 group 的真实块大小，可用来反推 mamba 组的 block_size）。
#   1552 = 2^4 × 97 ⇒ 候选 2/4/8/16/97/194/388/776。
#   ✅ 2026-09-16 实测固化默认值 **16**：命中率 0% → 99.6~100%（见下），冷/热答案一致性已验证。
#      留空串 "none"/"off"/"auto"/"default" 可退回默认（= 各 group 块大小的 GCD）。
#
#      实测（同一 prompt 连发两次，第 2 次的命中率 / 耗时）：
#        默认(整块 1552)  241 tok → 0.0%  (1.31s，无收益)
#                        1121 tok → 0.0%  (3.30s，无收益)   ← 门限下方全灭
#                        4465 tok → 69.5% (14.37s → 5.04s)
#        --prefix-match-unit 16
#                         241 tok → 99.6% (1.31s → 0.26s)
#                        1121 tok → 99.9% (3.30s → 0.27s)
#                        4465 tok → 100%  (14.37s → 0.28s)  ← 51× 提速
#      正确性：针（取件码）放中段/末段 × 两种长度，冷跑与命中跑答案一致且都正确。
#      ⚠ 代价：每块的哈希条目数 ≈ 1552/16 = 97 倍（只影响匹配粒度，不增加状态存储频率）。
PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-16}"
case "$PREFIX_MATCH_UNIT" in
  ""|none|off|auto|default) PREFIX_MATCH_UNIT="" ;;
esac
PREFIX_ARGS=""
if [ -n "$PREFIX_MATCH_UNIT" ]; then
  PREFIX_ARGS=" --prefix-match-unit $PREFIX_MATCH_UNIT"
fi

# ---------- KV 池按字节定点（比 gmu 可控）---------------------------------------
# ⚠ flag 真名带 `-bytes`：`--kv-cache-memory-bytes`（**vLLM 自己的警告日志把它简写成
#   `--kv-cache-memory`，照抄会 unknown argument**）。接受 human-readable 整数（3G/3300M…）。
# 语义：**一旦设置就忽略 GPU_MEM_UTIL**；且**跳过 memory profiling**，直接按此字节数预留
#   （gpu_worker.py:489-510，日志会打 "reserved X GiB ... skipped memory profiling"）。**每卡**。
#   取值来源：vLLM 自己在日志里给的两个建议值 ——
#     `--kv-cache-memory=<A>` (fit into requested memory) / `--kv-cache-memory=<B>` (fully utilize gpu)
#   A = requested_memory − (consumed + peak_activation + CUDAGraph) − 150 MiB
#
# 默认 3389653545（= 3.16 GiB）= 本机 gmu 0.97 档的 "fit into requested memory" 值。
#   为什么要点定：profiler 在 gmu 0.97 下给出 3.67 GiB 池（超名义预算 ~0.5 GiB，vLLM 会 warn），
#   使运行期留给 activation 的余量只剩 ~1.44 GiB，**小于 profile 期实测的峰值 activation 1.68 GiB**
#   ⇒ 长 prefill 有真实 OOM 风险。定点到 3.16 GiB 后余量 ≈1.95 GiB > 1.68 GiB。
#   代价（实测）：GPU KV cache size 218,933 → 188,686 tokens（−14%），对 128K 仍是 **1.44×**。
#   ⚠ 该字节数是从「本机本次 free memory 15.63 GiB」算出的：换了卡/加了共租进程/改了 MAX_LEN
#     或 GPU_MEM_UTIL，都应重新看日志里的建议值再改。若日志说 OOM，按它给的差值调整。
#   ⚠ 缩池子会减少 num_blocks，可能触发 Mamba 块门禁（需 num_blocks ≥ MAX_NUM_SEQS=48）⇒ 改完要实测
#     （本档已实测通过：max_seqs=48 + 默认 capture 全量正常启动）。
#   退回 gmu 自动定池：KV_CACHE_MEM_BYTES=none（或 auto/off/空）。
KV_CACHE_MEM_BYTES="${KV_CACHE_MEM_BYTES:-3389653545}"
case "$KV_CACHE_MEM_BYTES" in none|off|auto|default) KV_CACHE_MEM_BYTES="" ;; esac
KV_MEM_ARGS=""
if [ -n "$KV_CACHE_MEM_BYTES" ]; then
  KV_MEM_ARGS=" --kv-cache-memory-bytes $KV_CACHE_MEM_BYTES"
fi

# ---------- KV offload / 多级存储（可选：复用 KV，少做重复 prefill）--------------
# 单级（只加 CPU/盘一层）：
#   KV_OFFLOAD_GB=16 ./start-vllm.sh
#   KV_OFFLOAD_GB=64 KV_OFFLOAD_BACKEND=disk KV_OFFLOAD_DISK_PATH=/logs/kv ./start-vllm.sh
# 多级（GPU → CPU 内存 → SSD，推荐）：
#   KV_OFFLOAD_GB=16 KV_TIER_DISK_DIR=/logs/kv-tier ./start-vllm.sh
#
# ⚠ 前提与限制：
#   * 必须开前缀缓存（V1 默认就开，本脚本不动它）；
#   * KV_OFFLOAD_GB 是 CPU 层**总量**，按 world_size 均分给每张卡（TP2 ⇒ 16 就是每卡 8）；
#   * **多级模式下 CPU 层实际落在 /dev/shm（tmpfs）**，容器带 --ipc=host 时上限 ≈ 内存一半
#     （本机 23.5 GiB，实测空闲 23.3 GiB），所以 KV_OFFLOAD_GB 别超过 ~16；
#   * SSD 层（fs）**没有容量上限**，文件只增不减 ⇒ 要自己盯盘，建议单独目录方便清理；
#   * 只有声明了 SupportsHMA 的连接器才能在混合模型（本模型 = GDN + full attn）上启动，
#     SimpleCPUOffloadConnector 与 OffloadingConnector 都满足。
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-cpu}"
KV_OFFLOAD_DISK_PATH="${KV_OFFLOAD_DISK_PATH:-}"
KV_OFFLOAD_DISK_GB="${KV_OFFLOAD_DISK_GB:-100}"
KV_OFFLOAD_LAZY="${KV_OFFLOAD_LAZY:-1}"
KV_OFFLOAD_POLICY="${KV_OFFLOAD_POLICY:-lru}"
KV_TIER_DISK_DIR="${KV_TIER_DISK_DIR:-}"
KV_TIER_IO_THREADS="${KV_TIER_IO_THREADS:-16}"

KV_ARGS=""
KV_MODE="off"
if [ -n "$KV_OFFLOAD_GB" ]; then
  _bytes=$(( KV_OFFLOAD_GB * 1024 * 1024 * 1024 ))
  if [ -n "$KV_TIER_DISK_DIR" ]; then
    # 多级：OffloadingConnector + TieringOffloadingSpec（CPU 主级 + fs 次级）
    [ "$KV_OFFLOAD_BACKEND" = cpu ] || echo "ⓘ 多级模式下忽略 KV_OFFLOAD_BACKEND，盘层由 KV_TIER_DISK_DIR 决定"
    docker exec "$CONTAINER" mkdir -p "$KV_TIER_DISK_DIR" 2>/dev/null \
      || echo "ⓘ 无法在容器内创建 $KV_TIER_DISK_DIR（请确认它落在已挂载的卷里）"
    _tiers="[{\"type\":\"fs\",\"root_dir\":\"$KV_TIER_DISK_DIR\",\"n_read_threads\":$KV_TIER_IO_THREADS,\"n_write_threads\":$KV_TIER_IO_THREADS}]"
    _extra="\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$_bytes"
    _extra="$_extra,\"eviction_policy\":\"$KV_OFFLOAD_POLICY\",\"secondary_tiers\":$_tiers"
    _conn="OffloadingConnector"
    KV_MODE="cpu+disk($KV_TIER_DISK_DIR)"
  else
    # 单级：SimpleCPUOffloadConnector（cpu 或 disk，二选一）
    case "$KV_OFFLOAD_LAZY" in 0|false|no|off) _lazy=false;; *) _lazy=true;; esac
    _extra="\"cpu_bytes_to_use\":$_bytes,\"kv_offload_backend\":\"$KV_OFFLOAD_BACKEND\",\"lazy_offload\":$_lazy"
    if [ "$KV_OFFLOAD_BACKEND" = disk ]; then
      [ -n "$KV_OFFLOAD_DISK_PATH" ] || { echo "❌ KV_OFFLOAD_BACKEND=disk 需要 KV_OFFLOAD_DISK_PATH"; exit 1; }
      docker exec "$CONTAINER" mkdir -p "$KV_OFFLOAD_DISK_PATH" 2>/dev/null || true
      _extra="$_extra,\"disk_path\":\"$KV_OFFLOAD_DISK_PATH\""
      _extra="$_extra,\"disk_capacity_bytes\":$(( KV_OFFLOAD_DISK_GB * 1024 * 1024 * 1024 ))"
    fi
    _conn="SimpleCPUOffloadConnector"
    KV_MODE="$KV_OFFLOAD_BACKEND"
  fi
  _kvjson="{\"kv_connector\":\"$_conn\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$_extra}}"
  KV_ARGS=" --kv-transfer-config $(printf '%q' "$_kvjson")"
fi

# ---------- 硬护栏：offload 与 per-token-head 量化**不能同开** ------------------
# 这不是性能问题，是**静默正确性**问题：
#   per-token-head 量化把 scale 内联在 KV cell 的 padding 尾部（隐含假设：该层
#   K/V 在显存里连续）；offload 搬运用**跨层共享缓冲** ⇒ 两者的 stride/布局假设
#   不一致 ⇒ scale 会错位、静默覆盖相邻层的 K/V。
#   症状：不崩、不 NaN、不报错，只是"流利而自信的错答案"，而且首个 forward 就中招。
# 所以这里直接拒绝启动，而不是留给人去 debug 一个看起来正常的服务。
# 要做 offload：先把 KV_CACHE_DTYPE 退回 fp16（KV_CACHE_DTYPE=none）。
case "$KV_CACHE_DTYPE" in
  *per_token_head*)
    if [ -n "$KV_OFFLOAD_GB" ]; then
      echo "❌ KV_OFFLOAD_GB=$KV_OFFLOAD_GB 与 KV_CACHE_DTYPE=$KV_CACHE_DTYPE 不能同开。"
      echo "   原因：per-token-head 的 scale 内联在 KV cell padding 尾部，而 offload 用跨层"
      echo "   共享缓冲 ⇒ stride 错位、静默覆盖相邻 K/V。不崩不 NaN，只会给出错误的答案。"
      echo "   要做 offload：KV_CACHE_DTYPE=none（fp16 KV）再起。"
      exit 1
    fi
    ;;
esac

# ---------- API key ----------------------------------------------------------
if [ -n "${API_KEY:-}" ]; then
  KEY="$API_KEY"
elif [ -s "$KEY_FILE" ]; then
  KEY="$(cat "$KEY_FILE")"
else
  KEY="$(openssl rand -hex 24 2>/dev/null || od -An -tx1 -N24 /dev/urandom | tr -d ' \n')"
  ( umask 077; printf '%s\n' "$KEY" > "$KEY_FILE" )
  chmod 600 "$KEY_FILE"
  echo "🔑 已生成新的 API key 并存到 $KEY_FILE"
fi

# ---------- 前置检查 ---------------------------------------------------------
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "❌ 容器不存在: $CONTAINER"
  echo "   先创建（示例）: docker run -d --name $CONTAINER \\"
  echo "       --device=/dev/kfd --device=/dev/dri --group-add video \\"
  echo "       --ipc=host --shm-size=16g --security-opt seccomp=unconfined \\"
  echo "       --restart unless-stopped -p $PORT:$PORT \\"
  echo "       -v /home/x/models:/models \\"
  echo "       -v /home/x/vllm-cache/vllm-w4a16-exllama:/root/.cache/vllm-w4a16-exllama \\"
  echo "       -v /home/x/logs:/logs \\"
  echo "       -e HSA_OVERRIDE_GFX_VERSION=9.0.6 -e VLLM_TRITON_ATTN_NUM_SEGMENTS=64 \\"
  echo "       -e VLLM_WNA16_EXLLAMA=1 -e PYTORCH_ROCM_ARCH=gfx906 -e VLLM_TARGET_DEVICE=rocm \\"
  echo "       vllm-gfx906:latest sleep infinity"
  echo "   注意 -p $PORT:$PORT 不能省，否则容器内起来了局域网也访问不到"
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "🔄 容器未运行，正在启动: $CONTAINER"
  docker start "$CONTAINER" >/dev/null
  sleep 2
fi

# 带 key 探测（vLLM 的 /v1/models 需要鉴权，不带 key 会 401，
# 用 /health 判断会把"已经在跑"误判成"没在跑"然后撞端口）
if [ "${DRY_RUN:-0}" != 1 ] && docker exec "$CONTAINER" curl -sf --max-time 3 -o /dev/null \
     -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "⚠️  端口 $PORT 上已有服务在跑"
  echo "   查看模型: curl -s -H 'Authorization: Bearer $KEY' http://localhost:$PORT/v1/models"
  echo "   先停掉:   ./stop-vllm.sh"
  exit 1
fi

mkdir -p "$LOG_DIR"

# ---------- 组命令并后台启动 --------------------------------------------------
# ⚠ 位置纪律：`VAR=val ...` 之后第一个非赋值词会被 bash 当成**命令名**。
#   所以所有 `--xxx` 形式的参数必须拼在 `gfx906-serve` **之后**；
#   拼在前面会出现 `bash: line 1: --kv-transfer-config: command not found`。
#   （KV_QUANT/PREFIX/KV_MEM 一直是拼在后面的，只有 KV_ARGS 曾拼错位置。）
inner="HOST=$HOST_BIND PORT=$PORT MAX_LEN=$MAX_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
inner="$inner GPU_MEM_UTIL=$GPU_MEM_UTIL TP=$TP DTYPE=$DTYPE EAGER=$EAGER"
inner="$inner$OPT_ENVS"
inner="$inner gfx906-serve $(printf '%q' "$MODEL") --api-key $(printf '%q' "$KEY")"
for a in "$@"; do inner="$inner $(printf '%q' "$a")"; done
inner="$inner$KV_ARGS"
inner="$inner$KV_QUANT_ARGS"
inner="$inner$PREFIX_ARGS"
inner="$inner$KV_MEM_ARGS"
inner="$inner > $LOG_CTR 2>&1"

echo "🚀 启动 vLLM..."
echo "   容器:   $CONTAINER"
echo "   模型:   $MODEL"
echo "   监听:   $HOST_BIND:$PORT   (TP=$TP, dtype=$DTYPE, max_len=$MAX_LEN, max_seqs=$MAX_NUM_SEQS, gmu=$GPU_MEM_UTIL)"
echo "   KV 量化: ${KV_CACHE_DTYPE:-off（fp16 per-tensor）}"
if [ -n "$PREFIX_MATCH_UNIT" ]; then
  echo "   前缀粒度: ${PREFIX_MATCH_UNIT} tokens（默认是整块 1552）"
else
  echo "   前缀粒度: 默认（各 group 块大小的 GCD；本模型实测 = 整块 1552）"
fi
if [ -n "$KV_CACHE_MEM_BYTES" ]; then
  echo "   KV 池:   ${KV_CACHE_MEM_BYTES} bytes（已定点 ⇒ 忽略 gmu=$GPU_MEM_UTIL）"
fi
if [ "$KV_MODE" = off ]; then
  echo "   KV复用: 仅 GPU 内前缀缓存（要多级复用加 KV_OFFLOAD_GB=16 KV_TIER_DISK_DIR=/logs/kv-tier）"
else
  echo "   KV复用: $KV_MODE 模式 —— CPU 层 ${KV_OFFLOAD_GB}GiB（每卡 $(( KV_OFFLOAD_GB / TP ))GiB）"
  echo "           连接器: $([ "${KV_TIER_DISK_DIR}" != "" ] && echo OffloadingConnector+TieringOffloadingSpec || echo SimpleCPUOffloadConnector)"
fi
echo "   日志:   $LOG_HOST"
echo "   提示:   缓存命中约 7-8 分钟；冷编译更久（实测命中 AOT 缓存 447s，"
echo "           其中 torch.compile 67s，其余是权重加载 + KV/Mamba cache profile）"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo
  echo "🔎 DRY_RUN=1 —— 只打印将执行的命令，不启动："
  echo "$inner"
  exit 0
fi

docker exec -d "$CONTAINER" bash -lc "$inner"

# ---------- 等就绪 -----------------------------------------------------------
echo "⏳ 等待服务就绪（最多 ${READY_TIMEOUT}s）..."
start=$(date +%s)
last_note=0
while :; do
  code=$(docker exec "$CONTAINER" curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
         -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    elapsed=$(( $(date +%s) - start ))
    echo "✅ 服务就绪（${elapsed}s）"
    echo
    echo "   Base URL : http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1"
    echo "   API Key  : $KEY"
    echo
    echo "   试一下:"
    echo "     curl -s http://localhost:$PORT/v1/chat/completions \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -H 'Authorization: Bearer $KEY' \\"
    echo "       -d '{\"model\":\"$(basename "$MODEL")\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":64}'"
    exit 0
  fi

  # 进程没了 = 启动失败。
  # 必须排除僵尸：容器 pid 1 是 `sleep infinity`，不回收孤儿进程，
  # 崩掉的 worker 会以 <defunct> 留在进程表，pgrep 仍能匹配到 ⇒ 误判"还活着"
  # 然后一直空等到超时。所以读 stat 状态列，首字符 Z 的算死的。
  if [ "$(docker exec "$CONTAINER" ps -eo stat=,args= 2>/dev/null \
         | awk 'substr($1,1,1)!="Z" && /vllm serve/ {n++} END{print n+0}')" -eq 0 ]; then
    echo "❌ 服务进程已退出，最后 20 行日志："
    tail -n 20 "$LOG_HOST" 2>/dev/null || docker exec "$CONTAINER" tail -n 20 "$LOG_CTR"
    echo
    echo "   ⓘ 若日志里是 OOM / 显存不足，试试调小 MAX_NUM_SEQS 或 MAX_LEN"
    exit 1
  fi

  now=$(( $(date +%s) - start ))
  if [ $(( now - last_note )) -ge 30 ]; then
    msg=$(tail -n 1 "$LOG_HOST" 2>/dev/null | cut -c1-100)
    echo "   ...${now}s  $msg"
    last_note=$now
  fi
  if [ "$now" -ge "$READY_TIMEOUT" ]; then
    echo "❌ 等待超时（${READY_TIMEOUT}s），检查日志: tail -f $LOG_HOST"
    exit 1
  fi
  sleep 5
done
