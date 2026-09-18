#!/bin/bash
# stop-vllm.sh -- 优雅停止 vLLM API 服务（默认不动容器，保留编译缓存）
#
# 用法:
#   ./stop-vllm.sh            # 只停服务进程
#   ./stop-vllm.sh --purge    # 连容器一起停掉
#   ./stop-vllm.sh --status   # 只看状态，不动手
#   ./stop-vllm.sh --reap     # 停止后重启容器，清掉 <defunct> 僵尸
#
# 环境变量: CONTAINER
#
# 【为什么不用 pgrep 判活】
# 容器 pid 1 是 `sleep infinity`，它不回收孤儿进程。被 kill 的 EngineCore /
# Worker 会以 `[VLLM::Worker_TP] <defunct>` 形态留在进程表，而 `pgrep -f`
# 照样匹配得到僵尸残留的 cmdline ⇒ 误报"服务还活着"。实测 kill 之后 VRAM
# 已从 14.9GB 掉到 61MB，服务其实死透了。
# 所以本脚本统一用 `ps -eo stat=,pid=,args=` 取状态列，排除首字符为 Z 的行。
set -uo pipefail

CONTAINER="${CONTAINER:-vllm}"
PAT='vllm serve|VLLM::|gfx906-serve'
DO_PURGE=0; DO_STATUS=0; DO_REAP=0
for a in "$@"; do
  case "$a" in
    --purge)  DO_PURGE=1 ;;
    --status) DO_STATUS=1 ;;
    --reap)   DO_REAP=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "❌ 容器不存在: $CONTAINER"
  exit 1
fi

running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }

# 取进程表快照一次，供下面复用格式：STAT PID ELAPSED ARGS
snap() { docker exec "$CONTAINER" ps -eo stat=,pid=,etime=,args= 2>/dev/null; }
# 存活的 vLLM 进程行（排除僵尸）
live_lines() { snap | awk -v pat="$1" 'substr($1,1,1)!="Z" && $0 ~ pat'; }
zomb_lines() { snap | awk -v pat="$1" 'substr($1,1,1)=="Z" && $0 ~ pat'; }
nlines()     { awk 'END{print NR+0}'; }

# 僵尸不占显存，但会让 `ps` 输出越来越脏（每停一次累积一批）。
# --reap / --purge 用重启或停容器一步清干净；否则只提示。
reap_if_needed() {
  local z
  z=$(zomb_lines "$PAT" | nlines)
  [ "$z" -eq 0 ] && return 0
  if [ "$DO_REAP" = 1 ]; then
    echo "🧹 重启容器清掉 $z 个僵尸..."
    docker restart "$CONTAINER" >/dev/null \
      && echo "✅ 僵尸已清（编译缓存在宿主 bind mount 上，不受影响）"
  else
    echo "ⓘ 有僵尸残留 $z 个（无害，不占显存）；./stop-vllm.sh --reap 可清掉"
  fi
}

show_status() {
  if ! running; then
    echo "── 容器 $CONTAINER: $(docker inspect -f '{{.State.Status}}' "$CONTAINER")"
    return 0
  fi
  local n z
  n=$(live_lines "$PAT" | nlines)
  z=$(zomb_lines "$PAT" | nlines)
  echo "── 容器 $CONTAINER: running"
  echo "── vLLM 存活进程数: $n"
  if [ "$n" -gt 0 ]; then
    live_lines "$PAT" | cut -c1-170 | sed 's/^/   /'
  fi
  if [ "$z" -gt 0 ]; then
    echo "── 僵尸残留: $z 个（pid 1 是 sleep infinity，不回收孤儿；不占显存）"
    echo "   ⓘ 清掉：./stop-vllm.sh --reap   或   docker restart $CONTAINER"
  fi
}

if [ "$DO_STATUS" = 1 ]; then
  show_status
  exit 0
fi

if ! running; then
  echo "ℹ️  容器未运行（$CONTAINER），无需停止"
  exit 0
fi

if [ "$(live_lines "$PAT" | nlines)" -eq 0 ]; then
  echo "ℹ️  没有正在运行的 vLLM 服务"
  if [ "$DO_PURGE" = 1 ]; then
    docker stop "$CONTAINER" >/dev/null && echo "✅ 容器已停止"
  else
    reap_if_needed          # ← 服务已停但可能剩僵尸，--reap 必须在这条路径上也生效
  fi
  exit 0
fi

echo "🛑 停止 vLLM..."

# 1) 先温和地请 API server 自己退出（它会带着 worker 一起走）
docker exec "$CONTAINER" pkill -TERM -f 'vllm serve' 2>/dev/null || true
for i in $(seq 1 60); do
  if [ "$(live_lines 'vllm serve' | nlines)" -eq 0 ]; then
    echo "   主进程已退出（${i}s）"
    break
  fi
  sleep 1
done

# 2) 收尾残留的 worker / engine core / 启动壳
for pat in 'VLLM::' 'gfx906-serve' 'vllm serve'; do
  if [ "$(live_lines "$pat" | nlines)" -gt 0 ]; then
    echo "   清理残留 [$pat]"
    docker exec "$CONTAINER" pkill -TERM -f "$pat" 2>/dev/null || true
  fi
done
sleep 3

# 3) 还不走就强制
if [ "$(live_lines "$PAT" | nlines)" -gt 0 ]; then
  echo "⚠️  仍有存活进程，强制终止"
  docker exec "$CONTAINER" pkill -9 -f 'vllm serve'   2>/dev/null || true
  docker exec "$CONTAINER" pkill -9 -f 'VLLM::'       2>/dev/null || true
  docker exec "$CONTAINER" pkill -9 -f 'gfx906-serve' 2>/dev/null || true
  sleep 2
fi

if [ "$(live_lines "$PAT" | nlines)" -gt 0 ]; then
  echo "❌ 仍有存活进程未退出，请手动检查"
  show_status
  exit 1
fi
echo "✅ vLLM 已停止"

if [ "$DO_PURGE" = 1 ]; then
  echo "🛑 停止容器 $CONTAINER ..."
  docker stop "$CONTAINER" >/dev/null \
    && echo "✅ 容器已停止（僵尸随之消失，下次启动仍会命中编译缓存）"
else
  reap_if_needed
fi
