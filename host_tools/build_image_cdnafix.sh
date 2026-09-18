#!/bin/bash
# 交付镜像重建（P0.6：gfx906 不再被判为 CDNA + fp8 KV 显式拒绝）
#
#   nohup bash /home/x/build_image_cdnafix.sh > /home/x/logs/image_build_20260917.log 2>&1 &
#
# 为什么要在 nohup 里跑：18 GB 的 dist-packages 重打 + 两段 build 合起来远超
# 任何 ssh 会话的存活时间；一旦 ssh 断开就必须能在主机侧继续跑完。
#
# 关键点：
#   * prepare_build_context.sh 的第 [7/7] 步在 tar 已存在时会**静默跳过重打**
#     ⇒ 必须 FORCE=1，否则新补丁根本进不了 build-context。
#   * Path B 的 context 是 docker/build-context/，Path A 的 context 是 docker/
#     （两条命令都从 docker/ 目录执行，见 Dockerfile 头部注释）。
#   * 先给现有 :latest 打一个回滚 tag，再动 :latest。
set -uo pipefail
cd /home/x/vllm-work/docker || exit 1

PAYLOAD="$PWD/patches/platforms_rocm.py"
WANT_MD5="221a70134a8d22f9d411cc0f7871dc10"
SP=/usr/local/lib/python3.12/dist-packages

echo "=== [0/6] 回滚锚点 ==="
docker tag vllm-gfx906:latest vllm-gfx906:pre-cdnafix-20260917 2>/dev/null \
  && echo "tagged: vllm-gfx906:pre-cdnafix-20260917" \
  || echo "(回滚 tag 已存在，跳过)"
docker images | grep -E "vllm-gfx906" | head -5

echo
echo "=== [1/6] repo payload 校验 ==="
md5sum "$PAYLOAD"
if [ "$(md5sum "$PAYLOAD" | cut -d' ' -f1)" != "$WANT_MD5" ]; then
  echo "!! payload md5 与已验证版本不符，中止"; exit 1
fi

echo
echo "=== [2/6] 重打 build-context（FORCE=1 破 tar 复用陷阱）==="
FORCE=1 bash prepare_build_context.sh vllm || { echo "!! prepare 失败"; exit 1; }

echo
echo "=== [3/6] 校验 build-context 里的 payload ==="
BC="$PWD/build-context/patches/platforms_rocm.py"
md5sum "$BC" || { echo "!! build-context 里没有 payload"; exit 1; }
grep -q '(not _ON_GFX906) and any(' "$BC" || { echo "!! payload 不含 _ON_CDNA 修复"; exit 1; }
grep -q 'ROCm FP8 posture' "$BC" || { echo "!! payload 不含 fp8 守卫"; exit 1; }
echo "payload OK"

echo
echo "=== [4/6] Path B: rebuild stage（authoritative，离线重建）==="
docker build -f Dockerfile.rocm_gfx906 --target rebuild \
  -t vllm-gfx906:rebuilt build-context/ || { echo "!! Path B 失败"; exit 1; }

echo
echo "=== [5/6] Path A: 默认 target -> :latest ==="
docker build -f Dockerfile.rocm_gfx906 -t vllm-gfx906:latest . \
  || { echo "!! Path A 失败"; exit 1; }

echo
echo "=== [6/6] 验收：新镜像内 rocm.py 应与 payload 逐字节一致 ==="
# 只做静态验收：这里**不能** import vllm.platforms.rocm，因为 rocm.py 在 import
# 期就会 torch.cuda.get_device_properties("cuda")，而这个 docker run 没有传
# --device=/dev/kfd，只会得到一个与补丁无关的报错。运行时验收在真机上做
# （已由活体容器的 posture 日志 + fp8 拒绝 + 三项回归完成）。
docker run --rm --entrypoint bash vllm-gfx906:latest -lc \
  "md5sum $SP/vllm/platforms/rocm.py; \
   echo -n 'cdna-fix hits: '; grep -c 'not _ON_GFX906) and any(' $SP/vllm/platforms/rocm.py; \
   echo -n 'fp8-guard hits: '; grep -c 'ROCm FP8 posture' $SP/vllm/platforms/rocm.py; \
   echo -n 'cache empty: '; ls -A /root/.cache 2>/dev/null | wc -l"

echo
echo "=== 构建结束 ==="
docker images | grep -E "vllm-gfx906"
