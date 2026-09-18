#!/usr/bin/env bash
# 实测修复后的 [2/7] 回退：从镜像 rocblas library 抽取 156 个 gfx906 文件
set -uo pipefail
IMG=vllm-gfx906:latest
SP=/usr/local/lib/python3.12/dist-packages
C=tmp-supptest
OUT=/tmp/supp_test_$$
mkdir -p "$OUT"

docker rm -f "$C" >/dev/null 2>&1 || true
docker run -d --name "$C" --entrypoint bash "$IMG" -c 'sleep 600' >/dev/null
echo "container $C started (no GPU requested)"

echo
echo "=== 模拟修复后的 [2/7] 回退分支（SUPPLEMENT_SRC 不存在）==="
docker exec "$C" bash -lc \
  "cd $SP/torch/lib/rocblas/library && ls | grep gfx906 | tar -cf /tmp/gfx906_supp.tar -T -"
echo "  tar rc=$?"
docker cp "$C:/tmp/gfx906_supp.tar" "$OUT/_supp.tar"
tar -xf "$OUT/_supp.tar" -C "$OUT"
rm -f "$OUT/_supp.tar"
n=$(ls "$OUT" | wc -l)
echo "  抽到文件数: $n  （期望 156）"
if [ "$n" -ne 156 ]; then echo "  ❌ FAIL"; docker rm -f "$C" >/dev/null; exit 1; fi
ls "$OUT" | grep -q 'TensileLibrary_lazy_gfx906.dat' && echo "  ✅ 关键文件 TensileLibrary_lazy_gfx906.dat 在"

echo
echo "=== 与镜像内原文件逐字节比对（md5 全量）==="
bad=0
for f in $(ls "$OUT"); do
  a=$(docker exec "$C" md5sum "$SP/torch/lib/rocblas/library/$f" | cut -d' ' -f1)
  b=$(md5sum "$OUT/$f" | cut -d' ' -f1)
  if [ "$a" != "$b" ]; then echo "  DIFF $f"; bad=$((bad+1)); fi
done
echo "  不一致文件数: $bad / 156"

echo
echo "=== 与主机 /home/x/backup-rocm-tensors 比对（若还在）==="
if [ -d /home/x/backup-rocm-tensors ]; then
  bad2=0
  for f in $(ls "$OUT"); do
    if [ -f "/home/x/backup-rocm-tensors/$f" ]; then
      a=$(md5sum "/home/x/backup-rocm-tensors/$f" | cut -d' ' -f1)
      b=$(md5sum "$OUT/$f" | cut -d' ' -f1)
      [ "$a" != "$b" ] && { echo "  DIFF $f"; bad2=$((bad2+1)); }
    else
      echo "  主机缺少 $f"; bad2=$((bad2+1))
    fi
  done
  echo "  不一致: $bad2 / 156"
else
  echo "  (主机目录已不在，跳过)"
fi

echo
echo "=== 清理 ==="
docker rm -f "$C" >/dev/null && echo "removed $C"
rm -rf "$OUT"

echo
echo "=== docker save 进度 ==="
ls -l /home/x/images/ 2>/dev/null
ps -eo pid,etime,args | grep '[d]ocker save' || echo "(docker save 进程已结束)"
