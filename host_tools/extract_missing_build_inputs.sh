#!/usr/bin/env bash
# 只读：从镜像抽取 build-context 缺失的输入，并验证可再生性（v2：修容器名）
set -u
IMG=vllm-gfx906:latest
SP=/usr/local/lib/python3.12/dist-packages
TMPL=/tmp/rocblas_lib_check
OUT=/home/x/_rebuild_check
C=tmp-extract
mkdir -p "$OUT"
rm -rf "$TMPL"

echo "=== 0) 建一个不运行的容器用于 docker cp ==="
docker rm -f "$C" >/dev/null 2>&1 || true
docker create --name "$C" "$IMG" >/dev/null && echo "created $C"

echo
echo "=== 1) 抽取 3 个「仓库里没有」的补丁源文件 ==="
extract() {
  if docker cp "$C:$1" "$OUT/$2" 2>/dev/null; then
    printf '  OK   %8s  %s\n' "$(stat -c%s "$OUT/$2")" "$2"
  else
    echo "  FAIL $2"
  fi
}
extract "$SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py" wna16_exllama.py
extract "$SP/vllm/model_executor/kernels/linear/__init__.py"                      linear__init__.py
extract "$SP/vllm/v1/attention/ops/triton_decode_attention.py"                    triton_decode_attention.py
extract "$SP/torch/lib/librccl.so.1.0"                                           librccl.so.1.0

echo
echo "=== 2) Tensile gfx906：镜像 rocblas/library vs 主机 backup-rocm-tensors ==="
docker cp "$C:$SP/torch/lib/rocblas/library/." "$TMPL" >/dev/null 2>&1
ls "$TMPL" 2>/dev/null | grep gfx906 | sort > "$OUT/from_image.txt"
echo "  镜像 gfx906 文件数 : $(wc -l < "$OUT/from_image.txt")"
ls /home/x/backup-rocm-tensors | grep gfx906 | sort > "$OUT/from_host.txt"
echo "  主机 gfx906 文件数 : $(wc -l < "$OUT/from_host.txt")"
echo "  --- 文件名 diff（空 = 完全一致）---"
if diff -q "$OUT/from_image.txt" "$OUT/from_host.txt" >/dev/null; then
  echo "  ✅ 156 个文件名完全一致"
else
  echo "  ⚠ 有差异："; diff "$OUT/from_image.txt" "$OUT/from_host.txt" | head -20
fi
echo "  --- 全量字节大小 + md5 一致性 ---"
same=0; diffn=0; list=""
while read -r f; do
  a=$(stat -c%s "/home/x/backup-rocm-tensors/$f" 2>/dev/null || echo -1)
  b=$(stat -c%s "$TMPL/$f" 2>/dev/null || echo -2)
  if [ "$a" = "$b" ]; then same=$((same+1)); else diffn=$((diffn+1)); list="$list $f"; fi
done < "$OUT/from_host.txt"
echo "  字节一致: SAME=$same  DIFF=$diffn"
[ -n "$list" ] && { echo "  不一致清单:"; for f in $list; do echo "    $f"; done; }
echo "  --- md5 抽样 5 个 ---"
head -5 "$OUT/from_host.txt" | while read -r f; do
  A=$(md5sum "/home/x/backup-rocm-tensors/$f" 2>/dev/null | cut -d' ' -f1)
  B=$(md5sum "$TMPL/$f" 2>/dev/null | cut -d' ' -f1)
  printf '   %-58s %s\n' "${f:0:58}" "$([ "$A" = "$B" ] && echo MD5-SAME || echo "MD5-DIFF")"
done

echo
echo "=== 3) RCCL 对比 ==="
if [ -f "$OUT/librccl.so.1.0" ]; then
  echo -n "  镜像 md5: "; md5sum "$OUT/librccl.so.1.0" | cut -d' ' -f1
else
  echo "  镜像抽取失败"
fi
[ -f /home/x/rccl-gfx906-2.27.7-rocm7.2.4/librccl.so.1.0 ] && {
  echo -n "  主机 md5: "; md5sum /home/x/rccl-gfx906-2.27.7-rocm7.2.4/librccl.so.1.0 | cut -d' ' -f1; }

echo
echo "=== 4) 镜像里是否也有 gfx906-supplements 的等价物（[2/7] 回退可行性）==="
echo "  镜像 $SP/torch/lib/rocblas/library 里 gfx906 文件数 = $(wc -l < "$OUT/from_image.txt")"
echo "  → 若与主机 156 一致，则 [2/7] 的回退可以改成「从镜像 rocblas/library 抽 gfx906」"

echo
echo "=== 5) 清理 ==="
docker rm -f "$C" >/dev/null && echo "removed $C"
rm -rf "$TMPL"

echo
echo "=== 6) 结果 ==="
ls -l "$OUT"

echo
echo "=== 7) docker save 进度 ==="
ls -l /home/x/images/ 2>/dev/null
ps -eo pid,args | grep -c 'docker save' || true
