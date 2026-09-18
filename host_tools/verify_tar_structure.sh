#!/usr/bin/env bash
# 只读：校验 docker save tar 的结构完整性（v2：修 manifest 摘要解析）
set -uo pipefail
TAR=/home/x/images/vllm-gfx906-latest-2f6d5bf6.tar
TMP=/tmp/tarcheck_$$
mkdir -p "$TMP"

echo "=== 0) 文件 ==="
ls -l "$TAR"

echo
echo "=== 1) 归档内 blob 清单（只取 64 位十六进制）==="
tar -tf "$TAR" > "$TMP/all.txt"
grep -oE '[0-9a-f]{64}$' "$TMP/all.txt" | sort -u > "$TMP/have.txt"
echo "  tar 条目总数: $(wc -l < "$TMP/all.txt")"
echo "  blob 文件数  : $(wc -l < "$TMP/have.txt")"
echo "  非 blob 条目 :"; grep -vE '[0-9a-f]{64}$' "$TMP/all.txt" | sed 's/^/    /'

echo
echo "=== 2) manifest / index 概要 ==="
tar -xf "$TAR" -C "$TMP" manifest.json index.json 2>/dev/null
python3 - "$TMP" <<'PY'
import json,os,sys
t=sys.argv[1]
raw=open(os.path.join(t,'have.txt')).read().split()
have=set(raw)
m=json.load(open(os.path.join(t,'manifest.json')))
def norm(d):
    d=str(d).split('sha256')[-1].lstrip(':/')
    return d.rsplit('/',1)[-1]
tot=miss=0; missl=[]
for e in m:
    refs=[e['Config']]+e['Layers']
    print('  tag=%s  config=%s  layers=%d (occurrences)'%(e.get('RepoTags'), norm(e['Config'])[:16], len(e['Layers'])))
    for d in refs:
        h=norm(d); tot+=1
        if h not in have: miss+=1; missl.append(h)
print('  引用条目总数(含重复)=%d   唯一缺失=%d'%(tot, len(set(missl))))
for h in sorted(set(missl)): print('    MISS', h)
ok = len(set(missl))==0
print('  RESULT:', 'OK' if ok else 'INCOMPLETE')
sys.exit(0 if ok else 1)
PY
rc=$?

echo
echo "=== 3) 0 字节 blob 检查（只看真 blob）==="
zero=0
while read -r b; do
  s=$(tar -tvf "$TAR" "blobs/sha256/$b" 2>/dev/null | awk 'NR==1{print $3}')
  if [ "${s:-x}" = "0" ]; then echo "  ZERO $b"; zero=$((zero+1)); fi
done < "$TMP/have.txt"
echo "  0 字节 blob: $zero"

echo
echo "=== 4) 关键 blob 是否在档 ==="
for k in 2f6d5bf6b47e56654b16fb8839df806fa81bff41597df90de10be8bc4e426d61 \
         caa9c817aa9abc6953d83178526d807b56a084cd6f1bbed5909be834d6f153c3 \
         4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1; do
  printf '  %s %s\n' "$(grep -qx "$k" "$TMP/have.txt" && echo IN || echo MISS)" "${k:0:24}…"
done

echo
echo "=== 5) 结论 ==="
if [ "$rc" = 0 ] && [ "$zero" = 0 ]; then
  echo "  ✅ tar 结构完整：manifest 引用的全部唯一 blob 在档，无 0 字节 blob"
else
  echo "  ❌ 有问题（manifest rc=$rc / zero=$zero）—— 先不要删任何东西"
fi

rm -rf "$TMP"
