#!/usr/bin/env bash
# p8_verify_tar.sh -- prove the delivery tar is SELF-CONTAINED, i.e. it will
# load on a machine that has none of our blobs.
#
# Why this and not another `docker load`: the earlier load test ran on the same
# host, where containerd already held every blob -- it could have re-tagged from
# local state instead of importing the archive. Here we instead check that every
# blob referenced by the archive's own manifest.json is physically present in the
# tar. If they all are, the archive cannot be missing anything.
set -uo pipefail

F=${1:-/home/x/vllm-gfx906-c31a8b65.tar}
W=/tmp/tarverify
mkdir -p "$W"

echo "=== 1) 归档的 manifest.json ==="
tar -xOf "$F" manifest.json > "$W/manifest.json" 2>/dev/null || { echo "  无法读取 manifest"; exit 1; }
head -c 200 "$W/manifest.json"; echo; echo

echo "=== 2) manifest 引用了多少个 blob ==="
python3 - <<'PY' > "$W/needed.txt"
import json
m = json.load(open("/tmp/tarverify/manifest.json"))[0]
need = [m["Config"]] + list(m["Layers"])
for b in need:
    print(b)
PY
wc -l < "$W/needed.txt" | awk '{print "  需要 " $1 " 个 blob（config + layers）"}'
echo

echo "=== 3) 归档内容清单（一次扫完 22 GB）==="
tar -tf "$F" > "$W/tarlist.txt" 2>/dev/null
wc -l < "$W/tarlist.txt" | awk '{print "  归档内条目 " $1 " 个"}'
echo

echo "=== 4) 逐个核对 ==="
ok=0; miss=0
while read -r b; do
  if grep -qxF "$b" "$W/tarlist.txt"; then ok=$((ok+1)); else miss=$((miss+1)); echo "  MISSING: $b"; fi
done < "$W/needed.txt"
echo "  在档: $ok   缺失: $miss"
echo
if [ "$miss" = 0 ]; then
  echo "VERDICT=SELF_CONTAINED  （对端无需任何本机 blob，直接 docker load 即可）"
else
  echo "VERDICT=INCOMPLETE"
fi
echo
echo "=== 5) 归档里的 RepoTags ==="
python3 -c "import json;print(' ', json.load(open('/tmp/tarverify/manifest.json'))[0]['RepoTags'])"
ls -la "$F" | awk '{printf "  %.2f GB  %s\n", $5/1e9, $9}'
