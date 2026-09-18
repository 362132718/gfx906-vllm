set -u
P=/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py
echo "=== md5 现状 ==="
md5sum "$P"; ls -l "$P".orig-* 2>/dev/null

echo
echo "=== [补丁 1 落点 236-252] ==="
awk 'NR>=236 && NR<=252 {printf "%4d| %s\n", NR, $0}' "$P"

echo
echo "=== [补丁 2 落点 928-975] ==="
awk 'NR>=928 && NR<=975 {printf "%4d| %s\n", NR, $0}' "$P"

echo
echo "=== [自检：两处补丁都在，且旧写法已消失] ==="
grep -c 'not _ON_GFX906) and any(' "$P"
grep -c 'ROCm FP8 posture' "$P"
echo "旧 _ON_CDNA 残留行数（应为 0）:"
grep -c '^_ON_CDNA = any(arch in _GCN_ARCH' "$P" || true

echo
echo "=== [生成 unified diff 供 patch ledger] ==="
B=$(ls -t "$P".orig-* | head -1)
diff -u "$B" "$P" || true
