set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [1] mmap 在 offload 路径里的使用点（含 /dev/shm 默认）==="
grep -rn --include=*.py -E "mmap|/dev/shm|shm_open|SharedMemory|shm_dir" \
  "$V/v1/kv_offload/" "$V/v1/simple_kv_offload/" 2>/dev/null | head -30

echo
echo "=== [2] shared_offload_region.py 的形态 ==="
grep -n -B3 -A18 "def __init__\|mmap" "$V/v1/kv_offload/cpu/shared_offload_region.py" 2>/dev/null | head -45

echo
echo "=== [3] TieringOffloadingSpec：CPU 主级用什么承载体 ==="
grep -n -B4 -A22 "class TieringOffloadingSpec" "$V/v1/kv_offload/tiering/spec.py" 2>/dev/null | head -50

echo
echo "=== [4] 次级层 fs 的支持 key 与是否校验 root_dir ==="
grep -rn --include=*.py -E "root_dir|n_read_threads|n_write_threads|type.*fs|SUPPORTED" \
  "$V/v1/kv_offload/tiering/fs/__init__.py" "$V/v1/kv_offload/tiering/factory.py" 2>/dev/null | head -25

echo
echo "=== [5] CPUOffloadingSpec 与 TieringOffloadingSpec 的注册表 ==="
grep -rn --include=*.py -E "CPUOffloadingSpec|TieringOffloadingSpec" "$V/v1/kv_offload/factory.py" 2>/dev/null | head -15
