set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [1] config/vllm.py：设了 kv_transfer_config 是否自动关 HMA ==="
grep -n -B8 -A12 "need_disable_hybrid_kv_cache_manager" "$V/config/vllm.py" | head -50

echo
echo "=== [2] factory.py：supports_hma 门（HMA 开 + 连接器不支持 ⇒ raise）==="
sed -n '40,80p' "$V/distributed/kv_transfer/kv_connector/factory.py"

echo
echo "=== [3] 谁声明了 SupportsHMA ==="
grep -rn --include=*.py "class .*SupportsHMA\|def supports_hma" "$V/distributed/kv_transfer/kv_connector/" "$V/v1/kv_offload/" 2>/dev/null | head -20

echo
echo "=== [4] OffloadingConnector / SimpleCPUOffloadConnector 的声明 ==="
grep -rn --include=*.py -A4 "class OffloadingConnector\|class SimpleCPUOffloadConnector" "$V" | head -30

echo
echo "=== [5] fs 次级层：索引是否落盘（决定能否跨重启复用）==="
ls "$V/v1/kv_offload/tiering/fs/" 2>/dev/null
grep -n -A12 "class .*FileMapper\|def __init__" "$V/v1/kv_offload/file_mapper.py" 2>/dev/null | head -40
