set -u
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [1] supports_hma_config 怎么判 ==="
sed -n '125,150p' "$V/distributed/kv_transfer/kv_connector/factory.py"

echo
echo "=== [2] fs 次级层：启动时是否扫描 root_dir 重建索引（决定跨重启能否命中）==="
grep -n -E "scandir|listdir|glob|os.walk|iterdir|prefix|suffix|\.kv|def __init__|def _load|_index" \
  "$V/v1/kv_offload/tiering/fs/manager.py" | head -30
echo "--- FileMapper 的文件名构造 ---"
grep -n -A18 "def file_name\|def _file_name\|def name_for\|def get_file" "$V/v1/kv_offload/file_mapper.py" | head -40

echo
echo "=== [3] offload 的 prometheus 指标名（A/B 要靠它读命中）==="
grep -rn --include=*.py -E "\"vllm:.*offload|kv_offload|prefix_cache_hits|gauge|counter" \
  "$V/v1/kv_offload/cpu/common.py" "$V/v1/kv_offload/tiering/metrics.py" 2>/dev/null | head -25
echo "--- 在 metrics/logger 里找已注册的 offload 指标 ---"
grep -rn --include=*.py "offload" "$V/v1/metrics/loggers.py" | head -20

echo
echo "=== [4] 多 group（hybrid）模型：block_size vs blocks_per_chunk 的校验 ==="
grep -rn --include=*.py -B4 -A12 "blocks_per_chunk" "$V/v1/kv_offload/config.py" 2>/dev/null | head -45
