set -u
P=/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py
V=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [RocmPlatform.check_and_update_config 开头 50 行] ==="
grep -n -A50 "def check_and_update_config" "$P" | head -60

echo
echo "=== [interface.py 里 check_and_update_config 的签名] ==="
grep -n -B4 -A12 "def check_and_update_config" "$V/platforms/interface.py" | head -40

echo
echo "=== [CacheConfig.kv_cache_dtype 字段与校验实现] ==="
sed -n '265,300p' "$V/config/cache.py" | cat -n

echo
echo "=== [triton_reshape_and_cache_flash.py 1-40] ==="
sed -n '1,40p' "$V/v1/attention/ops/triton_reshape_and_cache_flash.py" | cat -n

echo
echo "=== [backend.py 160-180 的 supported_kv_cache_dtypes 检查] ==="
sed -n '160,180p' "$V/v1/attention/backend.py" | cat -n

echo
echo "=== [RocmPlatform 类定义起点与其方法清单] ==="
grep -n "^class RocmPlatform\|^    def \|^    @classmethod" "$P" | sed -n '1,200p' | grep -n "" | tail -60
