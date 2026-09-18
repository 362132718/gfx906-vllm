#!/bin/bash
# 在容器内运行：核实三件事
SP=/usr/local/lib/python3.12/dist-packages/vllm

echo "=== [1] per-token-head 量化 + offload 的内存布局门禁是否存在（社区那个 PR 的修复）==="
grep -rn --include=*.py -iE "per.token.head.*(offload|connector)|(offload|connector).*per.token.head" "$SP" | head -20
echo "--- 若上面为空，换个角度找 scale 缓冲区的分配闸门 ---"
grep -rn --include=*.py "kv_cache_config" "$SP/v1/worker/gpu_model_runner.py" | head -10
grep -rn --include=*.py -iE "k_scale_cache|v_scale_cache" "$SP/v1/worker/gpu_model_runner.py" | head -20

echo
echo "=== [2] prefix_match_unit 是否已被本版本支持（hybrid 前缀缓存 0% 命中的缓解开关）==="
grep -rn --include=*.py "prefix_match_unit" "$SP" | head -20

echo
echo "=== [3] 版本号 ==="
python3 -c "import vllm; print(vllm.__version__)"
