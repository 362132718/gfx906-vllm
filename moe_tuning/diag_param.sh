#!/usr/bin/env bash
P=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py
echo "=== parameter.py __torch_function__ 全貌 ==="
grep -n "__torch_function__" -A 40 "$P" | head -60
echo
echo "=== 谁是 BasevLLMParameter 的子类 ==="
grep -n "^class " "$P"
echo
echo "=== shared_expert_gate 的权重要走哪个参数类 ==="
grep -rn "shared_expert_gate\|class ReplicatedLinear" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py | head -5
