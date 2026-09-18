import json

p = "/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16/config.json"
c = json.load(open(p))

keys = [
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "torch_dtype",
]
for k in keys:
    print("  %-26s %s" % (k, c.get(k)))

q = c.get("quantization_config") or {}
if q:
    print("  %-26s %s" % ("quantization_config",
                          {k: q.get(k) for k in ("quant_method", "bits", "group_size") if k in q}))
    mtc = q.get("modules_to_not_convert")
    if mtc:
        print("  %-26s %d 项" % ("modules_to_not_convert", len(mtc)))

print("  %-26s %s" % ("vision_config", "有（多模态 VLM）" if "vision_config" in c else "无"))

# 注意：qwen3_5 这类复合 config 可能把文本参数放在 text_config 子节点
tc = c.get("text_config") or {}
if tc:
    print("  --- text_config ---")
    for k in ("num_hidden_layers", "hidden_size", "num_attention_heads",
              "num_key_value_heads", "head_dim", "max_position_embeddings"):
        if k in tc:
            print("  %-26s %s" % (k, tc.get(k)))
    lt = tc.get("layer_types")
    if lt:
        from collections import Counter
        print("  %-26s %s" % ("layer_types", dict(Counter(lt))))
