import json, re
from collections import Counter
from safetensors import safe_open
M = "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16/"
idx = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
packed = {k: idx[k] for k in idx if k.endswith("weight_packed")}
print("weight_packed tensors:", len(packed))
c = Counter()
for k in packed:
    tail = k.split("layers.")[-1] if "layers." in k else k
    c[re.sub(r"^\d+\.", "N.", tail[:-len(".weight_packed")])] += 1
for name, n in sorted(c.items()):
    print("  %-46s x%d" % (name, n))
seen, rows = set(), []
for k in packed:
    base = k[:-len(".weight_packed")]
    key = re.sub(r"layers\.\d+\.", "layers.N.", base).split("model.")[-1]
    if key in seen:
        continue
    seen.add(key)
    with safe_open(M + packed[k], framework="pt") as f:
        t = f.get_tensor(k)
        sk = base + ".weight_scale"
        s = f.get_tensor(sk) if sk in idx else None
    rows.append((key, tuple(t.shape), str(t.dtype), tuple(s.shape) if s is not None else None))
print("--- shapes (N,K packed) ---")
for r in rows:
    print("  %-42s %s %s scale=%s" % r)
