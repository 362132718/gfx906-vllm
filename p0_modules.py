# p0_modules.py -- map the two dense fp16 Tensile kernels found in the Task 3
# trace (48 launches/step and 1 launch/step) back to concrete modules.
#
# The trace cannot tell us: inside a FULL_DECODE_ONLY CUDA graph the kernels have
# no per-op CPU parent, and we did not enable record_shapes. The checkpoint can.
# This reads the safetensors shard headers (names + shapes + dtypes, no index
# file needed), separates quantized linear weights (those that have a
# weight_packed/weight_scale sibling) from plain fp16 ones, and prints the (K,N) of
# every fp16 matmul weight together with how many copies exist -- which is the
# launches/step a decode step will make.
#
#   docker exec vllm-gfx906 python3 /work/p0/p0_modules.py [MODEL]
import collections
import json
import os
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
TP = int(os.environ.get("TP", "2"))

cfg = json.load(open(os.path.join(MODEL, "config.json")))
print("config: layers=%s hidden=%s vocab=%s heads=%s kv_heads=%s "
      "ffn=%s num_exp=%s" % (cfg.get("num_hidden_layers"), cfg.get("hidden_size"),
                             cfg.get("vocab_size"), cfg.get("num_attention_heads"),
                             cfg.get("num_key_value_heads"),
                             cfg.get("intermediate_size"), cfg.get("num_experts")))
qcfg = cfg.get("quantization_config", {})
print("quant: algo=%s bits=%s group=%s symmetric=%s dynamic=%s"
      % (qcfg.get("quant_method"), qcfg.get("bits"), qcfg.get("group_size"),
         qcfg.get("symmetric"), qcfg.get("dynamic")))
ignore = qcfg.get("ignore") or qcfg.get("modules_to_not_convert") or []
print("quant ignore list: %d patterns, e.g. %s" % (len(ignore), ignore[:6]))

# shapes are not in the index; vLLM only needs them for the census, so read the
# tensor headers (cheap: 8-byte length + JSON header per shard).
import struct
import glob as _glob
shapes = {}
for shard in sorted(_glob.glob(os.path.join(MODEL, "*.safetensors"))):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k != "__metadata__":
            shapes[k] = (v["shape"], v["dtype"])

QUANT_SIBLING = ("weight_packed", "weight_scale", "qweight", "weight_zeros")
dense = []
for name in shapes:
    base = name.rsplit(".", 1)[-1]
    if base == "weight" and name[: -len("weight")] + "weight_packed" not in shapes \
            and len(shapes[name][0]) == 2:
        dense.append(name)

def ignored(nm):
    return any(nm == p or nm.startswith(p.rstrip(".*")) or p.rstrip(".*") in nm
               for p in ignore)

print("\n== quantized (packed) weight shapes ==")
qc = collections.Counter()
for name in shapes:
    if name.rsplit(".", 1)[-1] in QUANT_SIBLING:
        s = shapes[name]
        qc[(tuple(s[0]), s[1])] += 1
for (shape, dt), n in sorted(qc.items(), key=lambda kv: -kv[1]):
    print("   x%-4d %-24s %s" % (n, shape, dt))

print("\n== plain fp16/fp32 2-D weights that a decode step multiplies ==")
rows = collections.defaultdict(list)
for nm in sorted(set(dense)):
    shape, dt = shapes[nm]
    rows[(tuple(shape), dt)].append(nm)
for (shape, dt), names in sorted(rows.items(), key=lambda kv: -len(kv[1])):
    if shape[1] % TP:
        print("   SKIP  %-18s %s  (vocab/odd dim, %d copies)" % (str(shape), dt, len(names)))
        continue
    per_rank = (shape[0], shape[1] // TP)
    mb = shape[0] * shape[1] * (2 if dt == "F16" else 4) / 1e6
    print("   x%-4d %-18s -> per-rank K=%d N=%d  %s  %.0f MB total  %s"
          % (len(names), str(shape), per_rank[0], per_rank[1], dt, mb,
             "IGNORED(kept fp16)" if any(ignored(n) for n in names) else ""))
    for n in names[:4]:
        print("        %s" % n)
print("\ncompare against the trace: the dense_fp16 bucket had 48 launches/step and"
      " 1 launch/step kernels; a module replicated across every layer of one type"
      " (48 GDN or 64 all) is the 48/step one, a once-per-step module (lm_head)"
      " is the 1/step one.")
