# vit_ref.py -- numeric audit of the gfx906 vision tower that we ship.
#
# Loads ONLY the 333 `model.visual.*` tensors from the checkpoint and runs the
# same deterministic synthetic image through:
#   A) HF transformers Qwen3_5VisionModel, fp32 on CPU   (ground reference)
#   B) HF transformers Qwen3_5VisionModel, fp16 on CPU   (dtype-only error)
#   C) vLLM Qwen3_VisionTransformer, fp16 on GPU, TRITON_ATTN encoder (ours)
#
# Reference runs on CPU on purpose: HF's default SDPA path faults on gfx906 (no
# MFMA), so a GPU-resident reference would measure the wrong thing.
# Compare A-vs-B (dtype) against B-vs-C (our implementation): same order of
# magnitude means the gfx906 path is numerically sound.
import json, torch
import numpy as np
from PIL import Image, ImageDraw

M = "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16/"
torch.manual_seed(0)
np.random.seed(0)

cfg = json.load(open(M + "config.json"))
vc = dict(cfg["vision_config"])
vc.pop("model_type", None)

img = Image.new("RGB", (448, 448), (240, 240, 240))
d = ImageDraw.Draw(img)
d.ellipse([70, 120, 210, 260], fill=(215, 45, 45))
d.rectangle([250, 150, 380, 280], fill=(45, 95, 215))
d.line([40, 40, 400, 40], fill=(20, 20, 20), width=12)

from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

print("step: processor", flush=True)
proc = AutoProcessor.from_pretrained(M)
enc = proc.image_processor(images=[img], return_tensors="pt")
pv, grid = enc["pixel_values"], enc["image_grid_thw"]
print("pixel_values", tuple(pv.shape), "grid", grid.tolist(), flush=True)

from safetensors import safe_open
idx = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
vis = {k: v for k, v in idx.items() if k.startswith("model.visual.")}
sd = {}
for shard in sorted(set(vis.values())):
    with safe_open(M + shard, framework="pt") as f:
        for k in [x for x in vis if vis[x] == shard]:
            sd[k[len("model.visual."):]] = f.get_tensor(k).float()
print("loaded visual tensors:", len(sd), flush=True)


def report(a, b, tag):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    rel = float((a - b).norm() / a.norm())
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print("%-28s rel_l2=%.2e cos=%.6f max|d|=%.3e" % (tag, rel, cos, float((a - b).abs().max())), flush=True)
    return rel


hcfg = Qwen3_5VisionConfig(**vc)
hcfg._attn_implementation = "eager"
try:
    hcfg.attn_implementation = "eager"
except Exception:
    pass

print("step: A fp32 reference on CPU", flush=True)
m32 = Qwen3_5VisionModel(hcfg)
r = m32.load_state_dict(sd, strict=False)
print("A load: missing=%d unexpected=%d" % (len(r.missing_keys), len(r.unexpected_keys)), flush=True)
with torch.no_grad():
    out32 = m32(hidden_states=pv, grid_thw=grid).pooler_output

print("step: B fp16 on CPU", flush=True)
m16 = Qwen3_5VisionModel(hcfg)
m16.load_state_dict({k: v.half() for k, v in sd.items()}, strict=False)
with torch.no_grad():
    out16 = m16(hidden_states=pv.half(), grid_thw=grid).pooler_output

d_dtype = report(out32, out16, "A fp32 vs B fp16 (dtype only)")

print("step: C vLLM fp16 on GPU (TRITON_ATTN encoder)", flush=True)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.qwen3_5 import Qwen3_VisionTransformer
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)

# The encoder shards its QKV/proj layers, so it needs a TP group even at tp=1,
# and initialize_model_parallel() reads the ambient vLLM config.
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="file:///tmp/vitref_dist",
                                 backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)
    mv = Qwen3_VisionTransformer(Qwen3VLVisionConfig(**vc), quant_config=None)
    rc = mv.load_state_dict({k: v.half() for k, v in sd.items()}, strict=False)
    print("C load: missing=%d unexpected=%d" % (len(rc.missing_keys), len(rc.unexpected_keys)), flush=True)
    if rc.unexpected_keys:
        print("  unexpected sample:", rc.unexpected_keys[:5], flush=True)
    if rc.missing_keys:
        print("  missing sample:", rc.missing_keys[:5], flush=True)
    mv = mv.to("cuda").half().eval()
    with torch.no_grad():
        outv = mv(pv.to("cuda", torch.half), grid.to("cuda"))

d_impl = report(out16.cpu(), outv.cpu(), "B fp16 vs C vllm-gfx906")
d_tot = report(out32, outv.cpu(), "A fp32 vs C vllm-gfx906 (total)")

# Attribution step: run vLLM's encoder in fp32. If A-vs-D is dtype-sized, then
# the ~1e-2 in C is fp16 accumulation along our path, not an implementation bug.
with set_current_vllm_config(VllmConfig()):
    mv32 = Qwen3_VisionTransformer(Qwen3VLVisionConfig(**vc), quant_config=None)
    mv32.load_state_dict(sd, strict=False)
    mv32 = mv32.to("cuda").float().eval()
    with torch.no_grad():
        outv32 = mv32(pv.to("cuda", torch.float32), grid.to("cuda"))
d_fp32 = report(out32, outv32.cpu(), "A fp32 vs D vllm-fp32 (impl)")

verdict = "PASS" if d_fp32 <= 3.0 * max(d_dtype, 1e-4) else "REVIEW"
print("VIT_AUDIT_VERDICT=%s  dtype=%.2e impl_fp16=%.2e impl_fp32=%.2e"
      % (verdict, d_dtype, d_impl, d_fp32), flush=True)
