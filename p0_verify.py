"""P0 verification: prove the is_gfx906 landmine is gone and the patched
triton_decode_attention path actually executes on gfx906.

Run inside the delivery container:
    docker exec <c> python3 /tmp/p0_verify.py
"""
import math
import torch

print("== 1. the old call site still raises (control) ==")
from vllm.platforms import current_platform
try:
    current_platform.is_gfx906()
    print("   [!!] current_platform.is_gfx906() did NOT raise -- unexpected")
except TypeError as e:
    print("   [ok] old form raises TypeError: %s" % e)

print("== 2. the new predicate resolves ==")
from vllm.platforms.rocm import on_gfx906
print("   on_gfx906() =", on_gfx906())

print("== 3. import the patched module ==")
from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd,
    _decode_grouped_att_m_fwd,
)
import inspect
src = inspect.getsource(_decode_grouped_att_m_fwd)
print("   module imported; 'is_gfx906' in source:", "is_gfx906" in src)
print("   'on_gfx906' in source:", "on_gfx906" in src)
if "is_gfx906" in src:
    raise SystemExit("[!!] dangling is_gfx906 still present")

print("== 4. execute decode_attention_fwd (GQA => grouped path) ==")
dev = "cuda"
B, H, KVH, D, SEQ, SPLITS, PAGE = 1, 4, 1, 128, 8, 2, 1
NBLK = 16

q = torch.randn(B, H, D, dtype=torch.float16, device=dev)
k_buffer = torch.randn(NBLK, PAGE, KVH, D, dtype=torch.float16, device=dev)
v_buffer = torch.randn(NBLK, PAGE, KVH, D, dtype=torch.float16, device=dev)
o = torch.empty(B, H, D, dtype=torch.float16, device=dev)
lse = torch.empty(B, H, dtype=torch.float32, device=dev)
req_to_token = torch.arange(NBLK, dtype=torch.int32, device=dev).unsqueeze(0)
b_seq_len = torch.tensor([SEQ], dtype=torch.int32, device=dev)
attn_logits = torch.empty(B, H, SPLITS, D, dtype=torch.float32, device=dev)

decode_attention_fwd(
    q, k_buffer, v_buffer, o, lse, req_to_token, b_seq_len,
    attn_logits, SPLITS, 1.0 / math.sqrt(D), PAGE,
)
torch.cuda.synchronize()
print("   [ok] executed without TypeError")
print("   o finite:", bool(torch.isfinite(o).all()), "| o.norm =", float(o.norm()))
print("   lse finite:", bool(torch.isfinite(lse).all()))
print("P0_VERIFY_PASSED")
