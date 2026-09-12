import torch
print("dev=", torch.cuda.get_device_name(0))
try:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    topk_ids = torch.tensor([[1, 2], [0, 3], [3, 3]], dtype=torch.int32, device="cuda")
    out = moe_align_block_size(topk_ids, 16, 4)
    print("MOE_ALIGN OK", type(out).__name__, len(out) if isinstance(out, tuple) else "single")
except Exception as e:
    print("MOE_ALIGN FAIL", type(e).__name__, e)
try:
    from vllm.model_executor.layers.fused_moe.activation import moe_sum
    x = torch.randn(6, 2, 8, dtype=torch.float16, device="cuda")
    o = torch.empty(2, 8, dtype=torch.float16, device="cuda")
    moe_sum(x, o)
    print("MOE_SUM OK", tuple(o.shape))
except Exception as e:
    print("MOE_SUM FAIL", type(e).__name__, e)
try:
    import vllm._custom_ops as ops
    print("awq_gemm attr:", hasattr(ops, "awq_gemm"), "awq_dequantize attr:", hasattr(ops, "awq_dequantize"))
except Exception as e:
    print("OPS FAIL", type(e).__name__, e)
