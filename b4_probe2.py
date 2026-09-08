# b4_probe2.py -- construct CustomAllreduce, verify numerics + graph capture.
# run: torchrun --standalone --nproc_per_node=2 b4_probe2.py
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

gloo_grp = dist.new_group(backend="gloo")
car = CustomAllreduce(group=gloo_grp, device=f"cuda:{rank}")
print(f"[r{rank}] disabled={car.disabled} max_size={getattr(car, 'max_size', None)}",
      flush=True)

if not car.disabled:
    torch.manual_seed(100 + rank)
    for sz in (16, 1024, 512 * 1024):  # elements; 64KB fp
        inp = torch.randn(sz, device=f"cuda:{rank}", dtype=torch.float16)
        ref = inp.clone()
        dist.all_reduce(ref)
        out = car.all_reduce(inp)
        torch.cuda.synchronize()
        d = (out.float() - ref.float()).abs().max().item()
        print(f"[r{rank}] size={sz} maxdiff={d:.3e}", flush=True)
    # graph capture path
    try:
        with torch.cuda.graph(car.graph):
            for _ in range(5):
                x = torch.full((4096,), 0.5, device=f"cuda:{rank}",
                               dtype=torch.float16)
                y = car.all_reduce(x)
        torch.cuda.synchronize()
        val = y.item()
        print(f"[r{rank}] graph-capture OK replay value={val}", flush=True)
    except Exception as e:  # noqa
        print(f"[r{rank}] graph capture FAILED: {type(e).__name__} {e}", flush=True)

dist.barrier()
dist.destroy_process_group()
print(f"[r{rank}] PROBE2_DONE", flush=True)
