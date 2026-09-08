import os
import sys
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
print("rank", rank, "dev", torch.cuda.get_device_name(rank), flush=True)

dist.init_process_group(backend="nccl")
t = torch.ones(4, device=f"cuda:{rank}") * (rank + 1)
dist.all_reduce(t)
torch.cuda.synchronize()
print("rank", rank, "allreduce_result", t.tolist(), flush=True)

dist.barrier()
if rank == 0:
    print("RCCL_TP2_TEST: PASS", flush=True)
dist.destroy_process_group()
