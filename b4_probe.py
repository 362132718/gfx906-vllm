# b4_probe.py -- why is custom all-reduce unavailable on this pair?
# run: torchrun --standalone --nproc_per_node=2 b4_probe.py
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")

if rank == 0:
    print("peer_access(0,1):", torch.cuda.can_device_access_peer(0, 1))
    p0 = torch.cuda.get_device_properties(0)
    print("props:", p0.gcnArchName, "SMs:", p0.multi_processor_count,
          "P2P support flag:", getattr(p0, "crossLinkVersion", "n/a"),
          {k: v for k, v in vars(p0).items() if "link" in k.lower() or "p2p" in k.lower()})

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

try:
    car = CustomAllreduce(group=dist.group.WORLD, device=f"cuda:{rank}")
    print(f"[rank{rank}] CustomAllreduce disabled attr:", car.disabled,
          flush=True)
except Exception as e:  # noqa
    print(f"[rank{rank}] ctor raised:", type(e).__name__, str(e)[:500], flush=True)

# try enable_custom explicitly
try:
    import vllm.envs as envs
    print(f"[rank{rank}] VLLM envs custom flags:",
          {k: v for k, v in vars(envs).items() if "ALL_REDUCE" in k.upper() or "CUSTOM" in k.upper()},
          flush=True)
except Exception as e:  # noqa
    print("envs err", e)

dist.destroy_process_group()
print(f"[rank{rank}] done", flush=True)
