# b4_stale.py -- author-style stress: continuously varying message sizes and
# contents; custom-AR result must match the fp32 a+b reference each round.
# Stale-data races (peer's previous-round buffer) show up as large diffs.
# run: torchrun --standalone --nproc_per_node=2 b4_stale.py
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

gloo = dist.new_group(backend="gloo")
car = CustomAllreduce(group=gloo, device=f"cuda:{rank}")
assert not car.disabled, "custom AR disabled!"

sizes = [4096, 5120, 6144, 8192, 12288, 16384, 20480, 32768, 65536,
         131072, 262144, 524288, 1048576]
worst = 0.0
for it in range(4):
    for n in sizes:
        g = torch.Generator(device="cpu").manual_seed(1000 * it + n)
        a = torch.randn(n, generator=g).to(f"cuda:{rank}", torch.float16)
        g2 = torch.Generator(device="cpu").manual_seed(7777 - 1000 * it - n)
        b = torch.randn(n, generator=g2).to(f"cuda:{rank}", torch.float16)
        # rank0 contributes a, rank1 contributes b; expected = a+b (fp32)
        mine = a if rank == 0 else b
        expected_a = (a.float() + b.float())
        out = car.all_reduce(mine.clone())
        torch.cuda.synchronize()
        d = (out.float() - expected_a).abs()
        rel = (d / expected_a.abs().clamp_min(1.0)).max().item()
        worst = max(worst, rel)
        if rel > 1e-2:
            print(f"[r{rank}] it={it} n={n} REL={rel:.3e} STALE_SUSPECT",
                  flush=True)
print(f"[r{rank}] worst_rel={worst:.3e} {'CLEAN' if worst < 1e-2 else 'DIRTY'}",
      flush=True)
dist.barrier()
dist.destroy_process_group()
