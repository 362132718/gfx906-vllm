"""p7_peer_bw.py -- measure the real GPU0<->GPU1 collective latency/bandwidth.

Why (2026-09-11): the MoE decode profile attributes 13.07 ms/step to
`ncclDevKernel_Generic_4` at 82 calls/step = 163 us per all-reduce for a 4 KB
payload. Two very different explanations fit that number:

  (a) the PCIe peer path is genuinely slow/latency-bound, or
  (b) the kernel is mostly *spin-waiting for the peer rank* because the two
      GPUs are imbalanced -- i.e. the cost is load imbalance, not transport.

The distinction decides what to fix: (a) => shrink/replace collectives,
(b) => rebalance the work. A microbenchmark of RCCL alone (no model, no
imbalance) separates them: if a bare 4 KB all_reduce is ~30 us, then ~130 us of
the in-engine 163 us is waiting.

Also measures point-to-point send/recv (the path PP would use) and the
host<->device rate, so the "PCIe 3.0 x8" claim can be checked directly.

Run (2 processes, one per GPU):
    python3 p7_peer_bw.py
Env: OUT (json path), ITERS, WARMUP
"""
import json
import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.0.6")

OUT = os.environ.get("OUT", "/work/p7/peer_bw.json")
WORLD = 2

# bytes; fp16 elements = size/2
SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20,
         4 << 20, 16 << 20, 64 << 20, 256 << 20]


def _bench_allreduce(rank, dist, torch):
    import statistics
    rows = []
    for nbytes in SIZES:
        nelem = max(nbytes // 2, 1)
        t = torch.ones(nelem, dtype=torch.float16, device=f"cuda:{rank}")
        # warmup
        for _ in range(10):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        iters = 200 if nbytes <= (1 << 20) else 30
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        # per-iteration samples for the small sizes (jitter matters)
        lat = []
        if nbytes <= (64 << 10):
            for _ in range(50):
                torch.cuda.synchronize()
                s = time.perf_counter()
                dist.all_reduce(t)
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - s) * 1e6)
        rows.append({
            "bytes": nbytes,
            "us": dt * 1e6,
            "alg_GBps": 2.0 * nbytes / dt / 1e9 if dt > 0 else 0.0,
            "min_us": min(lat) if lat else None,
            "p50_us": statistics.median(lat) if lat else None,
        })
    return rows


def _bench_p2p(rank, dist, torch):
    """send/recv both ways: this is the PP stage-boundary path."""
    rows = []
    peer = 1 - rank
    for nbytes in [4 << 10, 256 << 10, 4 << 20, 64 << 20]:
        nelem = max(nbytes // 2, 1)
        send = torch.ones(nelem, dtype=torch.float16, device=f"cuda:{rank}")
        recv = torch.zeros(nelem, dtype=torch.float16, device=f"cuda:{rank}")
        for _ in range(5):
            if rank == 0:
                dist.send(send, peer)
                dist.recv(recv, peer)
            else:
                dist.recv(recv, peer)
                dist.send(send, peer)
        torch.cuda.synchronize()
        iters = 100 if nbytes <= (4 << 20) else 20
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            if rank == 0:
                dist.send(send, peer)
                dist.recv(recv, peer)
            else:
                dist.recv(recv, peer)
                dist.send(send, peer)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        rows.append({"bytes": nbytes, "us": dt * 1e6,
                     "GBps": nbytes / dt / 1e9 if dt > 0 else 0.0})
    return rows


def worker(rank, ret):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    print(f"[rank{rank}] device={torch.cuda.get_device_name(rank)}", flush=True)
    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD)
    out = {"rank": rank, "device": torch.cuda.get_device_name(rank)}
    try:
        out["allreduce"] = _bench_allreduce(rank, dist, torch)
        out["p2p"] = _bench_p2p(rank, dist, torch)
    finally:
        dist.destroy_process_group()
    ret[rank] = out


def main():
    import torch.multiprocessing as mp
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(worker, args=(ret,), nprocs=WORLD, join=True)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    res = {str(k): dict(v) for k, v in ret.items()}
    json.dump(res, open(OUT, "w"), indent=1)

    r0 = res.get("0", {})
    print("\n=== all_reduce (4 KB payload = one decode-step collective) ===")
    print("%12s %12s %10s %10s %10s" % ("bytes", "mean_us", "min_us", "p50_us", "alg_GB/s"))
    for row in r0.get("allreduce", []):
        print("%12d %12.1f %10s %10s %10.3f" % (
            row["bytes"], row["us"],
            "-" if row["min_us"] is None else "%.1f" % row["min_us"],
            "-" if row["p50_us"] is None else "%.1f" % row["p50_us"],
            row["alg_GBps"]))
    print("\n=== point-to-point send/recv (the PP stage-boundary path) ===")
    print("%12s %12s %10s" % ("bytes", "us", "GB/s"))
    for row in r0.get("p2p", []):
        print("%12d %12.1f %10.3f" % (row["bytes"], row["us"], row["GBps"]))
    print("\nP7_PEER_DONE_OK -> %s" % OUT)


if __name__ == "__main__":
    main()
