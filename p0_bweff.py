# p0_bweff.py -- achievable HBM read bandwidth for gfx906. Nothing else in
# #20 is trustworthy until this number is, so the nominal 1024GB/s is never
# used as the floor. Only 1 fp32 is written per program, so traffic ~= pure read.
import os, statistics, sys
import torch, triton, triton.language as tl


@triton.jit
def read_probe(ptr, out, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # int64: a 4 GiB fp16 buffer is 2**31 elements, which overflows the default
    # int32 offset arithmetic and silently wraps.
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    x = tl.load(ptr + offs, mask=offs < n_elem, other=0.0)
    tl.store(out + pid, tl.sum(x.to(tl.float32)))


def bench(gib: float, iters: int = 20) -> float:
    n = int(gib * 2**30 / 2)                      # fp16 elements
    x = torch.zeros(n, dtype=torch.float16, device="cuda")
    out = torch.zeros(triton.cdiv(n, 8192), dtype=torch.float32, device="cuda")
    grid = (out.shape[0],)
    for _ in range(3):
        read_probe[grid](x, out, n, BLOCK=8192, num_warps=8)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); read_probe[grid](x, out, n, BLOCK=8192, num_warps=8); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    del x, out
    return n * 2 / (statistics.median(ts) / 1e3) / 1e9


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("DEV", "0")))
    print("device:", torch.cuda.get_device_name(int(os.environ.get("DEV", "0"))))
    rows = [(gib, bench(gib)) for gib in (0.25, 1.0, 4.0)]
    for gib, bw in rows:
        print("working_set=%5.2f GiB  read_bw=%7.1f GB/s" % (gib, bw))
    big = rows[-1][1]
    print("BW_eff = %.0f GB/s" % big)
    if big > 950:
        print("SUSPECT: above spec peak -- buffer probably L2-resident", file=sys.stderr)
        sys.exit(2)
    if big < 600:
        print("SUSPECT: far below expectation -- check clocks/thermals", file=sys.stderr)
        sys.exit(3)
