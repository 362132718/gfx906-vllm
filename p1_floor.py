# p1_floor.py -- the denominator for every L1 percentage. Bytes the kernel
# actually pulls from HBM (qweight + scales + zeros), plus BW_eff measured twice
# under the clocks L1 will use: a pure-read figure and a read+write figure.
#
# The first draft of this file timed `torch.add(x[:64], x[-64:], out=y)` with a
# one-element y -- it raised on shape, and even had it run it would have charged
# 258 bytes for a 512 MiB buffer. A floor that is wrong invalidates every
# percentage downstream, so both measurements here move whole buffers.
import json
import torch

DEV = "cuda"
BW_FLOOR_REF = 821.0       # GB/s, design's conservative floor
BW_READ_REF = 928.0        # GB/s, design's pure-read ceiling
G = 128
SHAPES = [(5120, 7168, 16), (3072, 5120, 64), (5120, 17408, 64),
          (8704, 5120, 64), (5120, 8192, 48)]   # (K, N, calls per token)
SPLITK = {(5120, 7168): 2, (3072, 5120): 4, (5120, 17408): 1,
          (8704, 5120): 4, (5120, 8192): 2}      # design :102-108


def _median_us(fn, reps=15):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def bw_pure_read(mib=1024):
    """Stream a whole buffer and reduce it: bytes = numel*2, one read pass."""
    x = torch.randn(mib * 1024 * 1024 // 2, device=DEV, dtype=torch.float16)
    us = _median_us(lambda: torch.sum(x))
    return x.numel() * 2 / 1e9 / (us / 1e6)


def bw_read_write(mib=1024):
    """add(x, x, out=o) really reads 2 buffers and writes 1 -- closer to what a
    GEMV's neighbourhood does, and a stricter bound on sustained bandwidth."""
    x = torch.randn(mib * 1024 * 1024 // 2, device=DEV, dtype=torch.float16)
    o = torch.empty_like(x)
    us = _median_us(lambda: torch.add(x, x, out=o))
    return 3 * x.numel() * 2 / 1e9 / (us / 1e6)


def bytes_pulled(K, N):
    """One int32 word carries 8 codes for one output, so qweight is K*N/2 bytes.
    Splits tile K disjointly, each reading only its own group slice of scales and
    zeros => the grid reads metadata exactly ONCE, not splitk times."""
    qw = (K // 8) * N * 4
    sc = (K // G) * N * 2
    zp = (K // G) * (N // 8) * 4
    return qw + sc + zp, qw, sc + zp


def main():
    br, b_rw = bw_pure_read(), bw_read_write()
    print("BW pure-read=%.1f GB/s (ref %.0f) | read+write=%.1f GB/s" % (br, BW_READ_REF, b_rw))
    assert br > BW_FLOOR_REF * 0.9, (
        "measured read bandwidth %.1f GB/s is below 90%% of the %.0f floor -- "
        "check `p0_clock.sh status`: mclk must be pinned level 2 = 1000MHz" % (br, BW_FLOOR_REF))
    bw = br                          # a GEMV is read-bound; b_rw stays as context
    total_read_only = total_full = 0.0
    shapes = {}
    for K, N, calls in SHAPES:
        pulled, qw, meta = bytes_pulled(K, N)
        floor_us = pulled / (bw * 1e9) * 1e6
        floor_qw_only = qw / (bw * 1e9) * 1e6
        per_tok = calls * floor_us                             # us/token
        total_full += per_tok
        total_read_only += calls * floor_qw_only
        shapes["%d,%d" % (K, N)] = dict(pulled_bytes=pulled, qw_bytes=qw,
                                        meta_bytes=meta, splitk=SPLITK[(K, N)],
                                        floor_us=floor_us, calls_per_token=calls)
        print("K=%-6d N=%-6d splitK=%-2d pulled=%.2fMB (meta %.1f%%) floor=%.2f us "
              "x%d/token=%.1f us" % (K, N, SPLITK[(K, N)], pulled / 1e6,
                                     100 * meta / qw, floor_us, calls, per_tok))
    print("SUM floors = %.1f us/token (qweight-only would be %.1f) | today's "
          "w4a16_gemv bucket = 19867 us/step" % (total_full, total_read_only))
    print("headroom vs today's bucket = %.1f ms/step" % ((19867 - total_full) / 1000))
    json.dump({"bw_pure_read_gb_s": br, "bw_read_write_gb_s": b_rw,
               "bw_floor_ref": BW_FLOOR_REF, "bw_read_ref": BW_READ_REF,
               "sum_floor_us_per_token": total_full, "shapes": shapes},
              open("/work/p1/p1_floors.json", "w"), indent=1)
    print("wrote /work/p1/p1_floors.json")
    print("FLOOR-OK")


if __name__ == "__main__":
    main()
