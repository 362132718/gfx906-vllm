# p2_unroll_ab.py -- verify the two-level-unroll fix. Three arms, one process, one
# shared weight buffer, order rotated per round:
#   proto_torch : p1_kernels.cu via torch load_inline  (the 60.4 us reference)
#   prod_new    : p2_kernels_prod.hip, two-level loop   (must land on proto)
#   prod_old    : same file with `#pragma unroll 8` on the single 16-trip loop
# Gates: prod_new bit-equal to prod_old AND to proto_torch (same function, less
# time), and |prod_new - proto_torch| within 1% -- the whole point is to recover
# the 2.2-2.4 us that G-P0's approved 99% aggregate budget cannot spare.
import ctypes
import sys

sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l1_bench as b

DEV = b.DEV
SK, U = 10, 8
ROUNDS = 5


def load(path, sym="p2_gemv"):
    lib = ctypes.CDLL(path)
    f = getattr(lib, sym)
    f.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    f.restype = ctypes.c_int
    return f


NEW = load("/work/p2/libp2gemv.so")
OLD = load("/work/p2/libp2gemv_old.so")


def call(fn, x, qw, sc, zr, part, out, K, N):
    rc = fn(x.data_ptr(), qw.data_ptr(), sc.data_ptr(), zr.data_ptr(),
            part.data_ptr(), out.data_ptr(), K, N, SK,
            torch.cuda.current_stream().cuda_stream)
    assert rc == 0, "rc=%d" % rc


def capture(fn):
    fn()
    fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    torch.cuda.synchronize()
    return gr


for K, N in ((5120, 8192), (5120, 7168)):
    x, qw, sc, zr = b.tensors(K, N, seed=1)
    part = torch.zeros((SK, N), device=DEV, dtype=torch.float32)
    o_new = torch.zeros(N, device=DEV, dtype=torch.float16)
    o_old = torch.zeros(N, device=DEV, dtype=torch.float16)
    scratch = torch.zeros(SK * N, device=DEV, dtype=torch.float32)

    call(NEW, x, qw, sc, zr, part, o_new, K, N)
    call(OLD, x, qw, sc, zr, part, o_old, K, N)
    proto = b.mod.p1_gemv(x, qw, sc, zr, scratch, 3, SK, U, True)
    torch.cuda.synchronize()
    print("== K=%d N=%d  new==old: %s   new==prototype: %s ==" % (
        K, N, bool(torch.equal(o_new, o_old)), bool(torch.equal(o_new, proto))),
        flush=True)

    arms = {
        "proto_torch": capture(
            lambda: b.mod.p1_gemv(x, qw, sc, zr, scratch, 3, SK, U, True)),
        "prod_new": capture(lambda: call(NEW, x, qw, sc, zr, part, o_new, K, N)),
        "prod_old": capture(lambda: call(OLD, x, qw, sc, zr, part, o_old, K, N)),
    }
    acc = {k: [] for k in arms}
    order = ["proto_torch", "prod_new", "prod_old"]
    for r in range(ROUNDS):
        line = "   round %d" % r
        for name in order[r % 3:] + order[:r % 3]:
            t = b.timeit(arms[name].replay, warm=8, reps=32, R=24)
            acc[name].append(t)
            line += "  %s=%.2f" % (name, t)
        print(line + "  temps=%s" % b.temps_c(), flush=True)
    med = {k: sorted(v)[len(v) // 2] for k, v in acc.items()}
    print("   MEDIAN proto=%.2f new=%.2f old=%.2f -> new-proto=%+.2f us (%+.2f%%), "
          "old-new=%+.2f us" % (med["proto_torch"], med["prod_new"], med["prod_old"],
                                med["prod_new"] - med["proto_torch"],
                                100 * (med["prod_new"] / med["proto_torch"] - 1),
                                med["prod_old"] - med["prod_new"]), flush=True)
print("UNROLL-AB-DONE", flush=True)
