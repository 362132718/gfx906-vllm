# p2_body_ab.py -- localize the production-vs-prototype gap: kernel BODY or BUILD
# PATH? Three arms, one process, one shared weight buffer, alternating rounds:
#   proto_torch : p1_kernels.cu through torch load_inline (Task 0's instrument)
#   prod_so     : p2_kernels_prod.hip through bare `hipcc -O3 -shared -fPIC`
#   body_swap   : PROTOTYPE body + PRODUCTION launcher + PRODUCTION build path
# If body_swap lands on proto_torch, the body is what costs 2.3 us; if it lands on
# prod_so, the compiler flags are. (bit-equality is asserted first -- a "faster"
# body_swap that computes something else would be worthless.)
import ctypes
import sys

sys.path.insert(0, "/work/p1")
sys.path.insert(0, "/work/p2")

import torch

import p1_l1_bench as b

DEV = b.DEV
SK, U = 10, 8
ROUNDS = 5


def load(path, sym):
    lib = ctypes.CDLL(path)
    f = getattr(lib, sym)
    f.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    f.restype = ctypes.c_int
    return f


PROD = load("/work/p2/libp2gemv.so", "p2_gemv")
COPY = load("/work/p2/libp2protocopy.so", "p2_gemv_proto")


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
    out = torch.zeros(N, device=DEV, dtype=torch.float16)
    o2 = torch.zeros(N, device=DEV, dtype=torch.float16)
    scratch = torch.zeros(SK * N, device=DEV, dtype=torch.float32)

    call(PROD, x, qw, sc, zr, part, out, K, N)
    part.zero_()
    call(COPY, x, qw, sc, zr, part, o2, K, N)
    torch.cuda.synchronize()
    same = bool(torch.equal(out, o2))
    print("== K=%d N=%d  body_swap bit-equal to prod_so: %s ==" % (K, N, same),
          flush=True)
    assert same, "the swapped body is not the same function -- A/B is meaningless"

    arms = {
        "proto_torch": capture(
            lambda: b.mod.p1_gemv(x, qw, sc, zr, scratch, 3, SK, U, True)),
        "prod_so": capture(lambda: call(PROD, x, qw, sc, zr, part, out, K, N)),
        "body_swap": capture(lambda: call(COPY, x, qw, sc, zr, part, o2, K, N)),
    }
    acc = {k: [] for k in arms}
    order = ["proto_torch", "body_swap", "prod_so"]
    for r in range(ROUNDS):
        line = "   round %d" % r
        rot = order[r % 3:] + order[:r % 3]     # rotate: monotone drift must not
        for name in rot:                        # bias whichever arm goes first
            t = b.timeit(arms[name].replay, warm=8, reps=32, R=24)
            acc[name].append(t)
            line += "  %s=%.2f" % (name, t)
        print(line + "  temps=%s" % b.temps_c(), flush=True)
    med = {k: sorted(v)[len(v) // 2] for k, v in acc.items()}
    print("   MEDIAN proto=%.2f swap=%.2f prod=%.2f  -> swap-proto=%+.2f "
          "prod-swap=%+.2f us" % (med["proto_torch"], med["body_swap"],
                                  med["prod_so"],
                                  med["body_swap"] - med["proto_torch"],
                                  med["prod_so"] - med["body_swap"]), flush=True)
print("BODY-AB-DONE", flush=True)
