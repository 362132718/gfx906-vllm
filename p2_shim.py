# p2_shim.py -- ctypes loader for the production M=1 W4A16 GEMV.
# No torch C++ ABI: raw pointers + the CURRENT torch stream (capture-safe).
import ctypes

import torch

_LIB = None


def p2_lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(
            "/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so")
        lib.p2_gemv.argtypes = ([ctypes.c_void_p] * 6
                                + [ctypes.c_int] * 3 + [ctypes.c_void_p])
        lib.p2_gemv.restype = ctypes.c_int   # 0 ok / -1 rejected: MUST be checked
        _LIB = lib
    return _LIB


def p2_gemv(x, qw, sc, zr, part, out, K, N, splitk=10):
    lib = p2_lib()
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.p2_gemv(x.data_ptr(), qw.data_ptr(), sc.data_ptr(), zr.data_ptr(),
                     part.data_ptr(), out.data_ptr(), K, N, splitk, stream)
    if rc != 0:
        raise RuntimeError(
            f"p2_gemv rejected/failed K={K} N={N} splitk={splitk} rc={rc}")
