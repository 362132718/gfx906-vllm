# gfx906_gemv_fix.py -- ROCm 7.2.4 rocBLAS sgemv/dgemv segfaults on gfx906
# for fp32/fp64 matrix*vector (torch.mv, addmv, matmul with 1-D operand).
# Redirect those aten ops to the (working) GEMM path.
#
# Usage: import once at process start (e.g. via sitecustomize in the image).
import torch
import torch.library as TL

APPLIED = False
_LIB = None  # must outlive apply(): GC of a Library revokes its impls


def _mv(mat, vec):
    return (mat @ vec.unsqueeze(1)).squeeze(1)


def _addmv(bias, mat, vec, beta=1, alpha=1):
    r = (mat @ vec.unsqueeze(1)).squeeze(1)
    if r.dtype != bias.dtype:
        r = r.to(bias.dtype)
    return bias * beta + r * alpha


def apply() -> bool:
    global APPLIED
    if APPLIED:
        return True
    if not torch.cuda.is_available() or not torch.version.hip:
        return False
    try:
        arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    except Exception:
        return False
    if "gfx906" not in arch:
        return False
    global _LIB
    _LIB = TL.Library("aten", "IMPL")
    _LIB.impl("mv", _mv, "CUDA")
    _LIB.impl("addmv", _addmv, "CUDA")
    APPLIED = True
    return True


try:
    apply()
except Exception:  # pragma: no cover - never break imports
    pass
