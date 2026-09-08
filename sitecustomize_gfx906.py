# sitecustomize.py (gfx906 vLLM image)
# Applies the gfx906 rocBLAS GEMV workaround (fp32/fp64 matrix*vector
# segfaults on gfx906 with ROCm 7.2.4; redirected to the GEMM path).
# Set VLLM_GFX906_NO_FIX=1 to skip (e.g. for profiling raw interpreter
# startup); the fix itself no-ops on non-gfx906 / non-HIP platforms.
import os

if os.environ.get("VLLM_GFX906_NO_FIX", "0") != "1":
    try:
        import gfx906_gemv_fix  # noqa: F401  (self-applies, guarded)
    except Exception:  # pragma: no cover - never break interpreter startup
        pass
