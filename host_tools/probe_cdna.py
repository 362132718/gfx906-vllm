"""只读探针：确认交付镜像里 _ON_CDNA / on_cdna() 的真实状态与全部调用点后果。"""
import inspect
import re

import vllm
import vllm.platforms.rocm as m

print("vllm:", vllm.__version__)
print("rocm.py:", m.__file__)

print("\n--- 标志位 ---")
print("_GCN_ARCH      =", repr(m._GCN_ARCH))
for name in ("_ON_GFX906", "_ON_CDNA", "_ON_GFX9", "_ON_GFX90A", "_ON_GFX942", "_ON_GFX950", "_ON_MI3XX"):
    print(f"{name:14s} =", getattr(m, name, "<absent>"))

print("\n--- 谓词 ---")
for fn in ("on_cdna", "on_gfx906", "on_gfx9", "on_rdna4", "on_mi3xx", "on_gfx90a", "on_gfx942", "on_gfx950"):
    f = getattr(m, fn, None)
    if f is None:
        print(f"{fn:14s} <absent>")
        continue
    try:
        print(f"{fn:14s} =", f())
    except Exception as e:  # noqa: BLE001
        print(f"{fn:14s} raised {type(e).__name__}: {e}")

try:
    print("get_cdna_version() =", m.get_cdna_version())
except Exception as e:  # noqa: BLE001
    print("get_cdna_version() raised", type(e).__name__, e)

print("\n--- RocmPlatform 能力（bug 的下游） ---")
P = m.RocmPlatform
for fn in (
    "supports_fp8",
    "is_fp8_fnuz",
    "fp8_dtype",
    "supports_mx",
    "use_custom_allreduce",
    "opaque_attention_op",
):
    f = getattr(P, fn, None)
    if f is None:
        print(f"{fn:24s} <absent>")
        continue
    try:
        print(f"{fn:24s} =", f())
    except Exception as e:  # noqa: BLE001
        print(f"{fn:24s} raised {type(e).__name__}: {e}")
try:
    print("supported_dtypes         =", P.supported_dtypes)
except Exception as e:  # noqa: BLE001
    print("supported_dtypes raised", type(e).__name__, e)

print("\n--- on_cdna() 在 rocm.py 里的调用点（行号 + 上下文） ---")
src = inspect.getsource(m)
base = src.count("\n") - src.count("\n")  # noop
lines = inspect.getsourcelines(m)[1]
for i, ln in enumerate(src.split("\n"), start=lines):
    if re.search(r"on_cdna\(\)|_ON_CDNA", ln):
        print(f"{i:5d}| {ln.rstrip()[:150]}")
