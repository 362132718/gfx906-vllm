import traceback

print("=== 架构标志 ===")
try:
    from vllm.platforms.rocm import _GCN_ARCH
    print("  _GCN_ARCH       =", repr(_GCN_ARCH))
except Exception:
    print("  _GCN_ARCH 读取失败"); traceback.print_exc()

for name in ("on_gfx906", "on_cdna", "on_rdna4", "on_mi3xx", "on_gfx9"):
    try:
        mod = __import__("vllm.platforms.rocm", fromlist=[name])
        fn = getattr(mod, name, None)
        print("  %-14s = %s" % (name, fn() if callable(fn) else fn))
    except Exception as e:
        print("  %-14s 不可用 (%s)" % (name, type(e).__name__))

print()
print("=== 平台能力 ===")
try:
    from vllm.platforms import current_platform as cp
    for m in ("supports_fp8", "is_fp8_fnuz", "fp8_dtype", "get_device_capability",
              "get_device_name", "supports_mx"):
        try:
            v = getattr(cp, m)
            print("  %-22s %s" % (m, v() if callable(v) else v))
        except Exception as e:
            print("  %-22s <err %s: %s>" % (m, type(e).__name__, e))
except Exception:
    traceback.print_exc()

print()
print("=== torch 的 fp8 dtype 是否存在 ===")
try:
    import torch
    print("  torch", torch.__version__)
    for d in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
              "float8_e5m2fnuz", "float8_e8m0fnu"):
        print("  torch.%-18s %s" % (d, hasattr(torch, d)))
except Exception:
    traceback.print_exc()

print()
print("=== 支持的后端 / KV cache dtype 校验 ===")
try:
    from vllm.config.cache import CacheDType  # noqa
    import typing
    print("  CacheDType 取值:", list(typing.get_args(CacheDType)))
except Exception:
    traceback.print_exc()
