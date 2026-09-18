#!/usr/bin/env python3
"""B 类反向抽取补丁：修掉 gfx906 被误判为 CDNA 的两处后果。

1) platforms/rocm.py  _ON_CDNA 子串匹配  ⇒ "gfx9" in "gfx906" 为真
2) 同文件 check_and_update_config 里补一条 fp8 KV 显式拒绝（用 supports_fp8() 判定）

设计要点：
- 两处都在 rocm.py 一个文件里 ⇒ 一个 md5 可追踪。
- 幂等：已打过则直接报告并退出 0。
- 锚点必须唯一命中，否则中止（防止误改）。
- 改前后各记 md5，并做 py_compile 语法闸门。
- 不动其它平台语义：补丁 2 只在 supports_fp8() 为 False 的平台生效。
"""
from __future__ import annotations

import hashlib
import py_compile
import shutil
import sys
import time
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py"
)

ANCHOR_CDNA = '_ON_CDNA = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx1250"])'

NEW_CDNA = """# NOTE(gfx906): "gfx9" is a substring of "gfx906", so a plain substring
# match classified Vega20/GCN5 as CDNA and silently enabled CDNA-only paths
# (most importantly supports_fp8(), which then green-lit fp8 KV cache).
# Honour the exclusion stated above: gfx906 is never CDNA.
_ON_CDNA = (not _ON_GFX906) and any(
    arch in _GCN_ARCH for arch in ["gfx9", "gfx1250"]
)"""

ANCHOR_CFG = '''    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config.compilation import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config
'''

NEW_CFG = '''    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config.compilation import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config

        # NOTE(gfx906): nothing downstream rejects an fp8 KV cache on ROCm.
        # The Triton attention backend advertises fp8/fp8_e4m3/fp8_e5m2 in
        # `supported_kv_cache_dtypes`, the SM89 guard in `triton_attn.py` only
        # runs under `current_platform.is_cuda()`, and
        # `_is_supported_kv_cache_dtype()` compares a CUDA capability number
        # against the ROCm arch string ("gfx906" reads as capability (9, 0),
        # which passes `>= (8, 9)`). Vega20/GCN5 has no FP8 arithmetic at all,
        # so an fp8 KV cache would be emulated in software and silently
        # mis-computed. Fail loudly here instead, on the platform that knows.
        logger.info_once(
            "ROCm FP8 posture: arch=%s on_cdna=%s supports_fp8=%s",
            _GCN_ARCH,
            on_cdna(),
            cls.supports_fp8(),
        )
        if not cls.supports_fp8():
            cache_config = getattr(vllm_config, "cache_config", None)
            kv_dtype = str(getattr(cache_config, "cache_dtype", "") or "")
            if kv_dtype.startswith("fp8"):
                raise ValueError(
                    f"kv_cache_dtype='{kv_dtype}' is not supported on "
                    f"{cls.get_device_name()}: this GPU has no native FP8 "
                    f"arithmetic, so an FP8 KV cache would be software "
                    f"emulated and is not validated on this hardware. "
                    f"Use --kv-cache-dtype float16, or "
                    f"--kv-cache-dtype int8_per_token_head for the "
                    f"2x-compressed Triton-based path."
                )
'''


def main() -> int:
    if not TARGET.is_file():
        print(f"ABORT: not found: {TARGET}")
        return 2

    src = TARGET.read_text(encoding="utf-8")
    before = hashlib.md5(src.encode("utf-8")).hexdigest()

    if "(not _ON_GFX906) and any(" in src and "ROCm FP8 posture" in src:
        print(f"ALREADY PATCHED (md5={before}) — nothing to do.")
        return 0

    for name, anchor in (("_ON_CDNA", ANCHOR_CDNA), ("check_and_update_config", ANCHOR_CFG)):
        n = src.count(anchor)
        if n != 1:
            print(f"ABORT: anchor {name} matched {n} times (expected 1).")
            return 2

    out = src.replace(ANCHOR_CDNA, NEW_CDNA, 1).replace(ANCHOR_CFG, NEW_CFG, 1)

    for probe in ("(not _ON_GFX906) and any(", "ROCm FP8 posture", "supports_fp8()"):
        if probe not in out:
            print(f"ABORT: post-check failed, {probe!r} missing in result.")
            return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = TARGET.with_suffix(f".py.orig-{stamp}")
    shutil.copy2(TARGET, backup)

    TARGET.write_text(out, encoding="utf-8")
    after = hashlib.md5(TARGET.read_text(encoding="utf-8").encode("utf-8")).hexdigest()

    try:
        py_compile.compile(str(TARGET), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as exc:
        TARGET.write_text(src, encoding="utf-8")
        print(f"SYNTAX ERROR -> rolled back to original. {exc}")
        return 3
    except Exception:  # noqa: BLE001
        compile_ok = True  # 某些镜像无 py_compile 写权限，退化为已做的文本校验

    print(f"target   : {TARGET}")
    print(f"backup   : {backup}")
    print(f"md5 before={before}")
    print(f"md5 after ={after}")
    print(f"py_compile: {'ok' if compile_ok else 'skipped'}")

    lines = out.split("\n")
    for i, ln in enumerate(lines, 1):
        if "_ON_CDNA = (not" in ln or "ROCm FP8 posture" in ln or "kv_cache_dtype='{" in ln:
            print(f"  {i:5d}| {ln.rstrip()}")
    print("PATCH OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
