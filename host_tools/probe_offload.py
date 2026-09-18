"""探测 vLLM 0.28.0 的多级 KV offload 在本镜像里是否真的可用。

检查三件事：
  1. GPU<->CPU 搬运用的 C++ 算子 `swap_blocks_batch` 是否注册（ROCm 上 Triton 路径被禁用）
  2. 多级分层相关的类能否导入
  3. CPU 主级所在 /dev/shm 的可用空间
"""

import os
import shutil

import vllm._custom_ops as ops  # noqa: F401  注册 torch.ops._C_cache_ops

res: list[tuple[str, str]] = []

try:
    torch_op = ops.torch.ops._C_cache_ops.swap_blocks_batch
    res.append(("torch.ops._C_cache_ops.swap_blocks_batch", "OK %s" % type(torch_op)))
except Exception as exc:  # noqa: BLE001
    res.append(("torch.ops._C_cache_ops.swap_blocks_batch", "MISSING: %s" % exc))

for name in ("swap_blocks", "swap_blocks_batch"):
    res.append(("vllm._custom_ops.%s" % name, "OK" if hasattr(ops, name) else "MISSING"))

mods = [
    ("vllm.v1.kv_offload.tiering.spec", "TieringOffloadingSpec"),
    ("vllm.v1.kv_offload.tiering.fs.manager", "FileSystemTierManager"),
    ("vllm.v1.kv_offload.cpu.spec", "CPUOffloadingSpec"),
    ("vllm.v1.kv_offload.cpu.shared_offload_region", "SharedOffloadRegion"),
    ("vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
     "OffloadingConnector"),
    ("vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector",
     "SimpleCPUOffloadConnector"),
]
for mod_name, attr in mods:
    try:
        mod = __import__(mod_name, fromlist=[attr])
        res.append(("%s.%s" % (mod_name.rsplit(".", 1)[0], attr),
                    "OK" if hasattr(mod, attr) else "MISSING ATTR"))
    except Exception as exc:  # noqa: BLE001
        res.append(("%s.%s" % (mod_name.rsplit(".", 1)[0], attr),
                    "IMPORT FAIL: %s" % exc))

import importlib.util

_spec = importlib.util.find_spec("vllm.fs_io_C")
res.append(("vllm.fs_io_C (batch_lookup C 扩展)",
            "OK" if _spec is not None else "MISSING"))

try:
    su = shutil.disk_usage("/dev/shm")
    res.append(("/dev/shm", "total %.2f GiB, free %.2f GiB"
                % (su.total / 2**30, su.free / 2**30)))
except Exception as exc:  # noqa: BLE001
    res.append(("/dev/shm", "FAIL: %s" % exc))

try:
    su = shutil.disk_usage("/logs")
    res.append(("/logs", "total %.1f GiB, free %.1f GiB"
                % (su.total / 2**30, su.free / 2**30)))
except Exception as exc:  # noqa: BLE001
    res.append(("/logs", "FAIL: %s" % exc))

res.append(("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "<unset>")))

for key, val in res:
    print("%-52s %s" % (key, val))
