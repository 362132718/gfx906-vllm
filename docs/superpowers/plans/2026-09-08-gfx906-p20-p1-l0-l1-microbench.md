# #20-P1「L0 位映射钉死 + L1 单形状微基准」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用微基准证明或否掉"手写 wave64 W4A16 GEMV 能把 op/weight 压到 3.85 以下、在 graph 内比 ExLlama 现路径快 >1.1×"，据此对 P2（真写内核）下 **go/kill 判定**；不产出任何进交付镜像的代码。

**Architecture:** 五个独立产物，全部在开发容器 `/work/p1/` 里跑，**不碰 site-packages**（与 P0.5 的关键区别：P1 是纯测量）。L0 用 `x = e_k`（one-hot）把"内核解出来的值"变成对 `(k,n)` 的**逐位置直读**，因此位映射正确性检查零容差；L1 用一个 JIT 模块一次性编出全部候选变体（`variant` 作运行期参数，避免每变体重编），eager 与 CUDA graph 内各测一遍，选型**只看 graph 内数字**；op/weight 不靠计时反推，直接从编译产物的 ISA 里数。

**Tech Stack:** vLLM 0.28.0（容器内 site-packages）、ROCm 7.2.4 / `torch.version.hip`、`torch.utils.cpp_extension.load_inline`（gfx906）、`llvm-objdump`（数 ISA）、既有 `p0_clock.sh`（离散 DPM 钉频，sclk+mclk 一起钉）、`VLLM_WNA16_EXL_SHAPEDBG` 调试通道（`wna16_exllama.py:53`）。

---

## 前置事实（执行前必读，均来自本仓库/容器实测，不是推测）

| 事实 | 出处 |
|---|---|
| 桥接后的权重布局：`weight_packed [N,K/8] int32 (LSB-first, 沿 K)` → `qweight [K/8,N] int32`，**ExLlama 读法 = 第 i 个 k 在位 `(i//2)*4 + (i%2)*16`** | `container_src/wna16_exllama.py:14-26`、`transform_w_q` 的 `dst_bits` 同文件 `:179` |
| `scales` → `[K/G, N]` fp16；`qzeros` → `[K/G, N//8]` int32（GPTQv2：存储 nibble == 真 zp） | 同文件 `:17-21`、`:189-220` |
| `g_idx` 是 act-order **置换表**，无 act-order 时必须是空张量 | 同文件 `:32-34`、`:155-165` |
| 内核构造签名 `MPLinearKernel.__init__(c, w_q_param_name, w_s_param_name, w_zp_param_name=None, w_gidx_param_name=None)`，内部 `assert self.can_implement(c)` | 容器 `.../mixed_precision/MPLinearKernel.py:37-54` |
| `MPLinearLayerConfig` 字段：`full_weight_shape, partition_weight_shape, weight_type, act_type, group_size, zero_points, has_g_idx, out_type=None` | 同上 `:15-23` |
| `_transform_param` 用 `replace_parameter` 换回 `torch.nn.Parameter`（Dynamo 兼容） | 同上 `:69-79` |
| `can_implement` 首先要求 `VLLM_WNA16_EXLLAMA=1`，且 `has_g_idx` 必须 False | `container_src/wna16_exllama.py:136-146` |
| 现路径调用式：`ops.gptq_gemm_out(c_out, x_2d, w_q, w_zp, w_s, w_g_idx, True, True, bits, temp_dq)`，且 **c 用 atomicAdd 累加 ⇒ 调用前必须 `zero_()`** | 同文件 `:296-300`、记忆 [[exllama-fused-kernel-atomic-add]] |
| 钉频脚本：`p0_clock.sh lock [sclk_lvl] [mclk_lvl]`，**level 8 = 1700 MHz（交付态）、4 = 1425、mclk 2 = 1000**；`pp_od_clk_voltage` 在这两卡上不存在 | `p0_clock.sh:16-20` |
| 5 个 per-rank 形状（去重后）：`(5120,7168) (3072,5120) (5120,17408) (8704,5120) (5120,8192)`；每 token 256 次 GEMV，192 次需 Split-K | 设计 `:100-110` |
| **VALU 天花板**：3.85 op/weight ⇒ 860 GB/s；4.25 ⇒ 780 GB/s；实测 `BW_eff` 928–931 GB/s ⇒ **op/weight 是主 KPI** | 设计 `:134-158` |
| 交付态 serve 基线（P0.5 后）：**32.4 tok/s**，W4A16 GEMV 桶 **19 867 µs/step**，地板 **6 788 µs** ⇒ 头寸 13.1 ms | `docs/.../2026-09-08-gfx906-p05-report.md`「交付态复测」 |

## 硬约束（长期有效，覆盖本计划每一步）

1. **回复用中文**。
2. **宿主由我ssh 自己驱动，绝不让用户代跑命令**；>2 分钟的 GPU 作业一律 `nohup setsid ... > /绝对路径.log 2>&1 < /dev/null &` 分离跑（记忆 [[gfx906-host-instability]]、[[gfx906-env-pitfalls]] 第 10 条）。
3. **凡是往容器里喂脚本一律 `docker exec -i`**（漏 `-i` 会让 heredoc 静默丢弃，本仓库刚踩过）。
4. **永不修改 `git config`**；提交用行内 `git -c user.name=362132718 -c user.email=362132718@users.noreply.github.com`。**只有用户明确要求时才 commit/push**（记忆 [[github-repo-and-push-route]]）。
5. **不碰 site-packages、不进交付镜像**。P1 任何产物都不需要走 B→A 重建；如果哪个步骤让你觉得"必须改容器里的 python 才能测"，那是设计错误，停下来汇报。
6. **钉频必须两把一起钉**（`lock 8 2`），并在每次采样时把 `status` 回显进日志；只钉 sclk 会把 mclk 落在 350 MHz，所有带宽数 2-3× 失真且无人报错（[[gfx906-env-pitfalls]] 第 8 条）。
7. **L1 只跑单卡**。设计文档要求的是每形状的内核时间，不需要 TP2；单卡把 init 发热砍一半，直接降低静默掉电概率，也不需要 RCCL 参与。
8. 每形状测完记录 `rocm-smi` 温度；任一卡 ≥100°C 立即停、冷却、把该档降到 level 4 重测并在报告里标注口径变化（热降频会让数字系统性偏慢，见 [[gfx906-host-instability]]）。
9. 报错就说报错。P1 的价值在于**证伪**，L1 全形状 <1.1× 是一个完全可接受的答案。

## 风险与预处置

| 风险 | 处置 |
|---|---|
| `load_inline` 在容器里编不出来（缺 hipcc/headers/`--offload-arch`） | Task 0 是门，编不出来就停；后备方案（Task 0 Step 5）用 `hipcc -shared` 手工编一个 `.so` 由 python `ctypes`/`torch.ops.load_library` 载入 |
| 1700 MHz 锁频 + 双卡曾 5 分钟内触发热掉电 | 单卡（约束 7）+ 每形状间 `sleep 30` + ≥100°C 自动降级（约束 8）+ 全部作业分离跑，掉电只损失该次采样 |
| SHAPEDBG 在 dynamo 编译区内 ⇒ 记录数 ≠ 调用数 | 只用于**形状集合**；M=1 由 P0 的 device-trace launch 计数交叉证明（Task 2 Step 4） |
| 设计文档内部矛盾：`:100` 说"只有 5 个形状"，`:181` 说"§2 表里那 7 个 per-rank 组合" | Task 2 以运行时实测为准，把矛盾写进报告，不猜 |
| ISA 数指令时内联/展开导致无法定位热循环 | Task 4 Step 6 用 `-mllvm -unroll-count=1` 单独编一份"不展开"产物数指令，另存一份按 unroll 8 的产物用于计时；两者只用于各自目的 |

## 文件结构

| 文件 | 职责 |
|---|---|
| `p1_jit_gate.py` | Task 0：证明 JIT 编译+启动通道可用，记录编译耗时与工具存在性 |
| `p1_l0_mapping.py` | Task 1：L0a 桥接布局 vs 规范（纯 python 精确）；L0b 现成内核的 one-hot 直读 |
| `p1_shape_probe.sh` | Task 2：起一次 serve 抓真实 (m,k,n)，去重后与设计表互校 |
| `p1_floor.py` | Task 3：每形状真实 HBM 字节分母 + 地板 µs + `BW_eff` 重新标定 |
| `p1_kernels.cu` | Task 4：候选内核变体（一个模块，`variant` 运行期选择）+ reduce-launch 探针核 |
| `p1_l1_bench.py` | Task 4：微基准 harness（eager / in-graph、中位数、温度、ISA op/weight） |
| `docs/superpowers/plans/2026-09-08-p1-nibble-map.md` | Task 1 产物：位级映射表（实现文档，P2 直接引用） |
| `docs/superpowers/plans/2026-09-08-p1-report.md` | Task 5 产物：L1 结果 + P2 go/kill 判定 |
| `logs/p1/*` | 全部原始日志（`.gitignore` 已放行 `p0_*`/`p05_*`，本计划新增 `p1_*` 白名单） |

---

## Task 0: JIT 编译通道门（G0）

**Files:**
- Create: `p1_jit_gate.py`

- [x] **Step 1: 写门测试**

```python
# p1_jit_gate.py -- can we JIT-compile and launch a gfx906 HIP kernel in this
# container at all? P1's whole microbenchmark plan hangs on this, so prove it
# before designing anything around it. Nothing here touches site-packages.
import os, shutil, subprocess, sys, time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")   # ROCm ignores it; silences a warning
import torch
from torch.utils.cpp_extension import load_inline

print("torch", torch.__version__, "hip", torch.version.hip,
      "device", torch.cuda.get_device_name(0), flush=True)

_CPP = "torch::Tensor p1_double(torch::Tensor x, int64_t variant);"
_CU = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
__global__ void p1_double_k(const __half* x, __half* y, int n, int variant) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = __half2float(x[i]);
    // variant is only here to prove the runtime-switch pattern compiles.
    y[i] = __float2half(v * 2.0f + (float)(variant % 3));
  }
}
torch::Tensor p1_double(torch::Tensor x, int64_t variant) {
  auto y = torch::empty_like(x);
  int n = (int)x.numel();
  p1_double_k<<<(n + 255) / 256, 256>>>(
      (const __half*)x.data_ptr(), (__half*)y.data_ptr(), n, (int)variant);
  return y;
}
"""
t0 = time.time()
mod = load_inline(
    name="p1_gate",
    cpp_sources=_CPP,
    cuda_sources=_CU,
    extra_cuda_cflags=["--offload-arch=gfx906", "-O2"],
    verbose=False,
    build_directory="/tmp/p1_gate_build",
)
ct = time.time() - t0
print("compile_seconds=%.1f" % ct, flush=True)

x = torch.randn(4096, device="cuda", dtype=torch.float16)
for v in (0, 1, 2):
    y = mod.p1_double(x, v)
    want = (x.float() * 2.0 + (v % 3)).half()
    bad = (y.float() - want.float()).abs().max().item()
    assert bad < 2e-3, "variant %d wrong: maxdiff %g" % (v, bad)
print("KERNEL_OK variants=0,1,2", flush=True)

# A wave64 assumption P2 leans on: this device must report 64 lanes.
prop = torch.cuda.get_device_properties(0)
print("wavefront_size=%d warp_size_attr=%s" % (
    getattr(prop, "warp_size", -1), getattr(prop, "warp_size", None)), flush=True)
assert getattr(prop, "warp_size", 64) == 64, "unexpected wave size -- wave64 math invalid"

for tool in ("hipcc", "llvm-objdump", "objdump"):
    print("tool %-14s %s" % (tool, shutil.which(tool) or "MISSING"), flush=True)
rc = subprocess.run(["python3", "-c", "import torch;print(torch.cuda.mem_get_info(0)[0]>10e9)"],
                    capture_output=True, text=True)
print("vram_gt_10GB=%s" % rc.stdout.strip(), flush=True)
print("JIT-OK", flush=True)
```

- [x] **Step 2: 同步到宿主并跑（首次编译较慢，但仍是前台可等的量级）**

Run: `./p0_sync.sh p1_jit_gate.py && ssh x@192.168.10.11 "docker exec -i vllm-gfx906 mkdir -p /tmp/p1_gate_build && docker exec -i -w /work/p1 vllm-gfx906 python3 /work/p1/p1_jit_gate.py"`
Expected: `compile_seconds=<N>`、`KERNEL_OK variants=0,1,2`、`wavefront_size=64`、三个 tool 行、末行 `JIT-OK`，退出码 0。

- [x] **Step 3: 记录编译耗时（决定 Task 4 的扫参经济）**

把 `compile_seconds` 抄进执行记录。若 >180 s，Task 4 必须保持"一个模块、变体作运行期参数"的写法（已是计划默认），不得拆成多模块。

- [x] **Step 4: 若 Step 2 失败，走后备通道再停**

Run（后备：手工 hipcc 共享库）:
```bash
ssh x@192.168.10.11 'docker exec -i vllm-gfx906 bash -c "mkdir -p /tmp/p1h && cat > /tmp/p1h/k.cpp <<'"'"'EOF'"'"'
#include <hip/hip_runtime.h>
extern \"C\" int p1_probe() { int n=0; hipDeviceGetCount(&n); return n; }
EOF
hipcc --offload-arch=gfx906 -shared -fPIC -o /tmp/p1h/libp1h.so /tmp/p1h/k.cpp && echo HIPCC-OK"'
```
Expected: `HIPCC-OK`（说明通道在，只是 `load_inline` 的接线需要修）⇒ 把两种通道的诊断都写进记录，**停**，等决定；若 hipcc 本身不可用 ⇒ **G0 失败，整个 P1 停摆**，如实汇报（此时 P2 也无从谈起，设计文档的"构建成本"一节需重开）。

- [x] **Step 5: Commit（用户要求时才做）**

```bash
git add p1_jit_gate.py
git commit -m "p1: prove the gfx906 JIT kernel channel before any microbenchmark design"
```

---

## Task 1: L0 位映射钉死（G1）

**Files:**
- Create: `p1_l0_mapping.py`
- Create: `docs/superpowers/plans/2026-09-08-p1-nibble-map.md`

- [x] **Step 1: 写 L0a —— 桥接布局 vs 规范（纯 python，零容差）**

这一步先红后绿：断言直接按设计文档写的位规则来，如果桥接不是这么做的，测试会炸，那正是我们要知道的。

```python
# p1_l0_mapping.py -- L0: pin the bit-level mapping down.
#   L0a: the bridge's repack matches the documented shuffle_4bit_8 rule exactly
#        (pure python, zero tolerance -- layout questions must not be numeric).
#   L0b: the SHIPPED kernel, fed with a one-hot x = e_k, reproduces the analytic
#        dequant value for every (k, n) => the mapping is right where it counts.
# No site-packages is touched; this only imports and calls.
import os
os.environ["VLLM_WNA16_EXLLAMA"] = "1"      # can_implement() requires it
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig)
from vllm.model_executor.kernels.linear.mixed_precision.wna16_exllama import (
    WNA16ExllamaKernel)
from vllm.scalar_type import scalar_types

DEV = "cuda"
G = 128
BITS = 4


def codes_for(K, N):
    """Codes that make every (k, n) position distinguishable and every single
    change of code distinguishable by value: 0..15, structured, not random.
    Random data would let an off-by-one in the interleave hide behind collisions.
    """
    k = torch.arange(K, device=DEV, dtype=torch.int32).view(-1, 1)
    n = torch.arange(N, device=DEV, dtype=torch.int32).view(1, -1)
    return ((k * 7 + n * 13) % 16).to(torch.int8)     # [K, N]


def pack_checkpoint_format(codes_KN):
    """compressed-tensors pack-quantized: [N, K//8] int32, along K, LSB-first."""
    K, N = codes_KN.shape
    c = codes_KN.t().contiguous().to(torch.int32)          # [N, K]
    w = c.view(N, K // 8, 8)
    bits = torch.arange(8, device=c.device, dtype=torch.int32) * BITS
    return ((w << bits.view(1, 1, -1)) & 0xFFFFFFFF).sum(dim=2, dtype=torch.int64) \
        .to(torch.int32).contiguous()                       # [N, K//8]


def expected_qweight(codes_KN, dst_bits):
    """Reference: what qweight must be, computed independently of the bridge."""
    K, N = codes_KN.shape
    grouped = codes_KN.to(torch.int64).view(K // 8, 8, N)
    return (grouped << dst_bits.view(1, -1, 1)).sum(dim=1, dtype=torch.int64) \
        .to(torch.int32).contiguous()                        # [K//8, N]


def pack_zp_checkpoint_format(zp_KG_N):
    """[K//G, N] int32 (0..15) -> [N//8, K//G] int32 packed along N; nibble j of
    word (n8, g) is column 8*n8 + j (bridge :212-223).

    Deliberately NOT the bridge's own `_pack_gptq_sequential` -- L0 has to
    cross-check the bridge, and sharing its packer makes the test circular.
    """
    Kg, N = zp_KG_N.shape
    z = zp_KG_N.t().contiguous().view(N // 8, 8, Kg).to(torch.int64)
    bits = torch.arange(8, device=zp_KG_N.device, dtype=torch.int64) * 4
    return (z << bits.view(1, -1, 1)).sum(dim=1, dtype=torch.int64) \
        .to(torch.int32).contiguous()                        # [N//8, K//G]


def build_layer(K, N, codes_KN, scales_KG_N, zp_KG_N):
    """A bare module holding the three params in their CHECKPOINT forms:
    weight_packed [N, K//8], weight_scale [N, K//G], weight_zero_point
    [N//8, K//G]. Feeding zp as [N, K//G] instead makes the bridge die inside its
    own permute/reshape -- reproduced: "shape '[16, 2]' is invalid for input of
    size 256". If plain nn.Parameter is rejected by permute_param_layout_, wrap
    these in BasevLLMParameter rather than reimplementing the transform.
    """
    from torch import nn

    class L(nn.Module):
        pass

    layer = L()
    for name, t in (("weight_packed", pack_checkpoint_format(codes_KN)),
                    ("weight_scale", scales_KG_N.t().contiguous()),              # [N, K//G]
                    ("weight_zero_point", pack_zp_checkpoint_format(zp_KG_N))):  # [N//8, K//G]
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    return layer


def kernel_for(K, N):
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4, act_type=torch.float16,
        group_size=G, zero_points=True, has_g_idx=False)
    # w_zp param name is the checkpoint's; the bridge repacks in place.
    return WNA16ExllamaKernel(cfg, "weight_packed", "weight_scale",
                              "weight_zero_point")


def make_layer_and_kernel(K, N, seed=0):
    torch.manual_seed(seed)
    codes = codes_for(K, N)
    scales = (torch.rand(K // G, N, device=DEV, dtype=torch.float16) * 0.02 + 0.001)
    zp = torch.randint(0, 16, (K // G, N), device=DEV, dtype=torch.int32)
    layer = build_layer(K, N, codes, scales, zp)
    kern = kernel_for(K, N)
    return kern, layer, codes, scales, zp


def test_L0a(K=512, N=256):
    """Documented rule: k index i lands at bits (i//2)*4 + (i%2)*16."""
    i = torch.arange(8, device=DEV, dtype=torch.int64)
    dst_bits = (i // 2) * BITS + (i % 2) * 16
    kern, layer, codes, *_ = make_layer_and_kernel(K, N)
    kern.process_weights_after_loading(layer)
    want = expected_qweight(codes, dst_bits.to(DEV))
    got = layer.weight_packed.data
    assert got.shape == want.shape, "qweight shape %s want %s" % (tuple(got.shape), tuple(want.shape))
    diff = (got.to(torch.int64) != want.to(torch.int64))
    print("L0a K=%d N=%d words=%d mismatched_words=%d" % (K, N, got.numel(), int(diff.sum())))
    assert int(diff.sum()) == 0, "bridge layout != documented shuffle_4bit_8"
    # scales/zp must end up [K/G, N] contiguous
    assert tuple(layer.weight_scale.shape) == (K // G, N), layer.weight_scale.shape
    print("L0a PASS (scales %s dtype %s)" % (tuple(layer.weight_scale.shape), layer.weight_scale.dtype))
    return dst_bits, codes, layer, K, N
```

- [x] **Step 2: 跑 L0a，先看它是否真的绿**

Run: `./p0_sync.sh p1_l0_mapping.py && ssh x@192.168.10.11 "docker exec -i -w /work/p1 vllm-gfx906 python3 -c 'import sys; sys.path.insert(0,\"/work/p1\"); import p1_l0_mapping as m; m.test_L0a()'"`
Expected: `L0a ... mismatched_words=0` + `L0a PASS`。
若 `mismatched_words != 0` ⇒ **停**：设计文档 §映射 的位规则与桥接实现不一致，P2 的前提塌了。把前 3 个不匹配字（`torch.nonzero(diff)[:3]`）与两边值抄进记录。

- [x] **Step 3: 加 L0b —— 现成内核的 one-hot 直读**

追加到 `p1_l0_mapping.py`：

```python
def dequant_analytic(code_int8, scale_f16, zp_int32):
    """The reference every variant must match: (q - zp) * scale, fp32 math."""
    return (code_int8.to(torch.float32) - zp_int32.to(torch.float32)) * scale_f16.to(torch.float32)


def test_L0b(K=512, N=256):
    """x = e_k makes out[:, n] the dequantized weight at that single (k, n).
    That turns 'is the nibble mapping right' into a zero-tolerance structural
    readout instead of an error-budget argument."""
    dst_bits, codes, layer, K, N = test_L0a(K, N)
    w_q = layer.weight_packed.data.contiguous()
    w_s = layer.weight_scale.data.contiguous()
    # After the transform this IS the packed qzeros [K//G, N//8] int32 -- that is
    # the form gptq_gemm_out wants (torch_bindings.cpp schema; the kernel indexes
    # N/8 words and takes nibble n%8). Unpack it ONLY to build the reference.
    z_packed = layer.weight_zero_point.data.contiguous()
    assert z_packed.shape == (K // G, N // 8) and z_packed.dtype == torch.int32, \
        "unexpected qzeros form %s %s" % (tuple(z_packed.shape), z_packed.dtype)
    shift = torch.arange(32 // BITS, device=DEV, dtype=torch.int64) * BITS
    u = ((z_packed.to(torch.int64).unsqueeze(-1) >> shift.view(1, 1, -1)) & 0xF)
    u = u.reshape(K // G, N).to(torch.int32)               # [K/G, N], reference only
    print("L0b qzeros packed=%s unpacked=%s" % (tuple(z_packed.shape), tuple(u.shape)))

    x = torch.zeros((1, K), device=DEV, dtype=torch.float16)
    c_out = torch.zeros((1, N), device=DEV, dtype=torch.float16)
    # M=1 takes the fused branch, where tdq is nullptr (q_gemm.cu:1829-1836,
    # :1884-1892) and production passes an empty tensor -- mirror that, do not
    # allocate a [K, N] fp16 buffer (178 MB at the largest shape) for nothing.
    temp_dq = torch.empty(0, dtype=torch.float16, device=DEV)
    g_idx = layer.g_idx
    max_ulp, worst = 0.0, None
    for k in range(0, K, max(1, K // 32)):               # 32 spot positions
        x.zero_(); x[0, k] = 1.0; c_out.zero_()
        ops.gptq_gemm_out(c_out, x, w_q, z_packed, w_s, g_idx,
                          True, True, BITS, temp_dq)
        g = k // G
        want = dequant_analytic(codes[k], w_s[g], u[g])
        err = (c_out[0].float() - want).abs()
        rel = float((err / want.abs().clamp(min=1e-6)).max())
        if rel > max_ulp:
            max_ulp, worst = rel, (k, int(err.argmax()))
    print("L0b positions=32 max_rel=%.3e worst=%s" % (max_ulp, worst))
    # One-hot removes cancellation entirely: anything above fp16 representable
    # error means the kernel read a different (k, n) than we addressed.
    assert max_ulp < 1e-3, "kernel dequant does not match the mapping (rel %g)" % max_ulp
    print("L0b PASS")
```

- [x] **Step 4: 跑 L0b**

Run: `ssh x@192.168.10.11 "docker exec -i -w /work/p1 vllm-gfx906 python3 -c 'import sys; sys.path.insert(0,\"/work/p1\"); import p1_l0_mapping as m; m.test_L0b()'"`
Expected: `L0a PASS` 后接 `L0b positions=32 max_rel=<~1e-4>` 与 `L0b PASS`。
`gptq_gemm_out` 参数序若报错（签名不匹配），按 `container_src/_custom_ops.py:598` 的真实签名改，不要改断言。

- [x] **Step 5: 扩到全部 5 个真实形状的 (K mod 128, N mod 8) 邻域**

```python
SHAPES = [(5120, 7168), (3072, 5120), (5120, 17408), (8704, 5120), (5120, 8192)]


def main():
    test_L0a()
    test_L0b()
    for K, N in SHAPES:                       # full size: proves the K/G group walk
        test_L0b(K, N)
    print("L0 ALL PASS")


if __name__ == "__main__":
    main()
```

Run: `ssh x@192.168.10.11 "docker exec -i -w /work/p1 vllm-gfx906 python3 /work/p1/p1_l0_mapping.py" | tee logs/p1/p1_l0_mapping.log`
Expected: 5 行 `L0b PASS`（每形状一次）+ 末行 `L0 ALL PASS`。**这就是 G1**。
若只有大形状失败 ⇒ 几乎一定是 group 边界/整除假设，把失败的 (K,N) 与 `K%128`、`N%8` 一起记录。

- [x] **Step 6: 写位级映射实现文档（P2 直接引用它）**

创建 `docs/superpowers/plans/2026-09-08-p1-nibble-map.md`，内容必须包含：

```markdown
# W4A16 位级映射（L0 实测钉死，非推导）

## 一个 int32 字里的 8 个 k 位置
qweight[kw, n] 的第 i 个 code（i = 0..7，对应 k = kw*8 + i）位于位
`(i//2)*4 + (i%2)*16`，即 k 顺序 -> bit 位置：

| i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| bit | 0-3 | 16-19 | 4-7 | 20-23 | 8-11 | 24-27 | 12-15 | 28-31 |

=> 提取"连续 4 个 k"（i=0..3）**不能**用顺序掩码移位，必须按上表打乱；
   这正是 `v_dot2_f32_f16` 的 pair 边界：(i0,i1) 与 (i2,i3) 各自在 dword 里连续。

## 三个张量的最终布局（内核入口所见）
- qweight:  [K/8, N] int32
- scales:   [K/128, N] fp16，contiguous
- qzeros:   [K/128, N/8] int32，沿 N 每 8 个输出打包一个 word，LSB-first；
            GPTQv2 编码：存储 nibble 就是真 zp（不减 1）
- g_idx:    空张量（无 act-order）
## 校验方式与结论
L0a mismatched_words=0；L0b 在 5 个真实形状上 max_rel=<实测值>（one-hot 直读）。
```
把 `<实测值>` 换成 Step 5 日志里的真实数字（**不许留尖括号**）。

- [x] **Step 7: Commit（用户要求时才做）**

```bash
git add p1_l0_mapping.py docs/superpowers/plans/2026-09-08-p1-nibble-map.md logs/p1/
git commit -m "p1: pin the W4A16 bit mapping with a one-hot kernel readout (L0)"
```

---

## Task 2: 真实形状与真实 M 探针（G2）

**Files:**
- Create: `p1_shape_probe.sh`
- Create: `p1_shapes_report.py`

- [x] **Step 1: 写探针脚本（一次 serve，两个 rank 各落一份 CSV）**

```bash
#!/usr/bin/env bash
# p1_shape_probe.sh -- ONE real serve with the shape-debug channel on, so the
# microbenchmark runs on shapes we observed rather than shapes we derived.
# Run inside the dev container. TP2 stays on because that is how the model
# actually serves; probe cost is one engine start.
#
# The dump does NOT happen by itself: wna16_exllama.py:70-85 registers
# _dump_shapes on atexit AND SIGUSR1, with the comment that atexit alone is not
# enough because vLLM hard-kills the TP workers on shutdown. So we signal the
# workers ourselves. The bracket in the pattern stops pkill from matching this
# very command line (see memory gfx906-env-pitfalls #10).
set -uo pipefail
MODEL=${MODEL:-/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16}
OUTD=${OUTD:-/work/p1}
mkdir -p "$OUTD"
rm -f "$OUTD"/p1_shapes.csv*
export VLLM_WNA16_EXL_SHAPEDBG="$OUTD/p1_shapes.csv"
export VLLM_WNA16_EXLLAMA=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0        # engine core in-process; atexit reliable
python3 - "$MODEL" <<'PY'
import sys, os, subprocess, time
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], tensor_parallel_size=2, dtype="float16",
          max_model_len=4096, gpu_memory_utilization=0.85,
          disable_custom_all_reduce=True, enforce_eager=True,   # eager: python runs every step
          limit_mm_per_prompt={"image": 1, "video": 1})
sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)  # full window
llm.generate(["Explain in one paragraph why HBM bandwidth limits decode."], sp)
print("GENERATED 64 decode steps", flush=True)
rc = subprocess.run(["pkill", "-USR1", "-f", "VLLM::[W]or"])
print("pkill -USR1 rc=%s (1 means no worker matched -- then signal EngineCore)", rc.returncode)
if rc.returncode != 0:
    subprocess.run(["pkill", "-USR1", "-f", "[V]LLM::EngineCore"])
time.sleep(5)
PY
echo "shapedbg files (0-byte or missing = the dump never fired):"
ls -l "$OUTD"/p1_shapes.csv* 2>/dev/null || echo "NO SHAPE FILES"
```

**为什么用 `enforce_eager=True`**：Python 钩子只在 eager 下每步执行（[[gfx906-env-pitfalls]] 第 7 条）。形状**集合**在两种模式下都拿得到，但 eager 能证明"每一步都见到这些形状"，而 graph 内 M 只能靠 device trace（Step 4 用 P0 已有的 trace 数据交叉）。

- [x] **Step 2: 分离跑（一次引擎启动 ~8 分钟）**

Run: `./p0_sync.sh p1_shape_probe.sh && ssh x@192.168.10.11 "docker exec -d vllm-gfx906 bash -c '/work/p1/p1_shape_probe.sh > /work/p1/p1_shape_probe.log 2>&1' ; echo started"`
之后**不轮询**；用一条等待命令后台挂着，见到 `GENERATED 64 decode steps` 即完成。

- [x] **Step 3: 写去重与互校脚本**

```python
# p1_shapes_report.py -- dedupe per-rank shapedbg CSVs and cross-check against
# the design's shape table. A mismatch means our model of fusion/sharding is
# wrong, which is exactly what this step is for.
import csv, glob, collections

DESIGN = {                      # (K, N) per rank, from design :102-108
    (5120, 7168):  "attn qkv_proj",
    (3072, 5120):  "attn o_proj + GDN out_proj",
    (5120, 17408): "mlp gate_up_proj",
    (8704, 5120):  "mlp down_proj",
    (5120, 8192):  "GDN in_proj_qkvz",
}
seen = collections.Counter()
ms = collections.defaultdict(set)
for path in glob.glob("/work/p1/p1_shapes.csv.*"):
    with open(path) as f:
        for row in csv.DictReader(f):
            m, k, n = int(row["m"]), int(row["k"]), int(row["n"])
            seen[(k, n)] += 1
            ms[(k, n)].add(m)

print("distinct (K,N) observed = %d   design table has %d" % (len(seen), len(DESIGN)))
for (k, n), rows in sorted(seen.items()):
    tag = DESIGN.get((k, n), "*** NOT IN DESIGN TABLE ***")
    print("  K=%-6d N=%-6d csv_rows=%-4d m_values=%s  128|K=%s 8|N=%s  %s" % (
        k, n, rows, sorted(ms[(k, n)]), k % 128 == 0, n % 8 == 0, tag))
extra = set(seen) - set(DESIGN)
missing = set(DESIGN) - set(seen)
print("extra=%s missing=%s" % (sorted(extra) or "-", sorted(missing) or "-"))
assert not extra, "observed shapes outside the design table -> fix §2 first"
assert not missing, "designed shapes never observed -> the model path differs from §2"
print("SHAPE-GATE PASS")
```

Run: `./p0_sync.sh p1_shapes_report.py && ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 /work/p1/p1_shapes_report.py" | tee logs/p1/p1_shapes.log`
Expected: `distinct (K,N) observed = 5`、`SHAPE-GATE PASS`，且 `m_values` 里应出现 `{1, 130, ...}`（decode 的 1 与 prefill 的长度）。
若 `m_values` **没有 1**：说明这套 python 钩子根本没记录到 batch1 decode（可能被 CAPTURE 尺寸分档改变了 M），走 Step 4。

- [x] **Step 4: 用 device trace 证明"graph 内 M 真的是 1"（不靠 python）**

P0 已经落过 trace 聚合产物（`logs/p0/p05_buckets_ab_on.log` 等）。用现成链路核：decode 窗口内 `w4a16_gemv` 桶的 launch 次数应为 **256 次/step**，且 `FULL_DECODE_ONLY` 的 capture sizes 从 1 起。

Run: `ssh x@192.168.10.11 "docker exec -i -e OUT=/work/p0/ab_on -e PROF_DIR=/work/p0/ab_on/prof vllm-gfx906 python3 /work/p0/p0_buckets.py" | grep -E "w4a16_gemv|launches|ROUTING"`
Expected: `w4a16_gemv` 行 launches=16384（= 256 × 64 步窗口）。这与"batch1 时每步 256 次 M=1 GEMV"一致 ⇒ 真实 M 得证。
若该目录已被清 ⇒ 重跑 `p05_ab.sh` 的单臂代价太大，改为接受 python 钩子的 M 集合并**在报告里明确标注 M=1 的证据等级降为间接**，不要假装有直接证据。

- [x] **Step 5: 把"5 还是 7"这个设计内部矛盾钉掉**

设计 `:100` 说去重后 5 个形状，`:181` 说"§2 表里那 7 个 per-rank 组合"。以 Step 3 的实测计数为准，在本计划「执行记录」里写清是哪个数，并**回头改设计文档那一行**（改文档不改代码）。

- [x] **Step 6: Commit（用户要求时才做）**

```bash
git add p1_shape_probe.sh p1_shapes_report.py logs/p1/
git commit -m "p1: pin the real per-rank WNA16 shape set and prove decode M=1"
```

---

## Task 3: 地板与真实字节分母（G3）

**Files:**
- Create: `p1_floor.py`

- [x] **Step 1: 写地板计算器 + BW_eff 重新标定**

```python
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
    GEMV's neighbourhood does, and a stricter upper bound on sustained bandwidth."""
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
        "measured read bandwidth %.1f GB/s is below the 90%% of the %.0f floor -- "
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
    json.dump({"bw_pure_read_gb_s": br, "bw_read_write_gb_s": bw,
               "bw_floor_ref": BW_FLOOR_REF, "bw_read_ref": BW_READ_REF,
               "sum_floor_us_per_token": total_full, "shapes": shapes},
              open("/work/p1/p1_floors.json", "w"), indent=1)
    print("wrote /work/p1/p1_floors.json")
    print("FLOOR-OK")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: 钉频到交付档再跑（level 8 sclk + level 2 mclk）**

Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh lock 8 2 && docker exec -i vllm-gfx906 python3 /work/p1/p1_floor.py && bash /home/x/vllm-work/p0/p0_clock.sh status" | tee logs/p1/p1_floor.log`
Expected: `BW_eff_measured≈900-940`、5 行形状表、`SUM floors` 与 P0 的 6 788 µs/token 同量级（±15%）、`FLOOR-OK`。
`BW_eff` 明显偏低 ⇒ 先看 `status` 里 mclk 是不是 350 MHz（约束 6 的那个坑），不要先怀疑卡。

**注意**：`lock 8` 是交付态 1700 MHz。锁完后**必须立刻 `p0_clock.sh unlock`**（Task 4 的每轮结束时同理），宿主长时间钉在 1700 会顶到热墙。

- [x] **Step 3: 解锁 + 记录**

Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh unlock && bash /home/x/vllm-work/p0/p0_clock.sh status"`
Expected: 两卡 `level=auto`、sclk 860 / mclk 350（待机态）。
把 `SUM floors` 抄进执行记录 —— Task 5 的"距地板百分之几"全用它。

- [x] **Step 4: Commit（用户要求时才做）**

```bash
git add p1_floor.py logs/p1/p1_floor.log
git commit -m "p1: measure the per-shape HBM byte floor and recalibrate BW_eff at delivery clocks"
```

---

## Task 4: L1 微基准主体（G4）

**Files:**
- Create: `p1_kernels.cu`
- Create: `p1_l1_bench.py`

### 先说清楚三个变体各自在赌什么

| 变体 | 赌注 | 预测（写死，事后不许改口） |
|---|---|---|
| **V1** | 移位+掩码+fp16 magic，zp 提升出 K 循环 | 基准。每 int32 字 12 条提取类 VALU（4×(shr+and+add)）+ 4 条 dot2 |
| **V3** | V1 的结构，但 zp **逐元素**减（`v_pk_sub_f16`） | 每字多 4 条 ⇒ 预测 op/weight 4.25 vs V1 3.85、慢约 10%。若 ISA 数出来相同 ⇒ 编译器已替我们提升，设计预算表这一项作废 |
| **V2** | 设计 `:157` 提的 LDS LUT 换掉提取 | **修正后的预测：输**。本格式下一对的两个 nibble 相距 16 位 ⇒ 无法用"字节索引的 256 项 dword LUT"（设计设想的那个）一次读出一对，只能每项一次 `ds_read` ⇒ 每字 **24 条**（V1 是 12 条），只是把 VALU 换成了 LDS；而且它直接给出精确 `fp16(c)`，连 `S_g` 都不需要。**这条预测已被我第一版的"12 条/平的"说错一次，所以以实测为准 —— V2 存在的唯一理由是量出 64 lane 打 16 个地址的 bank conflict 到底多贵** |

magic 偏移是这一节唯一的非平凡改进：`codes = ((w >> 4p) & 0x000F000F) + 0x64006400` 得到
`(1024+c0, 1024+c1)`，于是 `Σ (1024+c_i)·x_i = raw`，真点积 `= raw − 1024·S_g`。它与被提升的 zp 项
合并成**每 group 一次的 `acc += scale_g · (raw − (1024 + zp) · S_g)`** ⇒ 去偏移**零额外指令**。

- [x] **Step 1: 写候选内核模块（一个模块，variant/splitk/unroll 全运行期）**

```cpp
// p1_kernels.cu -- prototype M=1 wave64 W4A16 GEMVs for L1. Measurement only:
// nothing here is meant to enter the delivery image, and P2 re-derives its own
// structure from whichever variant wins inside a CUDA graph.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cuda_fp16.h>

#define WAVE 64            // gfx906 is wave64. Not warpSize (not constexpr on ROCm 7).
#define GROUP 128
#define MAGIC 0x64006400u  // two fp16 1024.0 halves

__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
  float r;
  asm volatile("v_dot2_f32_f32_placeholder" : "=v"(r));   // Step 1a replaces this body
  return r;
}

// One wavefront owns 64 consecutive outputs (lane l -> n0+l) and walks K, so its
// 64 int32 loads of qweight[kw, n0..n0+63] are 256 B fully coalesced.
template <int UNROLL, int VARIANT>
__global__ __launch_bounds__(WAVE) void gemv(
    const __half* __restrict__ x, const int* __restrict__ qw,
    const __half* __restrict__ sc, const int* __restrict__ zr,
    float* __restrict__ part, int K, int N, int kper) {
  __shared__ __half xs[GROUP];
  __shared__ float Sg;
  __shared__ unsigned short half_lut[16];
  const int lane = threadIdx.x & (WAVE - 1);
  // Clamp, never `return`: a divergent exit before the in-loop __syncthreads is
  // undefined. All five real N are multiples of 64, so this is free here, but P2
  // must not inherit a half-full-block hang.
  const int n = min(blockIdx.x * WAVE + lane, N - 1);
  if (VARIANT == 2 && lane < 16) half_lut[lane] = __half_as_ushort(__float2half((float)lane));
  __syncthreads();
  const int kend = min(blockIdx.y * kper + kper, K);
  float acc = 0.f;

  for (int g = (blockIdx.y * kper) / GROUP; g < kend / GROUP; ++g) {
    for (int t = lane; t < GROUP; t += WAVE) xs[t] = x[g * GROUP + t];
    __syncthreads();

    // S_g is lane-independent -> compute it once per group as a wave butterfly:
    // 2 cvt/add + 6 shfl_xor + 6 add ~= 16 wave slots. The first draft had lane 0
    // serially summing 128 elements (~384 slots, MORE than the 304-slot main
    // body), which would have made L1 measure that loop instead of the variants.
    float s = (float)xs[lane] + (float)xs[lane + WAVE];
    for (int m = 1; m < WAVE; m <<= 1) s += __shfl_xor(s, m);
    if (lane == 0) Sg = s;
    __syncthreads();
    const float sg = Sg;

    const int zz = __ldg(&zr[g * (N / 8) + (n >> 3)]);
    const float zp = (float)((zz >> ((n & 7) * 4)) & 0xF);
    // V3 subtracts a BIASED zero point, so (1024+c)-(1024+zp) == c-zp exactly and
    // no per-group correction is left. (fp16 small ints are NOT 0x0000+c, so the
    // earlier "codes without magic" form of V3 was unimplementable.)
    const __half2 z2b = __halves2half2(__float2half(1024.f + zp), __float2half(1024.f + zp));
    float raw = 0.f;

    for (int kb = 0; kb < GROUP / 8; kb += UNROLL) {
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const unsigned w = (unsigned)__ldg(&qw[((g * GROUP / 8) + kb + u) * N + n]);
        const __half* xp = &xs[(kb + u) * 8];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
          const unsigned xv = *reinterpret_cast<const unsigned*>(xp + 2 * p);
          unsigned codes;
          if (VARIANT == 2) {
            // half_lut holds the *fp16 bit pattern* of c, so the pair is just the
            // two halves combined -- and NO magic may be added afterwards (adding
            // 0x64006400 to packed fp16 values is meaningless arithmetic).
            //
            // This is the real finding the LUT branch buys: on this interleaved
            // layout the design's "1 op/nibble LUT" degenerates, because a
            // byte-indexed 256-entry dword LUT (one ds_read per pair) is impossible
            // when a pair's two nibbles sit 16 bits apart. V2 therefore costs TWO
            // ds_reads per pair (24 ops/word vs V1's 12) and, having produced exact
            // fp16(c), needs no S_g term at all -- its tail is V3's. So V1-vs-V2
            // isolates "buy VALU with LDS", and what decides it is whether 64 lanes
            // hitting 16 addresses bank-conflict. L1 measures that; nothing here
            // assumes the answer.
            const unsigned lo = half_lut[(w >> (4 * p)) & 0xF];
            const unsigned hi = half_lut[(w >> (16 + 4 * p)) & 0xF];
            codes = lo | (hi << 16);
          } else {
            codes = ((w >> (4 * p)) & 0x000F000Fu) + MAGIC;
          }
          if (VARIANT == 3) {
            const __half2 c = *reinterpret_cast<const __half2*>(&codes);
            const __half2 d = __hsub2(c, z2b);          // v_pk_sub_f16, 1 per pair
            codes = *reinterpret_cast<const unsigned*>(&d);
          }
          raw = dot2(raw, codes, xv);
        }
      }
    }
    const float scale = __half2float(__ldg(&sc[g * N + n]));
    // Only V1 carries a +1024 bias into the dot, so only V1 folds it into the
    // hoisted term. V2 hands back exact fp16(c); V3 subtracted (1024+zp) inside
    // the loop. Both therefore need no correction here.
    acc += (VARIANT == 1) ? scale * (raw - (1024.f + zp) * sg) : scale * raw;
    __syncthreads();
  }
  if (n == blockIdx.x * WAVE + lane) part[(long)blockIdx.y * N + n] = acc;
}

__global__ void p1_reduce(const float* __restrict__ part, __half* __restrict__ out,
                          int N, int splits) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.f;
  for (int s = 0; s < splits; ++s) a += part[(long)s * N + n];
  out[n] = __float2half(a);
}

// L1 answers design :110 with this instead of an assumption: what does ONE extra
// launch cost inside a captured graph?
__global__ void p1_noop(int* p) { if (p) *p = 0; }

torch::Tensor p1_gemv(torch::Tensor x, torch::Tensor qw, torch::Tensor sc,
                      torch::Tensor zr, torch::Tensor scratch, int64_t variant,
                      int64_t splitk, int64_t unroll, bool reduce) {
  const int K = (int)qw.size(0) * 8, N = (int)qw.size(1);
  const int kper = ((K / GROUP) / (int)splitk) * GROUP;
  TORCH_CHECK(kper >= GROUP, "splitk ", splitk, " leaves an empty K chunk for K=", K);
  auto out = torch::empty({N}, x.options().dtype(torch::kHalf));
  float* part = scratch.data_ptr<float>();
  const dim3 gx((N + WAVE - 1) / WAVE), gy((unsigned)splitk);
  #define GO(V, U) gemv<U, V><<<gx, WAVE>>>(x.data_ptr<at::Half>(), (const int*)qw.data_ptr(), \
      sc.data_ptr<at::Half>(), (const int*)zr.data_ptr(), part, K, N, kper)
  #define FOR(V) do { if (unroll == 4) GO(V, 4); else if (unroll == 8) GO(V, 8); else GO(V, 16); } while (0)
  if (variant == 1) FOR(1); else if (variant == 2) FOR(2); else FOR(3);
  #undef FOR
  #undef GO
  if (!reduce) {
    // Timing-only arm: returns the float partials, never pass it to check().
    // Splitk==1 + reduce=false vs splitk==1 + reduce=true isolates the DEPENDENT
    // reduce launch -- two independent noops would let the graph overlap them and
    // understate the cost.
    return scratch.view({(long)splitk, (long)N});
  }
  p1_reduce<<<(N + 255) / 256, 256>>>(part, out.data_ptr<at::Half>(), N, (int)splitk);
  // NOTE for P2: this helper allocates `out` per call, which torch's graph pool
  // absorbs during capture. The production kernel must keep writing into the
  // caller-owned capture-safe slab; copying this allocation pattern would break
  // capture.
  return out;
}

int64_t p1_noop_launch(torch::Tensor dummy) {
  p1_noop<<<1, 1>>>((int*)dummy.data_ptr());
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("p1_gemv", &p1_gemv);
  m.def("p1_noop_launch", &p1_noop_launch);
}
```

- [x] **Step 1a: 把 dot2 换成真实内联汇编，并修掉 V3 的过渡写法**

`v_dot2_f32_f16` 在 gfx906（SI）上是合法指令，但 ROCm 7 的头文件未必暴露 C++ 内置。替换：

```cpp
__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
  float r;
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %3"
               : "=v"(r) : "v"(a), "v"(b), "v"(acc));
  return r;
}
```

V3 分支上面写的是**半成品**（`__float2half2_rn` 只填了一个值、且 `+1024` 又减 zp 会双重偏移）。
Step 1a 必须改成：`c2 = __halves2half2(__float2half((float)((w>>(4p))&0xF) - zp),
__float2half((float)((w>>(16+4p))&0xF) - zp))`，即**逐元素减 zp、不带 magic**，
且 V3 的末尾累加式不含任何 `S_g` 修正项（模板里的 `(VARIANT==3)?0.f:1024.f+zp` 已经这样写了）。
若你更省事，可以让 V3 直接由 V1 的 codes 结果 `__hsub2(codes, z2_with_bias)` 实现——但两条路都必须
先过 Step 2 的正确性门。

Run（编译门）:
```bash
./p0_sync.sh p1_kernels.cu && ssh x@192.168.10.11 "docker exec -i vllm-gfx906 bash -c 'mkdir -p /tmp/p1_bench_build && cd /work/p1 && python3 -c \"from torch.utils.cpp_extension import load; load(name=\\\"p1_kernels\\\", sources=[\\\"/work/p1/p1_kernels.cu\\\"], extra_cuda_cflags=[\\\"--offload-arch=gfx906\\\",\\\"-O3\\\"], build_directory=\\\"/tmp/p1_bench_build\\\", verbose=True)\" 2>&1 | tail -20'"
```
Expected: 无 `error:`，且生成 `/tmp/p1_bench_build/p1_kernels.so`。
若 `v_dot2_f32_f16` 被汇编器拒绝 ⇒ 说明该指令在此工具链下拼写/操作数不同，用
`echo 'v_dot2_f32_f16 v0, v1, v2, v0' | llc -march=amdgcn -mcpu=gfx906 -filetype=asm -` 之类先验证
汇编器接受的形式，再改；**改不动就记为"dot2 通路不可用"，V1/V2/V3 全部退化为标量 fp32 乘加并明确
标注这不是设计假设的那把尺**。

- [x] **Step 2: 正确性门先于任何计时**

```python
# p1_l1_bench.py -- correctness first, then timing. A wrong variant is not fast,
# it is noise, and its microseconds are worthless to P2.
import json, os, sys, time
os.environ["VLLM_WNA16_EXLLAMA"] = "1"
import torch
from torch.utils.cpp_extension import load

mod = load(name="p1_kernels", sources=["/work/p1/p1_kernels.cu"],
           extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
           build_directory="/tmp/p1_bench_build", verbose=False)
DEV, G = "cuda", 128
# Exactly the bit positions L0 pinned: code i -> (i//2)*4 + (i%2)*16
KBIT = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)],
                    device=DEV, dtype=torch.int64)


def unpack(qw):
    """[K/8, N] int32 -> [K, N] int8 using the L0-pinned positions.

    The permute is mandatory, not stylistic: `[K/8, N, 8]` reshaped straight to
    `[K, N`] interleaves the two axes. Reproduced in the container -- row k=1 came
    out [5,11,1,9,13,3] instead of [8,9,10,11,12,13], and only the permuted form
    matched. NOTE: `unpack_zp()` below must NOT get the same change; there the
    packed dimension is already last and the row index g is untouched.
    """
    K8, N = qw.shape
    w = qw.to(torch.int64).unsqueeze(-1)
    c = (w >> KBIT.view(1, 1, 8)) & 0xF                  # [K/8, N, 8]
    c = c.permute(0, 2, 1).reshape(K8 * 8, N)            # [K/8,8,N] -> [K, N]
    return c.to(torch.int8)


def unpack_zp(zr, N):
    K_G, N8 = zr.shape
    z = zr.to(torch.int64).unsqueeze(-1)
    sh = torch.arange(8, device=DEV, dtype=torch.int64) * 4
    return ((z >> sh.view(1, 1, 8)) & 0xF).reshape(K_G, N8 * 8)[:, :N].to(torch.int32)


def reference(x, qw, sc, zr):
    K, N = qw.shape[0] * 8, qw.shape[1]
    code = unpack(qw).to(torch.float32)
    zp = unpack_zp(zr, N).to(torch.float32)
    acc = torch.zeros(N, device=DEV, dtype=torch.float32)
    for g in range(K // G):
        xg = x[g * G:(g + 1) * G].to(torch.float32)
        acc += sc[g].float() * ((xg.view(-1, 1) * code[g * G:(g + 1) * G]).sum(0)
                                - zp[g] * xg.sum())
    return acc


def tensors(K, N, seed=0):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), device=DEV, dtype=torch.int32,
                       generator=g)
    sc = (torch.rand(K // G, N, device=DEV, generator=g) * 0.02 + 0.001).half()
    zr = torch.randint(0, 2**31 - 1, (K // G, N // 8), device=DEV, dtype=torch.int32,
                       generator=g)
    x = torch.randn(K, device=DEV, generator=g).half()
    return x, qw, sc, zr


def selfcheck_reference():
    """The reference is itself a decode implementation. Prove it round-trips
    against Task 1's packer before letting it judge a kernel."""
    sys.path.insert(0, "/work/p1")
    import p1_l0_mapping as L0
    K, N = 512, 256
    codes = L0.codes_for(K, N)                                  # [K, N] int8
    qw = L0.pack_checkpoint_format(codes)
    # pack_checkpoint_format is the GPTQ-checkpoint order; Task 1 showed the
    # bridge converts it to the shuffled order, so rebuild qweight the same way
    # the bridge does and unpack THAT.
    i = torch.arange(8, device=DEV, dtype=torch.int64)
    dst = (i // 2) * 4 + (i % 2) * 16
    grouped = codes.to(torch.int64).view(K // 8, 8, N)
    shuffled = (grouped << dst.view(1, -1, 1)).sum(1, dtype=torch.int64).to(torch.int32)
    back = unpack(shuffled)
    ok = torch.equal(back.to(torch.int16), codes.to(torch.int16))
    print("SELF_CHECK reference round-trip:", "OK" if ok else "FAIL")
    assert ok, "our own unpack() disagrees with the bridge -- fix it before timing"


def check(variant, splitk=1, unroll=8, K=1024, N=256):
    x, qw, sc, zr = tensors(K, N)
    scratch = torch.zeros(splitk * N, device=DEV, dtype=torch.float32)
    got = mod.p1_gemv(x, qw, sc, zr, scratch, variant, splitk, unroll, True).float()
    # Round the fp32 reference to fp16 FIRST so output quantisation is not charged
    # as an implementation error; 2e-2 (41 fp16 ULP) would happily accept a partial
    # (k,n) mis-wiring. Measured fp32-accumulation noise is ~3e-5, so 5e-4 has room.
    want = reference(x, qw, sc, zr).half().float()
    rel = float(((got - want).abs() / want.abs().clamp(min=1e-3)).max())
    ok = rel < 5e-4
    print("CHECK variant=%d splitk=%d unroll=%d K=%d N=%d max_rel=%.3e %s" % (
        variant, splitk, unroll, K, N, rel, "OK" if ok else "FAIL"), flush=True)
    return ok
```

> **文件顺序要求（P1-6）**：`if __name__ == "__main__":` 这一段**必须落在整个文件的最后**
> （Step 3/4 会继续往文件里追加 `timeit` / `launch_cost` / `main` 等定义，守卫若留在中间，
> `python3 p1_l1_bench.py bench` 会在 `main()` 定义之前就调用它 ⇒ `NameError`）。
> Step 4 末尾给出这个最终守卫，Step 2 这里**先不要写守卫**。

```python
# (Step 2 continues -- no __main__ block yet; see the note above.)
```

Run（Step 2 单独验证，守卫还没进文件）:
```bash
./p0_sync.sh p1_l1_bench.py && ssh x@192.168.10.11 "docker exec -i -w /work/p1 vllm-gfx906 python3 -c '
import p1_l1_bench as b
b.selfcheck_reference()
bad = [v for v in (1,2,3) if not b.check(v)] + [v for v in (1,3) if not b.check(v, splitk=4)]
print(\"CORRECTNESS:\", \"ALL OK\" if not bad else \"FAILED %s\" % bad)'"
```
Expected: `SELF_CHECK reference round-trip: OK` → 5 行 `... OK` → `CORRECTNESS: ALL OK`。
某变体反复不过 ⇒ **它的计时作废**，报告里列为"未通过正确性门"。**不许为了变绿放宽 5e-4。**

- [x] **Step 3: 计时 harness（eager 与 graph 各一遍；现路径对照用同一把尺）**

追加到 `p1_l1_bench.py`（定义在 `main()` 之前）：

```python
def timeit(fn, warm=16, reps=64, R=24):
    """Median of `reps` timings, each covering R launches divided by R.

    A single launch per event pair folds graph-submission overhead (1-3 us) into
    kernel time; the smallest real shape's whole floor is 8.7 us, so that error is
    ~20% -- against a 1.1x decision threshold. R>1 amortises it. The incumbent MUST
    be timed through this same helper or the ratio is not a ratio."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(R):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / R)
    ts.sort()
    return ts[len(ts) // 2]                      # median us


def graphed(fn):
    fn(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    return lambda: gr.replay()


def exllama_baseline(x, qw, sc, zr, N):
    """The incumbent measured with THIS instrument -- same tensors, same event
    timing, same graph wrapper, same R. P0's log numbers are 1425 MHz / TP2 /
    per-step aggregates and are not a valid denominator here."""
    from vllm import _custom_ops as ops
    K = qw.shape[0] * 8
    x2 = x.view(1, K).contiguous()
    c_out = torch.zeros(1, N, device=DEV, dtype=torch.float16)
    temp_dq = torch.empty(0, dtype=torch.float16, device=DEV)   # M=1: unused
    g_idx = torch.empty((0,), dtype=torch.int32, device=DEV)

    def once():
        c_out.zero_()                            # atomicAdd accumulator: mandatory
        ops.gptq_gemm_out(c_out, x2, qw, zr, sc, g_idx, True, True, 4, temp_dq)
        return c_out
    return timeit(once), timeit(graphed(once))


def launch_cost(K=5120, N=8192):
    """Price the DEPENDENT reduce launch, not two independent noops the graph is
    free to overlap: same shape, splitk=1, with vs without the reduce kernel."""
    x, qw, sc, zr = tensors(K, N)
    scratch = torch.zeros(N, device=DEV, dtype=torch.float32)
    with_r = lambda: mod.p1_gemv(x, qw, sc, zr, scratch, 1, 1, 8, True)
    no_r = lambda: mod.p1_gemv(x, qw, sc, zr, scratch, 1, 1, 8, False)
    a = timeit(graphed(with_r)); b = timeit(graphed(no_r))
    marg = a - b
    print("REDUCE marginal in-graph = %.3f us (with=%.2f without=%.2f) => "
          "192/token would cost %.1f us/step" % (marg, a, b, 192 * marg), flush=True)
    return dict(with_reduce=a, no_reduce=b, marginal=marg, projected_192=192 * marg)


def temps_c():
    """Both cards, via the tool the hard constraint names. `temp1_input` on these
    VII-BIOS cards has not been confirmed to be the junction temp -- Step 5 of this
    task prints rocm-smi alongside once so the two can be reconciled by eye."""
    import subprocess
    o = subprocess.run(["rocm-smi", "--showtemp"], capture_output=True, text=True)
    vals = [int(float(t)) for t in re.findall(r"(\d+(?:\.\d+)?)\s*C", o.stdout)]
    return dict(zip(("card0", "card1"), vals[:2])) if len(vals) >= 2 else {}
```

- [x] **Step 4: 主扫参（读 Task 3 的 floors JSON；热护栏；增量落盘）**

```python
FLOORS = json.load(open("/work/p1/p1_floors.json"))
VARIANTS, SPLITKS, UNROLLS = (1, 2, 3), (1, 2, 4, 8, 16), (4, 8, 16)


def clock_status():
    """Read the ACTIVE DPM level of both cards from sysfs. If the container cannot
    see these nodes, drop this and have the sweep wrapper echo
    `p0_clock.sh status` from the host between shapes -- the requirement is that a
    driver falling back to auto, or thermal throttling, is visible in the log."""
    import glob, re
    out = []
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            act = {}
            for ck in ("sclk", "mclk"):
                line = [l for l in open("%s/pp_dpm_%s" % (d, ck)) if l.strip().endswith("*")]
                act[ck] = line[0].strip() if line else "?"
            out.append("%s %s %s" % (d.split("/")[-2], act["sclk"], act["mclk"]))
        except OSError:
            out.append("%s sysfs-unreadable" % d.split("/")[-2])
    return " | ".join(out) or "no-drm-sysfs"


def main():
    rows = []
    incumbent = {}
    json.dump(launch_cost(), open("/work/p1/p1_launch.json", "w"), indent=1)
    for key, meta in sorted(FLOORS["shapes"].items()):
        K, N = (int(v) for v in key.split(","))
        floor_us = meta["floor_us"]
        # Two independent weight buffers, rotated per call. A graph replaying 64x
        # over one buffer can park it in the 4 MiB L2 -- (3072,5120) is only 7.9 MB
        # -- which would inflate precisely the ratios the 1.15x rule reads.
        bufs = [tensors(K, N, seed=1), tensors(K, N, seed=2)]
        e_us, g_us = exllama_baseline(*bufs[0], N)
        incumbent["%d,%d" % (K, N)] = dict(eager_us=e_us, graph_us=g_us,
                                           floor_us=floor_us)
        json.dump(incumbent, open("/work/p1/p1_incumbent.json", "w"), indent=1)
        print("== K=%d N=%d floor=%.2fus incumbent eager=%.2f graph=%.2f (%.1f%% of floor)"
              % (K, N, floor_us, e_us, g_us, 100 * floor_us / g_us), flush=True)
        for v in VARIANTS:
            for sk in SPLITKS:
                if (K // G) % sk or sk > K // G:
                    continue
                for u in UNROLLS:
                    scratch = torch.zeros(sk * N, device=DEV, dtype=torch.float32)
                    try:
                        if not check(v, sk, u, K, N):
                            print("   skip v%d sk%d u%d: incorrect" % (v, sk, u)); continue
                        state = [0]

                        def fn():
                            b = bufs[state[0] % 2]
                            state[0] += 1
                            return mod.p1_gemv(b[0], b[1], b[2], b[3], scratch,
                                               v, sk, u, True)
                        med, gm = timeit(fn), timeit(graphed(fn))
                    except Exception as ex:
                        print("   skip v%d sk%d u%d: %s" % (v, sk, u, str(ex)[:90])); continue
                    rows.append(dict(K=K, N=N, variant=v, splitk=sk, unroll=u,
                                     eager_us=med, graph_us=gm,
                                     pct_of_floor=round(100 * floor_us / gm, 1),
                                     vs_incumbent=round(g_us / gm, 3)))
                    json.dump(rows, open("/work/p1/p1_l1_rows.json", "w"), indent=1)
        t = temps_c()
        print("   temps=%s clocks=%s" % (t, clock_status()), flush=True)
        if max(list(t.values()) or [0]) > 95:
            print("THERMAL-BACKOFF: rerun with `p0_clock.sh lock 4 2` and label the "
                  "clock regime in the report", flush=True)
            sys.exit(3)
        time.sleep(30)
    print("BENCH-ROWS %d" % len(rows))
    best = {}
    for r in rows:
        k = (r["K"], r["N"])
        if k not in best or r["graph_us"] < best[k]["graph_us"]:
            best[k] = r
    for k, r in sorted(best.items()):
        print("BEST K=%d N=%d variant=%d splitk=%d unroll=%d graph=%.2fus %.1f%%floor x%.2f" % (
            k[0], k[1], r["variant"], r["splitk"], r["unroll"], r["graph_us"],
            r["pct_of_floor"], r["vs_incumbent"]), flush=True)
    # 漏4: what matters is the aggregate, not per-shape wins and losses. A variant
    # that wins 1.3x on gate_up (64 calls) and loses on down_proj (64 calls) can
    # still be net positive -- and vice versa.
    tot_inc = tot_best = tot_floor = 0.0
    for key, meta in FLOORS["shapes"].items():
        K, N = (int(t) for t in key.split(","))
        calls = meta["calls_per_token"]
        tot_inc += incumbent[key]["graph_us"] * calls
        tot_best += best[(K, N)]["graph_us"] * calls
        tot_floor += meta["floor_us"] * calls
    print("WEIGHTED over 256 calls/token: incumbent=%.1f us  best=%.1f us  floor=%.1f us"
          " => aggregate x%.2f, and %.0f%% of incumbent time is still above the floor"
          % (tot_inc, tot_best, tot_floor, tot_inc / tot_best,
             100 * (tot_best - tot_floor) / tot_inc), flush=True)
    json.dump(dict(rows=rows, best={("%d,%d" % k): v for k, v in best.items()},
                   incumbent=incumbent, weighted=dict(incumbent_us=tot_inc,
                   best_us=tot_best, floor_us=tot_floor)),
              open("/work/p1/p1_l1_summary.json", "w"), indent=1)


if __name__ == "__main__":
    selfcheck_reference()
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        main()
    else:
        bad = [v for v in (1, 2, 3) if not check(v)]
        bad += [v for v in (1, 3) if not check(v, splitk=4)]
        print("CORRECTNESS:", "ALL OK" if not bad else "FAILED %s" % bad)
        sys.exit(1 if bad else 0)
```

- [x] **Step 5: 分离跑全量扫参（钉频 → 跑，解锁单独一步）**

```bash
./p0_sync.sh p1_kernels.cu p1_l1_bench.py && ssh x@192.168.10.11 \
 "bash /home/x/vllm-work/p0/p0_clock.sh lock 8 2; \
  nohup setsid docker exec vllm-gfx906 bash -c 'cd /work/p1 && python3 p1_l1_bench.py bench > /work/p1/p1_l1_bench.log 2>&1; echo BENCH-EXIT=\$? >> /work/p1/p1_l1_bench.log' > /dev/null 2>&1 < /dev/null & echo started"
```
解锁**放在作业结束后单独一条命令**里跑（不要塞进同一个 `bash -c` 链，也不要指望它和 `pkill` 共存）。
规模 = 5 形状 × 3 变体 × ≤5 splitK × 3 unroll，每组先 `check()` 再计时，预计 20–40 分钟。
之后不轮询，用一条后台等待命令盯 `BENCH-EXIT=`。

- [x] **Step 6: op/weight 从 ISA 数，不从计时反推**

```bash
ssh x@192.168.10.11 "docker exec -i vllm-gfx906 bash -c 'llvm-objdump -d /tmp/p1_bench_build/p1_kernels.so > /work/p1/p1_isa.s; wc -l < /work/p1/p1_isa.s'"
```

然后按**回边**切热循环体统计（整个符号会把 prologue/epilogue 一起算进去，那是上界不是答案）：

```python
# p1_isa_count.py -- VALU ops per weight, read off the ISA.
import re, sys
src = open("/work/p1/p1_isa.s").read()
for m in re.finditer(r"^([0-9a-f]+) <(gemv[^>]*)>:", src, re.M):
    name, start = m.group(2), m.end()
    body = src[start: start + src[start:].find("\n\n")]
    # hot loop = from the last forward branch target back to the backward branch
    back = list(re.finditer(r"([0-9a-f]+):.*\bs_cbranch\S+\s+(\.?L[^\s]*)", body))
    loop = body if not back else body[max(0, body.rfind("s_cbranch")) - 2000:]
    valu = len(re.findall(r"\bv_[a-z0-9_]+", loop))
    dots = len(re.findall(r"\bv_dot2_f32_f16", loop))
    mem = len(re.findall(r"\b(ds_[a-z0-9_]+|global_load\S*|buffer_load\S*)", loop))
    weights = max(dots * 2, 8)
    print("%-46s valu_in_loop~%-5d dot2=%-3d mem=%-4d weights~%-4d op/weight<=%.2f" % (
        name[:46], valu, dots, mem, weights, valu / weights))
```
Run: `./p0_sync.sh p1_isa_count.py && ssh x@192.168.10.11 "docker exec -i -w /work/p1 vllm-gfx906 python3 p1_isa_count.py" | tee logs/p1/p1_isa_ops.log`
Expected: 每个 `(UNROLL, VARIANT)` 实例一行 `op/weight<=X`。
符号名不含模板参数 ⇒ 重编时加 `-g` 并用 `llvm-dwarfdump` 建立映射。
**切不出热循环就明说切不出来**，报告里 op/weight 标"未测得"，**禁止**用 µs 反推一个看着自洽的数。

- [x] **Step 7: Commit（用户要求时才做）**

```bash
git add p1_kernels.cu p1_l1_bench.py p1_isa_count.py logs/p1/
git commit -m "p1: L1 microbenchmark of three wave64 W4A16 GEMV variants, graph-timed"
```

---

## Task 5: 判定与报告（G5 = P2 的 go/kill）

**Files:**
- Create: `docs/superpowers/plans/2026-09-08-p1-report.md`
- Modify: `docs/superpowers/specs/2026-09-03-gfx906-w4a16-wave64-kernel-design.md`（只追加一节）

- [x] **Step 1: 判据（写死，不许事后调整）**

- **kill**：最优变体在**所有**形状上 `vs_incumbent ≤ 1.1`（与 §风险表同一道闸）。
- **go**：≥2 个形状族 `vs_incumbent > 1.15`，且其 `op/weight ≤ 4.0`。
- **有条件 go**：只有 `gate_up`（N=17408，唯一不需 Split-K 的族）赢 ⇒ P2 缩窄为只换这一族，必须算出
  收益上限 `64 calls/token × (incumbent − best) µs` 并换算成 tok/s 百分比再决定值不值得。

- [x] **Step 2: 写报告（结论先行，五张表 + 一张作废清单）**

1. **判定**一句话 + 它引用的具体数字（哪个形状、哪个变体、多少倍）。
2. 每形状最优表：`K,N / incumbent graph µs / best µs / %floor / op_per_weight / variant+splitk+unroll`。
3. persistent vs 两阶段：`p1_launch.json` 的 in-graph 每 launch µs × 192，与两阶段归约的收益对比。
4. L0 与形状门：`mismatched_words=0`、L0b `max_rel`、实测形状数（5 还是 7）、`m_values` 是否含 1、
   M=1 的证据等级（直接 trace / 间接）。
5. **作废与未测清单**：未过正确性门的变体、被跳过的 (splitk, unroll) 组合、是否发生热回退、ISA 是否
   成功切出热循环。这一节不许省 —— 它是下次估算工作量的唯一依据。

- [x] **Step 3: 回填设计文档**

在规格末尾追加 `## P1 实测结论（2026-09-08）：预测 vs 实测`，逐条对照 §映射、§VALU 预算、§被否清单中
被证实/证伪的项（尤其 V2 的 LUT 假设、zp 提升是否被编译器代劳、split-K 归约的真实代价、以及 P1 新增的
"magic 偏移与 zp 提升合并成零成本"这一处）。**被证伪的条目要留在文档里**，不许改写成"当初就这么想的"。

- [x] **Step 4: 环境收尾**

Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh unlock; bash /home/x/vllm-work/p0/p0_clock.sh status; rocm-smi --showuse | grep 'GPU use'; docker exec -i vllm-gfx906 python3 -c 'import torch;print([round(torch.cuda.mem_get_info(i)[0]/1e9,1) for i in (0,1)])'"`
Expected: `level=auto`、util 0%、两卡 ≥17 GB 空闲。P1 无进镜像义务 ⇒ **不触发 B→A 重建**。

- [x] **Step 5: 记忆（只写会再踩的）**

候选：L0 的 one-hot 直读法（把布局问题从误差预算里拎出来，零容差）；magic 偏移能与被提升的 zp 合并成
`(1024+zp)·S_g` 从而去偏移零成本；op/weight 必须数 ISA；`v_dot2_f32_f16` 在此工具链要靠内联汇编；
以及 P1 的判定本身（若 kill，写清证伪了哪条预测）。

- [x] **Step 6: Commit（用户要求时才做）**

```bash
git add docs/superpowers/plans/2026-09-08-p1-report.md docs/superpowers/specs/ logs/p1/
git commit -m "p1: L1 verdict and the go/kill call for #20 P2"
```

---

## 门（任一不过即停）

| 门 | 判据 |
|---|---|
| G0 | JIT 模块能编译并在 gfx906 上启动出正确结果（`JIT-OK`） |
| G1 | L0a `mismatched_words=0` 且 5 个形状 L0b 全过（`L0 ALL PASS`） |
| G2 | 实测形状集合与设计表互校无 extra/missing（`SHAPE-GATE PASS`），且 M=1 有直接或明确标注的间接证据 |
| G3 | 实测纯读带宽 ≥ 0.9 × 821 GB/s（`FLOOR-OK`）；读+写口径一并记录，两者都进报告 |
| G4 | 参考实现自证 round-trip OK；每个进入计时的变体先过正确性门；扫参未触发热回退（或已按 4 档重测） |
| G4b | `v_dot2_f32_f16` 在此工具链可用（Step 1a 汇编门）。**不可用 ⇒ 直接判 kill**，不得改用标量 fp16 FMA 凑数（MI50 上半速，clpeak 6523 vs 24971 GFLOPS）—— 那等于把被测假设换掉 |
| G5 | 报告含五张表 + 作废清单，且 go/kill 判定引用的是 Step 1 的写死判据而非事后解释 |

---

## 自查（作者，2026-09-08）

**1. 规格覆盖**：§3 L0 → Task 1；§3 L1 两条前置（读 `quantization_config.ignore`、以运行时实测为权威）
→ Task 2；§VALU 预算的 op/weight 主 KPI → Task 4 Step 6 + Task 5 判据；`:110` 的"先量 graph 内 reduce
launch 代价再决定 persistent"→ Task 4 Step 3/4 的 `launch_cost()` + Task 5 表 3；§锁时钟 → Task 3 Step 2、
Task 4 Step 5，并把设计原文只钉 sclk 的写法纠正为 `lock 8 2`。**P2 的内核本体、L2/L3 验收、进镜像**
明确不在本计划范围内（这是 P1 的边界）。

**2. 占位符**：三处必须在执行时补的地方，都写清了补法与判据，不是 TBD：
(a) Task 4 Step 1 的 `dot2()` 是显式占位，Step 1a 给出内联汇编替换式并修掉 V3 的半成品分支；
(b) Task 4 Step 5 的现路径对照在 Step 3 已给完整函数（`exllama_baseline`），但 `zr` 直接喂给
`gptq_gemm_out` 是否被它接受为"已按 N 打包的 int32"需按 `_custom_ops.py:598` 的真实签名核对；
(c) Task 4 Step 6 的 ISA 热循环切分给了具体规则与失败退路，并明文禁止用 µs 反推。
写计划时我确实不知道 V2 会不会 bank conflict、也不知道编译器会不会自己把 zp 提升掉 —— 知道就不叫 L1。

**3. 类型/命名一致性**：`p1_gemv(x, qw, sc, zr, scratch, variant, splitk, unroll)` 的参数序在 C++ 声明、
`check()`、`main()` 三处一致；返回 `[N]` fp16 而 `reference()` 返回 `[N]` fp32，比较处已 `.float()`；
`p1_floors.json` 的键为 `"K,N"` 字符串，Task 4 用 `key.split(",")` 解析，与 Task 3 写入格式一致；
`KBIT`（Task 4）与 `dst_bits`（Task 1）是同一个 `(i//2)*4+(i%2)*16` 规则的两次独立表达，故意不共用代码。

**4. 已知薄弱处（提前声明）**：
- `unpack_zp()` 假设 qzeros 沿 N 每 8 个打包一 word（桥接文档 `:20-21`）。若真实 ckpt 另有置换，参考实现
  会跟着错 ⇒ Task 1 的 L0b 是独立交叉验证，两边必须同时过。
- `main()` 对每个组合都跑全量 `check()`，fp32 Python 参考在 `K=17408,N=5120` 上是分钟级 ⇒ 慢是故意的；
  若不可接受，可以只在每形状首末做全量、中间用窄形状（`N=256`）验证变体逻辑，但**不许**整段跳过。
- `exllama_baseline()` 喂的是随机 int32 位模式（非真实权重）：内核只按位解码，数值合法性不影响计时，
  但若产生 NaN/Inf 导致内部分叉需记录 —— 那也正是"现路径在随机位模式下的耗时"这一口径的固有代价。

## 执行记录

### Task 0（JIT 编译通道门）— G0 PASS

同步方式：本计划所有 p1 产物放宿主 `/home/x/vllm-work/p1`（经现有绑定挂载即容器内 `/work/p1`，已
`mkdir -p` 并验证），同步用 `P0_HOST_DIR=/home/x/vllm-work/p1 P0_CONTAINER_DIR=/work/p1 ./p0_sync.sh <f>`。

**计划自带缺陷 #1（已修）**：`load_inline` 缺 `functions=["p1_double"]` ⇒ 不生成 `PyInit_p1_gate`，
报错 `ImportError: dynamic module does not define module export function`。**编译其实已经成功**
（`Successfully preprocessed all matching files`），错只在绑定 ⇒ 这类"编译过了、载入失败"要看清是哪一段。

实测（`/tmp/p1_gate_build` 清空后重编，取真实首次编译成本）：

```
torch 2.13.0+rocm7.2 hip 7.2.53211 device AMD Radeon Pro VII
compile_seconds=31.5          # 单个平凡核就 31.5 s ⇒ Task 4 的"一个模块、变体运行期选择"是硬要求
KERNEL_OK variants=0,1,2
warp_size=64                  # wave64 前提成立
tool hipcc        /usr/bin/hipcc
tool llvm-objdump MISSING
tool objdump      /usr/bin/objdump
vram_free_GB=16.96
```

⇒ **Task 4 Step 6 的路径要改**：裸 `llvm-objdump` 不在 PATH，AMD LLVM 22 的工具在
`/opt/rocm/llvm/bin/`（`llvm-objdump`、`llvm-dwarfdump`、`llvm-readelf` 都在）。用绝对路径
`/opt/rocm/llvm/bin/llvm-objdump -d`；`-g` + `llvm-dwarfdump` 建立 `(UNROLL,VARIANT)` 映射时同样走这个目录。
`/usr/bin/objdump` 是 binutils，对 AMDGCN 指令解码不可信，**不要**用它替代。

### Task 1 预备（写 harness 前先读桥接，两处会咬人的地方）

1. `weight_zero_point` 必须以**检查点原格式**喂进去：`[N//8, K//G]` int32、沿 **N** 打包、
   word `(n8, g)` 的第 j 个 nibble 是列 `8*n8 + j`（`wna16_exllama.py:212-223`）。桥接自己会
   unpack→置乱→`_pack_gptq_sequential(z, 4, pack_dim=1)` 成 `[K//G, N//8]`。
   所以 Task 1 里打包 zp 的代码是**独立实现**（不许复用桥接的 pack 函数，否则 L0 变成自我循环论证）。
2. `_transform_param` 内部走 `permute_param_layout_`，它对 `BasevLLMParameter` 有形状/属性假设，
   可能拒绝裸 `torch.nn.Parameter`。若报 `AttributeError`/断言 ⇒ 把合成参数包成
   `vllm.model_executor.parameter.BasevLLMParameter(data)`，**不要**为了跑通去自己实现 transform
   （那等于绕开了被测对象）。

---

### Task 1（L0 位映射钉死）— G1 PASS

结果（`logs/p1/p1_l0_mapping.log`，宿主与仓库各一份）：
- **L0a**：小形状 + 5 个真实形状，每次 `mismatched_words=0`，累计 **22,602,240 个 int32 字**逐字比对
  —— 桥接 repack 与文档位规则完全一致，零容差。
- **L0b**：one-hot `x = e_k` 直读，5 个真实形状全部 **`max_rel = 4.880e-04`** —— 恰为 1/2048，
  即一个 fp16 输出量子（内核输出 fp16、参考 fp32 的舍入下限），**无任何映射类误差**。
- 位级映射实现文档已落：`docs/superpowers/plans/2026-09-08-p1-nibble-map.md`（含检查点侧形态、
  bit 表、以及三个 harness 坑）。P2 可以直接按它实现提取。

**计划自带缺陷 #2（执行中发现并修掉）**：harness 的参数类不能是裸 `torch.nn.Parameter`
（`permute_param_layout_` 读 `input_dim/output_dim`，两者皆无 ⇒ "either input or output dim must be
set"），也不能是裸 `BasevLLMParameter`（其 `__init__` 调 `get_tensor_model_parallel_rank()`，
harness 里没有 vLLM 分布式状态 ⇒ "tensor model parallel group is not initialized"）。
**正解 = 继承 `torch.nn.Parameter` 直接挂这三个属性**（`weight_loader` 只在加载期用，桥接路径不碰）。
两次失败都留着原话，供下次造合成层的人少走两轮。

### Task 2（真实形状与真实 M 探针）— G2 PASS（两次失败两次修）

1. **失败一（已修）**：`python3 - <<PY` 的 stdin 脚本当主模块 ⇒ vLLM spawn worker 要重新 import
   主模块，而主模块叫 `<stdin>` ⇒ 两个 worker 全灭于 `FileNotFoundError '/<stdin>'`。
   修复 = payload 落成真文件 `p1_shape_probe_main.py` + `__main__` 守卫（无守卫 spawn 会递归）。
   两份 7 字节 CSV（只有表头）证明 SHAPEDBG 的 atexit 通道本身是通的。
2. **失败二（已修）**：`limit_mm_per_prompt={"image":1,"video":1}` 且未设
   `--mm-encoder-attn-backend` ⇒ `profile_run` 的 dummy ViT 前向回退到物化注意力的路径，
   `torch.OutOfMemoryError: Tried to allocate 128.00 GiB`，两个 worker 同时死。
   修复 = 形状普查改**纯文本**（`{"image":0,"video":0}`）—— 5 个 WNA16 形状全在 LLM 栈，
   视觉塔不量化，普查不需要 mm。交付路径靠 TRITON_ATTN 避开的是同一个坑。
3. **通过**（`logs/p1/p1_shapes_report.log` + 两 rank 原始 CSV）：两 rank 各 **16,896 行**、
   形状集合与设计 §2 **逐行一致（extra=∅, missing=∅）**，`m_values = [1, 12, 256, 8192]`
   —— **1（decode）是直接观测**（12/256/8192 是 prefill/chunk 档）；per-rank N 与全局 N/2 的
   `attn_output_gate` 事实（7168 而非 6144）与设计一致。
4. **M=1 的 graph 侧交叉证据（Step 4）**：P0.5 桶表（`logs/p0/p05_buckets_ab_on.log`）
   `w4a16_gemv launches=16384` ÷ 64 步 = **256 次/decode 步**，与 batch1 每步 256 次 M=1 GEMV
   吻合 ⇒ graph 内 M 证据等级：间接但量化吻合（eager 侧直接观测 + graph 侧 launch 计数）。
5. 设计 `:100`（5 形状）vs `:181`（"7 个 per-rank 组合"）的内部矛盾：**实测=5**，Task 5 回填时改设计。

### Task 3（地板与真实字节分母）— G3 PASS

锁 `lock 8 2`（1700/1000）实测（`logs/p1/p1_floors.json`）：
- **纯读 813.8 GB/s**（torch.sum 整缓冲；与设计保守地板 821 一致，低于手写纯读的 928）⇒ 断言过，
  地板按它算，**偏保守（地板偏高）**。
- **读+写 1300.4 GB/s —— 口径有诈，不得当带宽带宽引用**：`add(x, x, out=o)` 两个操作数是同一块
  buffer，第二遍读命中 L2，账面 3× 流量只付了 ~2× 的真 DRAM。要修就得用两块不同 buffer。
  纯读仍是唯一有效分母。
- 每形状地板（含 3.9% 元数据、元数据**不**随 splitK 放大）：(5120,7168) 23.43 µs、(3072,5120) 10.04、
  (5120,17408) 56.90、(8704,5120) 28.45、(5120,8192) 26.78 ⇒ **SUM = 7764.8 µs/token**。
- 对今天 1425 MHz 口径的 19,867 µs/step 桶：**头寸 12.1 ms/step**（两口径时钟不同，只作量级对照；
  L1 的百分比全部用本 JSON 的 per-shape floor，不用这个总和）。

### Task 4（L1 微基准主体）— G4 PASS（三轮仪器化调试）

本轮的教训形态高度一致：**每个 bug 都是被仪器抓到的，没有一个靠读代码**。
1. HIP 构建修复：`at::Half*`/`__half*` 不互转（改 `data_ptr()` 转换）、
   `__HIP_NO_HALF_CONVERSIONS__` 下必须 `__half2float()`。**G4b 成立**：`v_dot2_f32_f16`
   内联汇编被汇编器接受且结果正确。
2. one-hot 诊断抓到 **V2 整个丢了 zp 减法**（等效码表 = code+zp）与 **V1 的抵消噪声**
   （随机 x 3.2e-3 ⇒ per-word 去偏后 9.1e-4；one-hot 完全看不见 ⇒ 噪声与接线必须分开测）。
3. 毒值探针（scratch 填 -7777）连续抓到两个 launch bug：1-D grid ⇒ `blockIdx.y` 恒 0、
   chunk≥1 从不执行；`dim3(splitk)` 的 chunk 数在 `.x`，第一次"修复"传 `.y` 仍是空转。
4. 图计时全为 1.03 µs ⇒ 默认流上的 launch **不进 torch.cuda.graph 的捕获图**；
   改 `at::cuda::getCurrentCUDAStream()` 后 smoke 410 µs（> 地板 56.9）⇒ 修复确认。
5. 首轮扫参 171 行但图计时全废；重跑 171 行有效。随机 x 的 5e-4 门在真实形状低于
   1 fp16 ULP 的顺序噪声（精确的 V3 也测 6.3e-4）⇒ 门重构为「one-hot 接线门（严格）
   + 随机 x 数值记录（>5%/NaN 才坏）」。另修 `time` 未导入、temps 解析、clock glob。
6. op/weight/VGPR：两次设备码提取失败（`.so` bundler 列表空、`.cuda.o` 反汇编 1 行）
   ⇒ **记"未测得"**，禁用 µs 反推。

### Task 5（判定与报告）— G5 完成，判据未干净命中 ⇒ 决策交用户

- 报告：`docs/superpowers/plans/2026-09-09-p1-report.md`（结论先行 + 判定表 + 作废清单）。
  判定：**非 kill、非 go、非 gate_up-only** —— (5120,8192) ×1.19 与 (5120,7168) ×1.14 两族
  （均 V3）赢，加权全换 ×0.83，按形状分派 +3.6% decode（752 µs/token ≈ +0.9 tok/s）。
- 设计文档已回填「P1 实测结论」表：证伪 7e-7 噪声声明、"Split-K 重复读元数据"模型、
  LUT 可行性；证实位规则与 5 形状集；新增设计自由度（无偏提取的代价表）。
- 环境收尾：时钟 auto、两卡空闲。op/weight/VGPR 未测得已如实标注。

## 外部评审处置（2026-09-08，评审文件见 WorkBuddy 目录 `z100-...-p1-microbench-plan-review.md`）

**采纳前先核实**：P0-2（喂 `[N,K/G]` ⇒ 桥接抛 `shape '[16, 2]' is invalid for input of size 256`）、
P0-5（缺 `permute(0,2,1)` ⇒ row k=1 解成 `[5,11,1,9,13,3]`，正解 `[8,9,10,11,12,13]`）**都在容器里跑出来
了**；P0-4 直接读到 `wna16_exllama.py:70-85` 的 SIGUSR1 通道与"atexit 不够用"原注释；P0-1 的槽位算术
（lane 0 串行 128×3 ≈ 384 > 主体 304）与 P0-6 的 `out=` 形状不符一抛即验。
**这份评审最大的价值不是找到 bug 数量，而是指出 L1 会因此产出一个"假 kill"** —— 测出来的会是"lane 0
串行求和很慢"，而不是"wave64 W4A16 手写不划算"。这类"实验测了自己的脚手架"的错，事后看谁都懂，写的时候不觉得。

| 项 | 处置 |
|---|---|
| P0-1 `S_g` 串行 | **已改进计划代码**：wave butterfly（2 元/lane + 6 步 `__shfl_xor` + 一次广播） |
| P0-2 zp 喂错形状 | **已改**：新增独立 `pack_zp_checkpoint_format`，并注明不许复用桥接 packer |
| P0-3 L0b 给内核喂了解包后的 zp | **已改**：内核喂 packed `z_packed`，只有 `want` 用解包值；顺带加形状断言 |
| P0-4 不发 SIGUSR1 | **已改**：探针内 `pkill -USR1 -f 'VLLM::[W]or'`（带括号防自匹配）+ `VLLM_ENABLE_V1_MULTIPROCESSING=0` + 落盘文件大小回显 |
| P0-5 `unpack()` 缺 permute | **已改**，并写明 `unpack_zp()` 不要一起"修" |
| P0-6 `calibrate_bw_eff` 不是测带宽 | **已改**：纯读（whole-buffer `sum`）与读+写（`add(out=o)`）两个口径，assert 挂在纯读上 |
| P1-1 V3 形式不可实现 | **已改，但与评审写法不同**（见下"分歧"） |
| P1-2 `*(unsigned*)&__half` 越界 + V2 真实 24 条/字 | **已改**：`__half_as_ushort`；变体表里那句"预测是平的"一并改成"预测输"，并注明我第一版就这条说错过一次 |
| P1-3 单次 launch 计时 / 两个独立 noop | **已改**：`timeit(R=24)` 取均、incumbent 同尺；`launch_cost` 改为"同形状 splitk=1 带/不带 reduce"，`p1_gemv` 加 `reduce` 开关 |
| P1-4 元数据不该 ×splitK | **已改**：`bytes_pulled` 只算一份，JSON 的 `pulled_bytes` 与 `floor_us` 同口径。设计 §其他 那句"重复读 splitK 次"是错的，Task 5 Step 3 回填时一并纠 |
| P1-5 阈值 2e-2 太松 | **已改**：先 `want.half()` 再比，阈值 5e-4 |
| P1-6 `__main__` 早于 `main()` | **已改**：守卫移到文件末（Step 4 末），Step 2 明确"先不写守卫" |
| P1-7 单一 BW 参考值 | **已改**：821 保守地板 / 928 纯读上限两个都带进 JSON 与报告 |
| P1-8 温度只读 card0 | **已改**：`temps_c()` 走 `rocm-smi --showtemp` 双卡；`temp1_input` 是否结温**尚未核实**，Step 5 首次运行并列打印一次人工对齐 |
| P1-9 计划代码块缺 `functions=` | 脚本已修；**计划 Task 0 的代码块待同步**（执行 Task 0 时以仓库里的 `p1_jit_gate.py` 为准） |
| P1-10 Step 1 的 `.cu` 不是可编译文件 | **部分已改**：`dot2` 的直接给成真实内联汇编，Step 1a 只剩"汇编器是否接受该拼写"这一件事 |
| P1-11 `temp_dq` 白占 178 MB | **已改**：`torch.empty(0)`，与生产路径一致 |
| 漏1 dot2 在 gfx906 的 issue 速率未验 | **待办，钉在 Task 4 Step 1a**：加一个只跑 dot2 的微核，与 clpeak half2 24971 GFLOPS 对表；op/weight 的换算全部依赖这个速率 |
| 漏2 外部 gfx906 wave64 参考实现（`llama.cpp-gfx906` 的 wave primitives / `V_DOT2_F32_F16` 实战用法） | **未核实**（离线无法确认其存在与许可）。执行 Task 4 前若网络可用则查一次；即便有，P1 的目的就是自己量，不以外代码替代测量 |
| 漏3 ISA 与计时交叉校验 | **待办，钉在 Task 4 Step 6**：`op/weight = 6.62e12·t/(K·N/8)` 与 ISA 数出来的并列，差 >20% 即判"热循环切错了" |
| 漏4 按 calls/token 加权 | **已改**：`main()` 末尾输出 incumbent/best/floor 三个加权总量与聚合倍数，并落 `p1_l1_summary.json` |
| 漏5 `p1_gemv` 每次分配输出 | **已改**：注释写明 P2 不得照抄（生产内核写调用方持有的 capture-safe slab） |
| 漏6 unroll × VGPR × occupancy 未量 | **待办，钉在 Task 4 Step 6**：dump `.note`/`.amdhsa_kernel` 里的 VGPR 数（真实形状 `waves/CU` 只有 1.33–4.53，本就 occupancy 不足） |
| 漏7 小形状被 L2 兜住 | **已改**：每形状两份权重缓冲轮换 |
| 漏8 `ignore_eos` | **已改**：Task 2 探针加上 |
| 漏9 扫参期间无时钟回显 | **已改**：每形状打印 `clock_status()`（sysfs 活跃档）+ 双卡温度 |
| 漏10 dot2 不可用时的判据 | **待办，钉在门表**：见下 |
| 漏11 未校验 per-rank N 是否 = 全局 N/2 | **待办，钉在 Task 2 Step 3**：CSV 里 per-rank N 与全局 N/2 对照（`attn_output_gate=true` ⇒ q_proj 12288，TP2 后 6144 而非 7168 —— 这一条设计里已对，值得在报告显式写出） |

**与评审的一处分歧（写下来供裁决）**：评审的 V3 是 `v_pk_sub_f16(magic, zp2_biased)` + 尾项
`raw − 1024·S_g`；我写成"减带偏 zp `(1024+zp)`"，环内直接得到 `c−zp`、**尾项什么都不减**。两者数值等价，
但代价是 **V3 从此不需要 `S_g`** ⇒ V1-vs-V3 的差变成"zp 两种处理路径"的整体差，而不是设计预算表里那孤立的
4 条/字。报告必须按"整体差"描述，**不许**写成"实测复现了设计的 4.25 vs 3.85"。

**门表补一条（漏10）**：`dot2` 若在此工具链下不可用 ⇒ **直接判 kill**，不得用标量 fp16 FMA（MI50 上半速，
clpeak 6523 vs 24971 GFLOPS）凑数后再拿 op/weight 说事 —— 那把被测假设换掉了。
