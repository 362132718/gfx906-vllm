# P0.5「把 M=1 稠密 fp16 GEMM 从 Tensile 串行 K 循环里换出来」实施计划

> **状态（2026-09-08）：Task 0–7 全部执行完毕，G1–G8 全过。** step 53.663 → 33.008 ms（18.6 → 30.3 tok/s），256/256 贪心 token 一致，已进交付镜像。结论与偏差见 [`2026-09-08-gfx906-p05-report.md`](2026-09-08-gfx906-p05-report.md)，逐步证据见文末「执行记录」。下方逐步 checkbox 保留原样，以执行记录为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 gfx906 在 M=1 时用 vLLM 自带的 ROCm skinny GEMV（`ops.LLMM1`，设备内核 `LLGemm1_kernel`）回答两个稠密投影，把 decode step 从 53.68 ms 降到 ~33 ms（18.6 → ~30 tok/s），**不写任何新内核、不重编 C++**。

**Architecture:** 唯一改动点是 `vllm/model_executor/layers/utils.py:rocm_unquantized_gemm_impl` —— 它是 `UnquantizedLinearMethod.apply` 在 ROCm 上的落点（`dispatch_unquantized_gemm()` → `rocm_unquantized_gemm`），GDN 的 `in_proj_ba`（`MergedColumnParallelLinear`，`bias=False`，TP 分片后 per-rank m=48）与 `lm_head`（`ParallelLMHead`，per-rank m=124160）都走这里。该函数已被 `direct_register_custom_op` 注册成 `torch.ops.vllm.rocm_unquantized_gemm`（带 `fake_impl`），对 dynamo 与 CUDA graph 捕获天然透明，所以新增分支不触碰编译区。`_ON_GFX9` 对 gfx906 的排除（`platforms/rocm.py:236-239`）**保持不变** —— 只在稠密 dispatch 里显式开一个已实测的口子。

**Tech Stack:** vLLM 0.28.0（容器内 site-packages，非 editable）、ROCm 7.2.4、`torch.ops._rocm_C.LLMM1`（随 `PYTORCH_ROCM_ARCH=gfx906` 编进 `_rocm_C`，已实测存在且可跑，源 `csrc/rocm/skinny_gemms.cu:175`）、既有 trace 聚合链路（`p0_parse_trace.py` → `p0_buckets.py`）、交付配方 `docker/Dockerfile.rocm_gfx906` 的两条构建路径。

---

## 证据基线（写计划前已实测；执行时不必重复，但 Task 1 会正式重跑并落盘）

来源：`docs/superpowers/plans/2026-09-04-gfx906-w4a16-p0-report.md`、`logs/p0/`，以及本轮 3 个临时探针（cold card、auto 时钟、单卡）。

| 事实 | 数值 | 出处 |
|---|---|---|
| 基线 decode step（TP2、batch=1、sclk 1425/mclk 1000） | **53.678 ms = 18.6 tok/s** | `/work/p0/p0_wall.json` |
| `dense_fp16` 桶合计 | **22 242 µs/step = 41.4%** | `logs/p0/p0_buckets_final.log` |
| ├ GDN `in_proj_ba`（per-rank m=48, k=5120） | 15 260 µs/step，48 launches/step，`Cijk…MT32x32x16` | 同上 |
| └ `lm_head`（per-rank m=124160, k=5120） | 6 982 µs/step，1 launch/step，`Cijk…MT64x16x16` | 同上 |
| Tensile 对 M=1 是**串行 K 循环**，不是带宽瓶颈 | 318 µs 对 N=32..8192 **全平**，mclk 350 与 1000 **相同** | `logs/p0/p0_dense_proper.log` |
| `BW_eff`（本卡地板） | **931 GB/s**（保守值 821） | P0 Task 1 |
| `LLGemm1_kernel` 在 gfx906 上**可用且正确** | m=48 → **3.9 µs**（Tensile 258.6 µs，**61×**）；m=124160 → **1 449 µs = 877 GB/s = 94% of BW_eff** | 本轮探针 |
| 确定性 | 20 次重复调用**逐位相同**（该内核无 atomicAdd，不像 exllama 融合路径） | 本轮探针 |
| `rows_per_block` 1/2/4/8 | lm_head 1458/1525/1449/1475 µs；ba 4.1/3.5/3.9/5.0 µs ⇒ 差异 <15%，**沿用上游的 4**，不加新旋钮 | 本轮探针 |
| 两个调用点都无 bias | `create_ba_proj(... bias=False)`（`qwen_gdn_linear_attn.py:580`）；`ParallelLMHead` 无 bias ⇒ `bias is None` 是护栏不是限制 | 容器源码 |
| ba_proj 确实被 TP 分片 | `maybe_disable_tp` 只对 AutoAWQ/AutoGPTQ/INC 返回 True，我们是 compressed-tensors ⇒ False ⇒ per-rank m = 96/2 = 48，与 trace 的 48 launches/step 吻合 | `qwen_gdn_linear_attn.py:586-605` |
| 随机操作数下 vs fp64 | LLMM1 rel 4.1e-4 / 3.9e-4；Tensile 1.97e-4 / 3.25e-4（**同量级，LLMM1 略差 ~2×**） | 本轮探针 |
| **陷阱：`LLMM1` 不校验权重连续性** | 传 `randn(k,m).t()` 视图它**照跑并返回错的数**，不抛异常 ⇒ 门里必须有 `weight.is_contiguous()` | 本轮探针 |

预测收益：`53.678 − (15.26 − 0.19) − (6.98 − 1.45) = 33.1 ms` ⇒ **30.2 tok/s，+62%**。

---

## 文件结构

| 路径 | 责任 | 动作 |
|---|---|---|
| `docker/patches/layers_utils.py` | 打补丁后 `vllm/model_executor/layers/utils.py` 的**唯一权威副本**（放在消费它的配方旁边，避免与 `container_src/` 双份漂移） | Create |
| `p05_install.sh` | 把仓库文件装进开发容器的绝对路径并三方 md5 校验（补 `p0_sync.sh` 只管 `/work/p0` 脚本的空白） | Create |
| `p05_skinny_probe.py` | Task 1 裸测：形状/确定性/连续性陷阱/bf16/rpb，落 `p05_skinny.json` | Create |
| `p05_accuracy.py` | Task 2 真实权重数值门（vs fp64） | Create |
| `p05_gate_test.py` | Task 3 门谓词单元测试（spy `ops.LLMM1`），先红后绿 | Create |
| `p05_ab.sh` | Task 4/5 一次分离作业跑 off/on 两臂 | Create |
| `p0_step.py` | 输出目录/锚点参数化 + 保存生成文本与 token（否则 +42% 会被自己的 perf gate 下限判失败，且「输出没变」没有证据） | Modify `:34` `:42` `:122-125` |
| `p0_buckets.py` | 新增 `dense_gemv` 桶；`P0` 改读 `OUT` | Modify `:12` `:22-31` `:31` |
| `docker/prepare_build_context.sh` | 第 3 步多收集一个文件（Path B 需要） | Modify `:42` 附近 |
| `docker/Dockerfile.rocm_gfx906` | **两条路径**都要 COPY + 断言（Path A 的 `final` 阶段原本只重断言 entrypoint，不 COPY python 补丁） | Modify `:94` `:115` `:159` `:169` 附近 |
| `docker/entrypoint/selfcheck.sh` | 运行期断言补丁在场 | Modify `:40` 附近 |

约定（长期约束）：**容器内 GPU 作业一律 `docker exec -d` + 日志**（主机静默掉电无内核日志）；`p0_clock.sh lock <sclk> <mclk>` **两把时钟都要锁**（只锁 sclk 会让 mclk 停在 350 MHz，带宽数悄悄 3-4× 慢）；语法检查必须在容器里做（本地 `python` 是 Windows Store 占位符，退出码 49）。

---

## Task 0: 前置门与 harness 参数化

**Files:**
- Modify: `p0_step.py:34`、`:42`、`:122-125`
- Create: `p05_install.sh`

- [x] **Step 1: 确认仓库还没有提交，并向用户要 git 身份**

Run: `git -C D:/gfx906-vllm log --oneline -1`
Expected: `fatal: your current branch 'p0-attribution' does not have any commits yet`

**不要**运行 `git config`（长期约束：永不修改 git config）。在拿到用户给的 `user.name`/`user.email` 之前，各任务的「Commit」步骤降级为 `git add <文件>` 暂存，并在本计划末尾「执行记录」里标注「待提交」。这不阻塞任何测量。

- [x] **Step 2: `p0_step.py` 输出目录可参数化（全链路统一用一个 env 名 `OUT`）**

`p0_step.py:34` 原文 `OUT = "/work/p0"` 改为：

```python
# One env name for the whole chain (p0_parse_trace.py already reads OUT, and
# p0_buckets.py is being switched to it): an A/B run needs two archives that do
# not overwrite each other's p0_wall.json / prof/.
OUT = os.environ.get("OUT", "/work/p0")
```

- [x] **Step 3: perf gate 锚点可参数化**

`p0_step.py:42` 原文 `EXPECTED_MS = 47.0 * 1700.0 / CLOCK_MHZ` 改为：

```python
# P0.5 legitimately takes the step from 53.7 ms to ~33 ms, which is BELOW this
# gate's 0.6x floor -- the floor exists only to catch the ~5x Triton W4A16
# fallback, so an A/B run passes its own post-change anchor in here instead of
# weakening the band for everyone.
EXPECTED_MS = float(os.environ.get("EXPECTED_MS", 47.0 * 1700.0 / CLOCK_MHZ))
```

- [x] **Step 4: 让 wall 记录携带生成内容（Task 4 Step 5 的证据来源）**

`p0_step.py:122-125` 的 `json.dump({...}, open(f"{OUT}/p0_wall.json", "w"))` 整块改为：

```python
    json.dump({"prefill_ms": prefill * 1e3, "req_ms": req * 1e3,
               "per_step_ms": per_step * 1e3, "n_tok": N,
               "clock_mhz": CLOCK_MHZ, "capture_sizes": CAPTURE_SIZES},
              open(f"{OUT}/p0_wall.json", "w"))
    # Token ids, not just prose: swapping in a different GEMV kernel changes
    # fp16 rounding, so "the answer is unchanged" needs a greedy-decode
    # prefix-equality check rather than a vibe read of the text. (out is the
    # RequestOutput from gen(); the completion carries the ids.)
    json.dump({"token_ids": list(out.outputs[0].token_ids),
               "text": out.outputs[0].text},
              open(f"{OUT}/p0_step_out.json", "w"), ensure_ascii=False)
```

并把 `main()` 里 `req = min(gen(llm, N)[0] for _ in range(3))` 改为（多留一个 `out`）：

```python
    req_times = [gen(llm, N) for _ in range(3)]
    req, out = min(req_times, key=lambda t: t[0])
```

- [x] **Step 5: 写 `p05_install.sh`**

```bash
#!/usr/bin/env bash
# p05_install.sh -- install one repo file over an absolute path INSIDE the dev
# container and prove local == host == container by md5.
#
# p0_sync.sh only covers dev scripts under /work/p0; a patched site-packages
# file needs a different target, and "the container still has the old copy" is
# a bug class that already cost a measurement cycle. Refuses to install anything
# the delivery recipe does not COPY, so a change can't silently live only in
# the dev container.
#
# Usage: ./p05_install.sh <repo-path> <container-abs-path>
set -euo pipefail
HOST=${HOST:-x@192.168.10.11}
CONTAINER=${CONTAINER:-vllm-gfx906}
SRC=${1:?repo path}; DST=${2:?container absolute path}
[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
# The recipe spells the same target two ways: ${SP}/<rel> in COPY lines, the
# absolute path in test/grep assertions. Accept either, but refuse a file the
# delivery recipe never mentions. Fixed-string matching only -- an ERE here
# needs escaping, and sed's replacement-side & expands to the whole match.
REL=${DST#/usr/local/lib/python3.12/dist-packages/}
if ! grep -qF -e "$DST" -e "dist-packages/${REL}" -e '${SP}/'"${REL}" \
       docker/Dockerfile.rocm_gfx906; then
  echo "FATAL: $DST is not COPYed by docker/Dockerfile.rocm_gfx906 -- this edit" \
       "would not enter the delivery image. Do Task 6 first, or FORCE=1." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
fi
NAME=$(basename "$SRC")
scp -q -o BatchMode=yes "$SRC" "$HOST:/home/x/vllm-work/p0/$NAME"
ssh -o BatchMode=yes "$HOST" "docker cp /home/x/vllm-work/p0/$NAME $CONTAINER:$DST"
L=$(md5sum "$SRC" | cut -c1-32)
C=$(ssh -o BatchMode=yes "$HOST" "docker exec $CONTAINER md5sum $DST | cut -c1-32")
[ "$L" = "$C" ] || { echo "MISMATCH $SRC local=$L container=$C" >&2; exit 1; }
echo "ok  $SRC -> $CONTAINER:$DST  $L"
```

- [x] **Step 6: 语法检查**

Run: `./p0_sync.sh p0_step.py p05_install.sh && ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 -m py_compile /work/p0/p0_step.py && echo COMPILE-OK"`
Expected: 两行 `ok ...` + `COMPILE-OK`

- [x] **Step 7: Commit**

```bash
git add p0_step.py p05_install.sh
git commit -m "p05: parameterize attribution harness (OUT, EXPECTED_MS) and record greedy output"
```

---

## Task 1: 裸测确认（形状、连续性陷阱、bf16、确定性）

把写计划前的临时探针变成有落盘、有断言的正式证据。**这一步不装任何补丁。**

**Files:**
- Create: `p05_skinny_probe.py`

- [x] **Step 1: 写 `p05_skinny_probe.py`**

```python
# p05_skinny_probe.py -- does vLLM's ROCm skinny GEMV answer on gfx906, and
# under what conditions must we NOT hand it a call? No patch installed yet: we
# call torch.ops._rocm_C.LLMM1 directly.
#   ssh x@192.168.10.11 'docker exec -i -e BW_EFF=931 vllm-gfx906 python3 /work/p0/p05_skinny_probe.py'
import json, os, time, torch
from vllm import _custom_ops as ops

BW_EFF = float(os.environ["BW_EFF"])          # 931 measured / 821 conservative
REAL = [("in_proj_ba", 48, 5120), ("lm_head", 124160, 5120)]
OUT = os.environ.get("OUT", "/work/p0")


def gtime(fn, inner=10, iters=200):
    """Graph-replay time per call. Eager timing adds ~10 us of Python and would
    completely hide the ~4 us answer for in_proj_ba; capture also proves the
    kernel is graph-safe (it allocates its own output)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters / inner * 1e6


def check(name, m, k, dtype=torch.float16, contiguous=True, rpb=4):
    torch.manual_seed(0)
    w = (torch.randn(k, m, device="cuda", dtype=torch.float16).t() if not contiguous
         else torch.randn(m, k, device="cuda", dtype=dtype) / k ** 0.5)
    x = torch.randn(1, k, device="cuda", dtype=dtype)
    ref = torch.sum(w.double() * x.double(), dim=1)    # elementwise: fp64 GEMV
    ten = torch.nn.functional.linear(x, w).view(-1).double()   # incumbent Tensile
    try:
        out = ops.LLMM1(w, x, rpb)
    except Exception as e:
        return {"case": name, "error": repr(e)[:200]}
    torch.cuda.synchronize()
    o = out.view(-1).double()
    det, a = "stable", ops.LLMM1(w, x, rpb).clone()
    for _ in range(20):
        if not torch.equal(a, ops.LLMM1(w, x, rpb)):
            det = "NONDETERMINISTIC"
            break
    us = gtime(lambda: ops.LLMM1(w, x, rpb))
    return {"case": name, "m": m, "k": k, "rpb": rpb,
            "dtype": str(dtype).split(".")[-1], "contig": contiguous,
            "gemv_us": round(us, 1),
            "tensile_us": round(gtime(lambda: torch.nn.functional.linear(x, w)), 1),
            "gb_s": round(m * k * 2 / (us * 1e-6) / 1e9, 1),
            "pct_bw_eff": round(100 * m * k * 2 / (us * 1e-6) / 1e9 / BW_EFF, 1),
            "rel_vs_fp64": float(((o - ref).abs().max() / ref.abs().max()).item()),
            "rel_tensile_vs_fp64": float(((ten - ref).abs().max() / ref.abs().max()).item()),
            "determinism": det, "out_shape": list(out.shape)}


rows = [check(n, m, k) for n, m, k in REAL]
rows += [check("in_proj_ba_bf16", 48, 5120, torch.bfloat16),
         check("lm_head_bf16", 124160, 5120, torch.bfloat16),
         check("noncontig_weight", 48, 5120, contiguous=False)]
rows += [check("rpb%d_m48" % r, 48, 5120, rpb=r) for r in (1, 2, 4, 8)]
for r in rows:
    print(json.dumps(r, ensure_ascii=False), flush=True)

# --- hard assertions: this table IS the gate, not a report -------------------
by = {r["case"]: r for r in rows}
for name, m, k in REAL:
    r = by[name]
    assert "error" not in r, "%s: LLMM1 does not run on gfx906 -> this plan is void" % name
    assert r["determinism"] == "stable", "%s is nondeterministic" % name
    assert r["rel_vs_fp64"] < 3 * r["rel_tensile_vs_fp64"] + 1e-5, \
        "%s: LLMM1 rel %s not within 3x of Tensile rel %s" % (
            name, r["rel_vs_fp64"], r["rel_tensile_vs_fp64"])
assert by["in_proj_ba"]["gemv_us"] < 0.25 * by["in_proj_ba"]["tensile_us"], \
    "in_proj_ba gain under 4x -- not worth a dispatch change"
assert by["lm_head"]["pct_bw_eff"] > 70, \
    "lm_head only %.1f%% of BW_eff -- the bandwidth path is not what we think" % \
    by["lm_head"]["pct_bw_eff"]
nc = by["noncontig_weight"]
assert "error" not in nc and nc["rel_vs_fp64"] > 1e-2, \
    "expected a transposed weight view to SILENTLY return garbage (got rel %s)." \
    "If LLMM1 now rejects it, the contiguity clause is redundant -- say so in the" \
    "gate comment instead of deleting it blindly." % nc.get("rel_vs_fp64", "error")
bf16_ok = all("error" not in by[c] and by[c]["rel_vs_fp64"] < 1e-2
              for c in ("in_proj_ba_bf16", "lm_head_bf16"))
os.makedirs(OUT, exist_ok=True)
json.dump({"bw_eff_gb_s": BW_EFF, "bf16_ok": bf16_ok, "rows": rows},
          open("%s/p05_skinny.json" % OUT, "w"), indent=1)
print("bf16_ok =", bf16_ok, "| ALL PASS ->", "%s/p05_skinny.json" % OUT)
```

- [x] **Step 2: 锁时钟（两把都锁）**

Run: `./p0_sync.sh p05_skinny_probe.py`
Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh lock 4 2; cat /sys/class/drm/card0/device/pp_dpm_sclk /sys/class/drm/card0/device/pp_dpm_mclk"`
Expected: 星号分别落在 `4: 1425Mhz` 与 `2: 1000Mhz`

- [x] **Step 3: 跑**

Run: `ssh x@192.168.10.11 "docker exec -i -e BW_EFF=931 vllm-gfx906 python3 /work/p0/p05_skinny_probe.py" | tee logs/p0/p05_skinny.log`
Expected: `in_proj_ba` 行 `gemv_us≈4 / tensile_us≈259`，`lm_head` 行 `pct_bw_eff>70`，末行 `bf16_ok = <bool> | ALL PASS`。
任一 assert 失败 ⇒ **停**：计划前提不成立，把实测写进「执行记录」，回到 #20 的 P1。

- [x] **Step 4: 记下 bf16 结论**

`bf16_ok = True` ⇒ Task 3 Step 5 用「带 bf16 的那一版」dtype 条件；`False` ⇒ 用默认（仅 float16）。这是本计划里唯一由测量结果决定的一处分支，两版代码都写在 Task 3 里。

- [x] **Step 5: Commit**

```bash
git add p05_skinny_probe.py logs/p0/p05_skinny.log
git commit -m "p05: measure LLMM1 skinny GEMV on gfx906 (shapes, determinism, contiguity trap)"
```

---

## Task 2: 真实权重数值门

随机操作数不足以证明交付安全：真 `lm_head` 的行间 logit 分布、以及 a/b 门值误差要穿过 48 层 GDN。用 checkpoint 真权重 + fp64 参考。

**Files:**
- Create: `p05_accuracy.py`

- [x] **Step 1: 写 `p05_accuracy.py`**

```python
# p05_accuracy.py -- LLMM1 vs Tensile vs an fp64 reference on the REAL weights.
#   ssh x@192.168.10.11 'docker exec -i vllm-gfx906 python3 /work/p0/p05_accuracy.py'
# Per-row dot products do not care about row order, so the separate
# in_proj_a checkpoint tensor is a faithful operand for the fused in_proj_ba.
import glob, json, os, struct
import torch
from safetensors import safe_open            # vllm's own dependency
from vllm import _custom_ops as ops

MODEL = os.environ.get("MODEL",
                       "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
N_ROWS_LM_HEAD = 8192      # the full 124160-row fp64 reference needs ~5 GB
N_X = 64
OUT = os.environ.get("OUT", "/work/p0")
torch.manual_seed(1234)


def locate(suffix):
    """(shard_path, key) for the first tensor whose name ends with `suffix`.
    Reads only the 8-byte + JSON header of each shard -- no 2.5 GB load."""
    for shard in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
        with open(shard, "rb") as f:
            hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for k in hdr:
            if k.endswith(suffix):
                return shard, k
    raise SystemExit("FATAL: no tensor ending in %r under %s" % (suffix, MODEL))


def load_rows(shard, key, rows=None):
    with safe_open(shard, framework="pt") as f:
        t = f.get_tensor(key)
    return (t[:rows] if rows else t).contiguous()


def grade(tag, w_cpu, xs):
    w = w_cpu.to(torch.float16).cuda().contiguous()
    k = w.shape[1]
    rel_g, rel_t, flips, margins = [], [], 0, []
    for x in xs:
        x = (x * 0.55).cuda().half().view(1, k)      # a decode hidden state is
        ref = torch.sum(w.double() * x.t().double(), dim=1)      # not unit-norm
        g = ops.LLMM1(w, x, 4).view(-1).double()
        t = torch.nn.functional.linear(x, w).view(-1).double()
        rel_g.append(((g - ref).abs().max() / ref.abs().max()).item())
        rel_t.append(((t - ref).abs().max() / ref.abs().max()).item())
        if w.shape[0] > 8:
            flips += int(g.argmax().item() != t.argmax().item())
            s = g.sort(descending=True).values
            margins.append(float((s[0] - s[1]).abs().item() / s[0].abs().item()))
    r = {"case": tag, "m": w.shape[0], "k": k,
         "rel_gemv_max": round(max(rel_g), 6),
         "rel_tensile_max": round(max(rel_t), 6),
         "ratio": round(max(rel_g) / max(max(rel_t), 1e-12), 2),
         "argmax_flips": "%d/%d" % (flips, len(xs)),
         "top1_margin_min": round(min(margins), 5) if margins else None}
    print(json.dumps(r), flush=True)
    return r


xs = lambda k: [torch.randn(k) for _ in range(N_X)]
shard, key = locate(".linear_attn.in_proj_a.weight")
res = [grade("in_proj_ba[%s]" % key.rsplit(".", 2)[-2],
             load_rows(shard, key), xs(k=5120))]
shard, key = locate(".lm_head.weight")
res.append(grade("lm_head[:%d]" % N_ROWS_LM_HEAD,
                 load_rows(shard, key, N_ROWS_LM_HEAD), xs(k=5120)))

for r in res:
    assert r["ratio"] <= 3.0, \
        "%(case)s: skinny GEMV is %(ratio)sx Tensile's fp64 error (>3x) -> not a" \
        " drop-in, stop and revisit the kernel choice" % r
    assert r["rel_gemv_max"] < 5e-3, "%s: absolute error too large" % r["case"]
os.makedirs(OUT, exist_ok=True)
json.dump(res, open("%s/p05_accuracy.json" % OUT, "w"), indent=1)
print("ALL PASS -> %s/p05_accuracy.json" % OUT)
```

- [x] **Step 2: 跑**

Run: `./p0_sync.sh p05_accuracy.py && ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 /work/p0/p05_accuracy.py" | tee logs/p0/p05_accuracy.log`
Expected: 两行 JSON + `ALL PASS`。
`argmax_flips` **允许非零**（两个不同内核的 fp16 舍入不同，量级 4e-4 相对误差）；把实测值与 `top1_margin_min` 写进「执行记录」，Task 4 Step 5 按它解释分叉。

- [x] **Step 3: Commit**

```bash
git add p05_accuracy.py logs/p0/p05_accuracy.log
git commit -m "p05: numeric gate for the skinny GEMV on real checkpoint weights"
```

---

## Task 3: 补丁 + 门单元测试（先让它红）

**Files:**
- Create: `docker/patches/layers_utils.py`
- Test: `p05_gate_test.py`
- 安装目标：`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py`

- [x] **Step 1: 先写失败的门测试 `p05_gate_test.py`**

```python
# p05_gate_test.py -- which calls must reach ops.LLMM1, and which must not?
# Both seams are monkeypatchable: rocm_unquantized_gemm_impl looks up ops.LLMM1
# at call time, and imports on_gfx906 from vllm.platforms.rocm *inside* the
# function body, so the import re-reads the module attribute on every call.
#   ssh x@192.168.10.11 'docker exec -i vllm-gfx906 python3 /work/p0/p05_gate_test.py'
import torch
import vllm._custom_ops as ops
import vllm.model_executor.layers.utils as U          # noqa: F401 (registers op)
import vllm.platforms.rocm as R
from vllm import envs

calls = []
_real = ops.LLMM1
ops.LLMM1 = lambda a, b, rpb: (calls.append(rpb), _real(a, b, rpb))[1]
torch.manual_seed(7)


def dispatch(n, m, k, dtype=torch.float16, bias=False, contig=True,
             gfx906=True, skinny=True):
    """One call through the registered custom op. Returns (#LLMM1 calls, out)."""
    w = (torch.randn(k, m, device="cuda", dtype=torch.float16).t() if not contig
         else torch.randn(m, k, device="cuda", dtype=dtype) / k ** 0.5)
    if contig and dtype != torch.float16:
        w = w.to(dtype)
    x = torch.randn(n, k, device="cuda", dtype=dtype)
    b = torch.ones(m, device="cuda", dtype=torch.float16) if bias else None
    envs.VLLM_ROCM_USE_SKINNY_GEMM = skinny
    orig, R.on_gfx906 = R.on_gfx906, (lambda: gfx906)
    calls.clear()
    try:
        y = torch.ops.vllm.rocm_unquantized_gemm(x, w, b)
    finally:
        R.on_gfx906 = orig
        del envs.VLLM_ROCM_USE_SKINNY_GEMM      # restore the lazy default
    torch.cuda.synchronize()
    return len(calls), y


def expect(label, n, want):
    print("%-42s LLMM1=%d want=%d %s" % (label, n, want, "ok" if n == want else "FAIL"))
    assert n == want, label


got, y = dispatch(1, 48, 5120)
expect("M=1 in_proj_ba-shaped -> skinny", got, 1)
assert tuple(y.shape) == (1, 48), y.shape
expect("M=1 replicated N=96 -> skinny", dispatch(1, 96, 5120)[0], 1)
expect("M=2 must stay on Tensile", dispatch(2, 48, 5120)[0], 0)
expect("bias present -> fallback", dispatch(1, 48, 5120, bias=True)[0], 0)
expect("k=12288 > 8192 -> fallback", dispatch(1, 48, 12288)[0], 0)
expect("m=50 (m%4!=0) -> fallback", dispatch(1, 50, 5120)[0], 0)
expect("non-contiguous weight -> fallback", dispatch(1, 48, 5120, contig=False)[0], 0)
expect("not gfx906 -> unchanged", dispatch(1, 48, 5120, gfx906=False)[0], 0)
expect("VLLM_ROCM_USE_SKINNY_GEMM=0 -> fallback",
       dispatch(1, 48, 5120, skinny=False)[0], 0)
# bf16: default gate is float16-only. If Task 1 reported bf16_ok=True and you
# used the bf16 variant of the dtype clause, flip the expectation to 1.
expect("bf16 (fp16-only gate) -> fallback",
       dispatch(1, 48, 5120, dtype=torch.bfloat16)[0], 0)
print("ALL PASS")
```

- [x] **Step 2: 跑它，确认在未打补丁的树上是红的**

Run: `./p0_sync.sh p05_gate_test.py && ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 /work/p0/p05_gate_test.py"`
Expected: **FAIL** —— 第一行打印 `M=1 in_proj_ba-shaped -> skinny  LLMM1=0 want=1 FAIL` 并抛 `AssertionError`。这条红就是 P0 trace 的复现：证明「M=1 稠密 GEMM 今天不走 skinny」。

- [x] **Step 3: 从容器取原始文件作为权威副本底本**

Run: `mkdir -p docker/patches && ssh x@192.168.10.11 "docker cp vllm-gfx906:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py /home/x/vllm-work/p0/layers_utils.orig.py && md5sum /home/x/vllm-work/p0/layers_utils.orig.py" && scp -q x@192.168.10.11:/home/x/vllm-work/p0/layers_utils.orig.py docker/patches/layers_utils.py`
Expected: md5 `43a4addb9ab34c654baf9616df5b2079`。**不同则先停下来 diff** —— 说明容器里的文件与本计划所依据的版本不一致。

- [x] **Step 4: 补 import（`docker/patches/layers_utils.py:119`）**

原文（函数体内的局部 import）：

```python
    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250
```

改为：

```python
    from vllm.platforms.rocm import (
        on_gfx1x, on_gfx9, on_gfx906, on_gfx950, on_gfx1250)
```

> 保持局部 import 不动是有意的：它让 `R.on_gfx906` 的 monkeypatch 在每次调用时重新生效，Task 3 的门测试依赖这一点。

- [x] **Step 5: 插入门与分支**

定位 `use_skinny` 块结束处（原文 `            return out.reshape(*x.shape[:-1], weight.shape[0])` 的第二处，紧跟 `ops.LLMM1` 那行之后）与下一句 `if rocm_aiter_ops.is_tgemm_enabled():` 之间，插入。**放在 aiter/tgemm 判断之前**是有意为之：一旦 aiter 可用，本分支否则永远轮不到。

默认版（bf16 未通过 Task 1）：

```python
    # gfx906 is deliberately kept out of on_gfx9() so it can never inherit CDNA
    # (MFMA) fast paths -- see vllm/platforms/rocm.py _ON_GFX906. The single
    # exception measured on this hardware is the M=1 GEMV: Tensile answers every
    # K=5120 M=1 shape with a serial K loop (318 us flat for N=32..8192, and the
    # same 318 us at mclk 350 and 1000 MHz -- not bandwidth bound at all), while
    # LLGemm1_kernel reads lm_head's 1.27 GB/rank at 877 GB/s = 94% of this
    # card's measured BW_eff (931 GB/s). wvSplitK stays off: it has never run on
    # gfx906. weight.is_contiguous() (inside skinny_operands_compatible) is load
    # bearing -- LLMM1 ignores strides and silently returns garbage for a
    # transposed view.
    use_gfx906_gemv = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx906()
        and x.dtype == torch.float16
        and bias is None
        and n == 1
        and m % 4 == 0
        and k % 8 == 0
        and k <= 8192
        and skinny_operands_compatible
    )
    if use_gfx906_gemv:
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        out = ops.LLMM1(weight, x_view, 4)
        return out.reshape(*x.shape[:-1], weight.shape[0])

```

bf16 变体：仅把谓词里的 `and x.dtype == torch.float16` 这一行换成

```python
        and x.dtype in [torch.float16, torch.bfloat16]
```

- [x] **Step 6: 装进容器并跑测试（此时允许 FORCE=1，Task 6 才补 Dockerfile）**

Run: `FORCE=1 ./p05_install.sh docker/patches/layers_utils.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py`
Expected: `ok docker/patches/layers_utils.py -> vllm-gfx906:.../layers/utils.py <md5>`

Run: `ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 /work/p0/p05_gate_test.py"`
Expected: 10 行 `ok`，末行 `ALL PASS`。
若任何 `-> fallback` 行变成 `LLMM1=1`：门放得太宽（把 bias / M>1 / 非连续也换出来了），**修谓词，不要改测试期望**。

- [x] **Step 7: 确认 GPU 上跑的确实是新代码（「缓存键不含内核选择」的同款坑）**

Run: `ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 -c \"import inspect, vllm.model_executor.layers.utils as u; print('use_gfx906_gemv' in inspect.getsource(u.rocm_unquantized_gemm_impl))\""`
Expected: `True`

- [x] **Step 8: Commit**

```bash
git add docker/patches/layers_utils.py p05_gate_test.py
git commit -m "feat(p05): route gfx906 M=1 dense fp16 GEMM to the ROCm skinny GEMV"
```

---

## Task 4: 引擎内 A/B（交付路径的唯一可信数）

一次引擎启动同时产出 wall 与 per-worker trace，所以两臂各跑一遍 `p0_step.py` 即可。

**Files:**
- Create: `p05_ab.sh`

- [x] **Step 1: 写 `p05_ab.sh`**

```bash
#!/usr/bin/env bash
# p05_ab.sh -- run the P0 attribution harness twice inside one detached job:
#   ab_off: VLLM_ROCM_USE_SKINNY_GEMM=0 -> must reproduce 53.7 ms. That proves
#           the env lever reaches the workers AND that nothing else drifted
#           since the P0 trace, so ab_on is a single-variable comparison.
#   ab_on : patched default
# Sequential because each arm needs both cards; the 180 s gap is because the
# pair thermally trips if it goes straight back into load.
set -uo pipefail
for arm in off on; do
  out=/work/p0/ab_$arm
  mkdir -p "$out"
  if [ "$arm" = off ]; then export VLLM_ROCM_USE_SKINNY_GEMM=0
  else unset VLLM_ROCM_USE_SKINNY_GEMM; fi
  export OUT="$out" CLOCK_MHZ=1425
  if [ "$arm" = on ]; then export EXPECTED_MS=33.1; else unset EXPECTED_MS; fi
  echo "=== arm=$arm skinny=${VLLM_ROCM_USE_SKINNY_GEMM:-unset(=True)} OUT=$out EXPECTED_MS=${EXPECTED_MS:-auto} $(date -Is)"
  python3 /work/p0/p0_step.py 2>&1 | tee "/work/p0/ab_$arm.step.log"
  echo "=== arm=$arm rc=${PIPESTATUS[0]} $(date -Is)"
  sleep 180
done
echo "=== AB DONE $(date -Is)"
```

- [x] **Step 2: 起跑前检查（时钟 / 显存空 / 采样器活着）**

Run: `./p0_sync.sh p05_ab.sh p0_step.py p0_parse_trace.py`
Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh lock 4 2; docker exec vllm-gfx906 python3 -c 'import torch;print([round(torch.cuda.mem_get_info(i)[0]/1e9,1) for i in (0,1)])'; ls -l --time-style=+%H:%M:%S /home/x/vllm-work/blackbox_p0.log 2>/dev/null || echo NO-SAMPLER-LOG"`
Expected: 两卡空闲显存 > 14 GB；采样日志时间戳是刚刚（若断流，先按 `blackbox.sh` 重新起一个分离作业，它不影响测量但会影响热解释）。显存不足 ⇒ 有残留进程，**先查清是谁的**，不要杀不认识的任务。

- [x] **Step 3: 分离式启动（绝不前台跑）**

Run: `ssh x@192.168.10.11 "docker exec -d vllm-gfx906 bash -lc '/work/p0/p05_ab.sh > /work/p0/ab.log 2>&1'"`
Expected: 立刻返回 exec id。**不要 sleep 轮询**；有意义的检查点是 `ab.log` 末尾出现 `=== AB DONE`（约 25 分钟）或采样日志断流（主机掉电）。

- [x] **Step 4: 读结果**

Run: `ssh x@192.168.10.11 "grep -E 'arm=|decode wall/step|perf gate|FATAL|worker traces' /work/p0/ab.log; echo ---; docker exec vllm-gfx906 cat /work/p0/ab_off/p0_wall.json /work/p0/ab_on/p0_wall.json" | tee logs/p0/p05_ab.log`
Expected：
- `ab_off.per_step_ms` = 53.7 ± 3（复现基线；差得多 ⇒ 条件漂了，先查时钟、`n_tok`、`capture_sizes`）
- `ab_on.per_step_ms` ≤ 36.0（⇒ ≥ 27.8 tok/s），且日志有 `perf gate OK`
- 两臂各 `worker traces: 2 files`
未达标 ⇒ **停**并记录实测值；不允许「先合进去再说」。

- [x] **Step 5: 语义没被改坏（greedy token 前缀对齐）**

Run: `ssh x@192.168.10.11 "docker exec -i vllm-gfx906 python3 -c \"import json;a=json.load(open('/work/p0/ab_off/p0_step_out.json'))['token_ids'];b=json.load(open('/work/p0/ab_on/p0_step_out.json'))['token_ids'];k=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),-1);print('len',len(a),len(b),'first-diff-token',k,'of',min(len(a),len(b)));print('OFF',json.load(open('/work/p0/ab_off/p0_step_out.json'))['text'][:200]);print('ON ',json.load(open('/work/p0/ab_on/p0_step_out.json'))['text'][:200])\"" | tee logs/p0/p05_tokens.log`
Expected（判定标准）：
- `first-diff-token -1`（完全一致）最好；
- 若 k ≥ 32：数值级分叉，用 Task 2 的 `argmax_flips`/`top1_margin_min` 解释（并列 logit 的 4e-4 舍入翻转），可接受，写进记录；
- 若 k < 32，或两侧 200 字里有任一侧语义跑偏（乱码、语种切换、重复塌陷）⇒ **阻断**，回退补丁并回到 Task 2 收紧数值门。

- [x] **Step 6: Commit**

```bash
git add p05_ab.sh logs/p0/p05_ab.log logs/p0/p05_tokens.log
git commit -m "p05: in-engine A/B for the M=1 skinny GEMV path"
```

---

## Task 5: trace 复测与桶表收口

证明时间真的从 `dense_fp16` 桶搬走了，而不是只相信 wall。

**Files:**
- Modify: `p0_buckets.py:12`（`import os`）、`:22-31`（BUCKETS）、`:31`（`P0`）

- [x] **Step 1: 聚合 `ab_on` 的 trace，从 TOP25 里读 skinny 内核的字面名**

Run: `ssh x@192.168.10.11 "docker exec -i -e OUT=/work/p0/ab_on -e PROF_DIR=/work/p0/ab_on/prof vllm-gfx906 python3 /work/p0/p0_parse_trace.py" | tee logs/p0/p05_parse_ab_on.log`
Expected: `p0_parse_trace.py` 自带 `TOP25 by device time` 列表。在列表里找 skinny GEMV 的字面符号（源 `skinny_gemms.cu:175` 的模板实例 `LLGemm1_kernel<half, 4>`，trace 里可能是 demangled 或 mangled，都含 `LLGemm1`）。核对 `count` ≈ 49/step（= 48 × in_proj_ba + 1 × lm_head）、`sum_us` ≈ 1 600–1 900 µs/step。
若 TOP25 里没有含 `LLGemm1` 的名字 ⇒ Task 4 的 `ab_on` 根本没走到 skinny（回去查 env 是否进了 worker，别改桶）。

- [x] **Step 2: 加桶 + 参数化目录（`p0_buckets.py`）**

`import re` 与 `import sys` 之间加 `import os`。
在 `BUCKETS` 里 `("dense_fp16", ...)` **之前**插入（顺序即优先级；必须排在 `w4a16_gemv` 之后以免抢走 exllama 命名）：

```python
    # P0.5: the ROCm skinny GEMV now answering M=1 dense projections. Kept
    # apart from dense_fp16 on purpose -- its whole point is that it is NOT the
    # Tensile serial-K path. Literal from p0_parse_trace.py's TOP25 dump.
    ("dense_gemv",  r"LLGemm1"),
```

`P0 = "/work/p0"` 改为：

```python
P0 = os.environ.get("OUT", "/work/p0")   # same env as p0_step / p0_parse_trace
```

- [x] **Step 3: 两臂各算一次桶表**

Run: `ssh x@192.168.10.11 "docker exec -i -e OUT=/work/p0/ab_on vllm-gfx906 python3 /work/p0/p0_buckets.py"` 再对 `ab_off` 重复一次（把 `OUT` 换掉），各自 `tee logs/p0/p05_buckets_ab_on.log` / `..._ab_off.log`
Expected：
- `ab_on`：`dense_fp16` ≈ 0（M>1 的形状不在纯 decode 窗口内），`dense_gemv` ≈ 1.6–1.9 ms/step，`w4a16_gemv` 仍 ≈ 20.0 ms/step（**未受影响 ⇒ 单变量的直接证据**），`ROUTING NUMBER` 仍 ≥ 0.8，`UNCLASSIFIED` < 5% ⇒ `RC=0`
- `ab_off`：与 `logs/p0/p0_buckets_final.log` 一致（`dense_fp16` ≈ 22.2 ms）
- 若 `RC=1` 且提示 unclassified >5%：**先扩 BUCKETS 再相信表**，不要动阈值。

- [x] **Step 4: 对账（wall 与 kernel-sum 口径差要写明）**

`ab_off 的 (dense_fp16 + w4a16_gemv) − ab_on 的 (dense_fp16 + dense_gemv + w4a16_gemv)` 应 ≈ `(ab_off.per_step_ms − ab_on.per_step_ms) × 1000`，容差 ±15%。两个数字与差值写进「执行记录」。

- [x] **Step 5: Commit**

```bash
git add p0_buckets.py logs/p0/p05_parse_ab_on.log logs/p0/p05_buckets_ab_on.log logs/p0/p05_buckets_ab_off.log
git commit -m "p05: add dense_gemv bucket and re-attribute the decode step"
```

---

## Task 6: 进交付镜像（两条构建路径 + 构建期与运行期各一条断言）

**关键事实（本计划踩到的配方细节）：** Path A 的 `final` 阶段（`Dockerfile:144` 起）原本**只重断言 entrypoint，不 COPY python 补丁** —— 那 3 个 python 补丁靠 `phase3-working` 快照自带。所以若只加断言不加 COPY，Path A 构建会在静态检查处红，除非先重新 `docker commit`。为了让这个补丁不依赖「有人记得 commit」，本任务在 **两条路径都 COPY**。

**Files:**
- Modify: `docker/Dockerfile.rocm_gfx906`（rebuild 阶段 `:94` 组、rebuild 断言 `:115` 组、final 阶段 `:159` 之后、final 断言 `:169` 组）
- Modify: `docker/prepare_build_context.sh`（第 3 步）
- Modify: `docker/entrypoint/selfcheck.sh`

- [x] **Step 1: rebuild（Path B）阶段加 COPY 与断言**

在 `COPY patches/triton_decode_attention.py ...`（`:96`）之后：

```dockerfile
# P0.5: gfx906 M=1 dense fp16 GEMM -> ROCm skinny LLGemm1_kernel (877 GB/s vs
# Tensile's serial-K 318 us/launch, 22.2 ms -> ~1.7 ms per decode step).
COPY patches/layers_utils.py ${SP}/vllm/model_executor/layers/utils.py
```

在 `:115` 那组静态检查里 `grep -q "is_gfx906" ...` 之后：

```dockerfile
    && grep -q use_gfx906_gemv ${SP}/vllm/model_executor/layers/utils.py \
```

- [x] **Step 2: final（Path A）阶段也加 COPY —— 这是本次的新做法，注释要写清**

`COPY entrypoint/ /opt/gfx906/` 之前插入：

```dockerfile
# The snapshot carries the older python patches, but this one must not depend
# on someone remembering to re-commit :phase3-working: re-COPY the authoritative
# repo copy so :latest is correct by construction.
COPY patches/layers_utils.py ${SP}/vllm/model_executor/layers/utils.py
```

`:169` 那组 final 静态检查里同样加一条 `&& grep -q use_gfx906_gemv ${SP}/vllm/model_executor/layers/utils.py \`。

- [x] **Step 3: `prepare_build_context.sh` 第 3 步收集清单加一行**

在 `docker cp "$CONTAINER:$SP/gfx906_gemv_fix.py" ...` 之前：

```bash
docker cp "$CONTAINER:$SP/vllm/model_executor/layers/utils.py" \
  "$CTX/patches/layers_utils.py"
```

- [x] **Step 4: `selfcheck.sh` 运行期断言**

在 `chk "capture-safe gptq_gemm_out (python side)" ...` 之后：

```bash
chk "gfx906 M=1 dense GEMV on skinny LLGemm1" \
    grep -q use_gfx906_gemv "$SP/vllm/model_executor/layers/utils.py"
```

- [x] **Step 5: 三方一致性校验（权威副本 == 构建上下文 == 容器）**

Run: `./p05_install.sh docker/patches/layers_utils.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py`
Expected: `ok ...`（这次不该再需要 FORCE —— Step 1/2 已把该目标写进 Dockerfile；若仍报 FATAL，说明 grep 没匹配上 `${SP}/...` 的拼法，先修 `p05_install.sh` 别绕过它）

Run: `scp -q docker/Dockerfile.rocm_gfx906 docker/prepare_build_context.sh x@192.168.10.11:/home/x/vllm-work/docker/ && ssh x@192.168.10.11 "cd /home/x/vllm-work/docker && bash prepare_build_context.sh 2>&1 | tail -5 && echo -n 'build-context: ' && md5sum build-context/patches/layers_utils.py | cut -c1-32 && echo -n 'container:     ' && docker exec vllm-gfx906 md5sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py | cut -c1-32 && echo -n 'hits:          ' && grep -c use_gfx906_gemv build-context/patches/layers_utils.py"`
Expected: `build-context:` 与 `container:` 两行是**同一个 md5**，且 `hits: 2`（谓词 + `if`）。
本地那份的 md5 已由本步骤上一条 `p05_install.sh` 打印，三者必须一致。
`dist-packages.tar.gz` 已存在时不会重做（它抓的是**当前容器状态**，已含补丁）⇒ 无需 `FORCE=1`。

- [x] **Step 6: 构建两条路径并跑开箱自检（构建不占 GPU，但耗时长，仍按分离作业处理）**

Path A 的构建上下文就是 `docker/` 目录本身（`COPY entrypoint/`、`COPY patches/...` 都相对它解析），所以先把上下文同步过去：

Run: `scp -q -r docker/patches docker/entrypoint x@192.168.10.11:/home/x/vllm-work/docker/ && ssh x@192.168.10.11 "cd /home/x/vllm-work/docker && nohup sh -c 'docker build -f Dockerfile.rocm_gfx906 -t vllm-gfx906:latest . && docker build -f Dockerfile.rocm_gfx906 --target rebuild -t vllm-gfx906:rebuilt build-context/' > /home/x/vllm-work/p0/p05_build.log 2>&1 & echo started"`
Expected: `started`。之后查 `ssh x@192.168.10.11 "grep -E 'naming to|Successfully tagged|ERROR|use_gfx906_gemv' /home/x/vllm-work/p0/p05_build.log | tail"` —— 两条 tagged/`naming to` 成功、无 `ERROR`，且构建期静态检查全过（含新加的 `grep -q use_gfx906_gemv`；不通过会让 build 直接失败，这正是断言存在的意义）。
Path B 只改了 `patches/`，18 GB 解包层命中缓存 ⇒ 比 C2 那次快得多；仍然**耐心等日志，不要中途重跑**。若磁盘吃紧，先 `docker image rm vllm-gfx906:rebuilt` 释放上一次的同名旧层（这是唯一可以先删的产物，它随时可由 Path B 重做）。

Run: `ssh x@192.168.10.11 "docker run --rm --device /dev/kfd --device /dev/dri -v /home/x/models:/home/x/models:ro vllm-gfx906:latest gfx906-selfcheck" | tee logs/p0/p05_selfcheck.log`
Expected: 新增那条 `ok`，且整表无 `FAIL`。

- [x] **Step 7: 确认交付开关不依赖环境变量**

`VLLM_ROCM_USE_SKINNY_GEMM` 默认 `True`（`envs.py:152` 与 `:1348`），所以 `docker/entrypoint/serve.sh` **不需要改动**。但镜像 ENV 与容器 ENV 不同款坑要反向确认一次：
Run: `ssh x@192.168.10.11 "docker run --rm vllm-gfx906:latest bash -lc 'echo SKINNY=[\${VLLM_ROCM_USE_SKINNY_GEMM:-unset}]'"`
Expected: `SKINNY=[unset]`（⇒ 取代码默认 True）。若打印出 `0`，说明某处 ENV 把它关了，必须在 serve.sh 里显式设回 True。

- [x] **Step 8: Commit**

```bash
git add docker/Dockerfile.rocm_gfx906 docker/prepare_build_context.sh docker/entrypoint/selfcheck.sh docker/patches/layers_utils.py logs/p0/p05_selfcheck.log
git commit -m "build(p05): ship the gfx906 skinny-GEMV dispatch on both build paths"
```

---

## Task 7: 收口、文档与 #20 的输入更新

- [x] **Step 1: 恢复环境**

Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh unlock && bash /home/x/vllm-work/p0/p0_clock.sh status; docker exec vllm-gfx906 python3 -c 'import torch;print([round(torch.cuda.mem_get_info(i)[0]/1e9,1) for i in (0,1)])'"`
Expected: 两卡 `auto`（待机 860/350），显存回到空闲。

- [x] **Step 2: 写 `docs/superpowers/plans/2026-09-08-gfx906-p05-report.md`**

结论先行（tok/s 前后对照），必须含：一行 diff（门谓词）、A/B 两臂 `per_step_ms`/tok·s⁻¹/rc、两臂五桶对照 + 对账残差、数值门（真实权重 rel 与 `argmax_flips`）、greedy token 分叉位置、镜像两条构建路径与 `gfx906-selfcheck` 输出、**已知偏差**（1425/1000 锁频 ⇒ 绝对值偏保守；`ab_on` 的 `EXPECTED_MS` 是注入的；bf16 是否放开取决于 Task 1 结论；`N_ROWS_LM_HEAD=8192` 抽样）、**后续**（`M=2..5` 的 `wvSplitK` 从未在 gfx906 上跑过；`in_proj_ba` 的 3.9 µs 对 1.1 µs 地板是 12 block / 60 CU 的占用不足，`rows_per_block=2` 略好但只值 19 µs/step；1700 MHz 交付态复测）。

- [x] **Step 3: 更新 #20 的 P1 输入**

在 P0 报告末尾追加：`dense_fp16 桶已从归因中消失（22.2 -> ~1.7 ms），#20 的头部空间即 W4A16 GEMV 19 992 µs/step 对 6 788 µs 地板（33% 效率）`；并在 `docs/superpowers/plans/2026-09-04-gfx906-w4a16-p0-attribution.md` 执行记录里把 P0.5 标完成。

- [x] **Step 4: 记忆**

`gfx906-env-pitfalls` 追加：**vLLM 的 `_ON_GFX9` 有意排除 gfx906，所以 ROCm 的 skinny / M=1 快路径默认全部不可达；判断「这条路径没生效」时先看这类 arch 谓词，别怀疑内核没编进来（`PYTORCH_ROCM_ARCH=gfx906` 已把 `_rocm_C` 全量编好并实测可跑）。** 另在 `final-goal-shippable-image` 记 P0.5 已进镜像 + 前后 tok/s。

- [x] **Step 5: Commit**

```bash
git add docs/superpowers/plans/ logs/p0/
git commit -m "docs(p05): close out the dense M=1 GEMV swap, update #20 P1 inputs"
```

---

## 门（任一不过即停，不做「先合再说」）

| # | 门 | 位置 | 不过意味着 |
|---|---|---|---|
| G1 | LLMM1 在 gfx906 上可运行、确定性、误差 ≤3× Tensile | Task 1 Step 3 | 计划前提被证伪，回到 #20 |
| G2 | 收益：`in_proj_ba` ≥4×、`lm_head` ≥70% BW_eff | Task 1 Step 3 | 22 ms 拿不回来，不值得改 dispatch |
| G3 | 真权重 rel ≤ 3× Tensile 且 < 5e-3 | Task 2 Step 2 | 语义风险，需换内核或只修 lm_head |
| G4 | 门谓词精确：M>1 / bias / 非连续 / k>8192 / m%4 全部不接管 | Task 3 Step 6 | 会静默换错形状 ⇒ 数值/崩溃 |
| G5 | `ab_off` 复现 53.7 ± 3 ms 且 `ab_on` ≤ 36.0 ms | Task 4 Step 4 | 单变量假设或收益不成立 |
| G6 | greedy 前 32 token 一致，且无语义跑偏 | Task 4 Step 5 | 阻断，回退补丁 |
| G7 | 桶表 `UNCLASSIFIED` < 5%、`w4a16_gemv` 不变 | Task 5 Step 3 | 归因被污染，表不可信 |
| G8 | 两条构建路径静态断言全过 + `gfx906-selfcheck` 无 FAIL | Task 6 Step 6 | 改动没真正进镜像（交付标准） |

---

## 自查（作者，2026-09-08）

- **覆盖**：P0 报告给 P0.5 的三句要求 —— 换出两个稠密 GEMM（Task 3）、收益 ≥5%（G5 要求 ~38%）、可进镜像（Task 6 + G8）。P0 的教训「不要用微基准反推内核状态」⇒ Task 4 以引擎内 wall 为准、Task 5 以 trace 对账；「时钟要两把都锁」⇒ Task 1/4 的 Step 2 都显式锁 `4 2`。
- **无占位符**：唯一「按实测填」的是 Task 5 Step 1 的内核字面名，已给出取得它的完整命令与失败判据；Task 3 Step 5 的 bf16 两版代码都写全了。
- **名字/类型一致**：谓词统一 `use_gfx906_gemv`（测试、两条 Dockerfile 断言、selfcheck 都用这一字面量）；目录 env 统一 `OUT`（`p0_step` / `p0_parse_trace` / `p0_buckets`）；`rows_per_block=4` 与上游一致；文件权威副本只有一个 `docker/patches/layers_utils.py`。
- **最弱环**：Task 4 Step 5 依赖 Task 0 Step 4 给 `p0_step.py` 存的 token_ids —— 若那一步漏做，「输出没变」这句话就没有证据。G6 排在 G5 之后，顺序不能颠倒。

---

## 执行记录（执行时填）

> 勾选表示步骤已执行；各任务的「Commit」步骤按 Task 0 Step 1 的约定降级为 `git add` 暂存 ⇒ **勾上＝已暂存，尚未提交**（等 `user.name`/`user.email`）。

### Task 0（前置门与 harness 参数化）— 完成，待提交

- `p0_step.py`：`OUT` / `EXPECTED_MS` 参数化、`req_times` 取最小并保留 `out`、落盘 `p0_step_out.json`（token_ids + text）。容器内 `python3 -m py_compile` 通过（`COMPILE-OK`），md5 `a39cd0df0b72af146889897d5d786002`。
- `p05_install.sh`：md5 `f2627222de3277b405f19e5c6d194776`，`bash -n` 通过。
- **计划自带缺陷 #1（已修）**：交付守卫用 `sed 's/[.[\*^$]/\\&/g'` 做 ERE 转义 —— sed 替换侧的 `&` 展开成整个匹配，于是 `.` 变成了 `&`，结果**连在 Dockerfile 里的 `container_src/wna16_exllama.py` 也被判 reject**（假阴性 = 守卫全盲，任何安装都会被 FORCE 绕过）。改为固定串匹配 `grep -qF -e "$DST" -e "dist-packages/${REL}" -e '${SP}/'"${REL}"`。三例复测：在配方里的 ⇒ pass；不在的 ⇒ reject + exit 1；`FORCE=1` ⇒ 打印 FATAL 后放行。
- **提交降级**：仓库 0 提交且 `git config` 无 `user.name`/`user.email`（长期约束：不许改 git config）⇒ 各任务「Commit」步骤改为 `git add` 暂存，等身份。不阻塞测量。

### Task 1（裸测）— G1 PASS、G2 PASS

条件：sclk 4=1425 MHz / mclk 2=1000 MHz（两把都锁，`p0_clock.sh lock 4 2` 已回显两卡）、单卡、TP 无关、`BW_EFF=931`、CUDA-graph 计时。落盘 `/work/p0/p05_skinny.json`，本地副本 `logs/p0/p05_skinny.log`（`*.log` 被 .gitignore 第 2 行忽略，故证据以本节 + 宿主产物为准）。

| case | gemv_us | tensile_us | 比 | %BW_eff | rel vs fp64 | Tensile rel | 确定性 |
|---|---|---|---|---|---|---|---|
| `in_proj_ba` m48 k5120 fp16 | **4.6** | 309.4 | **67×** | 11.5 | 4.12e-4 | 1.97e-4 | stable |
| `lm_head` m124160 k5120 fp16 | **1468.3** | 6637.4 | 4.5× | **93.0**（865.9 GB/s） | 4.77e-4 | 4.13e-4 | stable |
| `in_proj_ba_bf16` | 6.6 | 1181.2 | 179× | 8.0 | 4.41e-3 | 2.36e-3 | stable |
| `lm_head_bf16` | 1851.7 | 45276.8 | 24× | 73.7 | 3.81e-3 | 2.65e-3 | stable |
| `noncontig_weight` | 4.6 | 427.9 | — | — | **1.357（=垃圾，且不抛异常）** | 2.21e-4 | stable |
| `rpb1/2/4/8` m48 | 4.6 / **4.2** / 4.6 / 5.9 | 309.5/309.5/309.4/309.3 | — | — | 4.12e-4 | 1.97e-4 | stable |

- **G1**：两形状均「可运行 + stable + rel < 3×Tensile+1e-5」（4.12e-4 < 6.0e-4；4.77e-4 < 1.25e-3）。✅
- **G2**：`in_proj_ba` 67× ≫ 4×；`lm_head` 93.0% > 70%。✅
- **连续性陷阱确认**：转置视图照跑不误、rel=1.357 静默返回垃圾 ⇒ `weight.is_contiguous()` 是护栏，注释按此写。
- **Step 4 结论：`bf16_ok = True`** ⇒ Task 3 Step 5 采用**含 bf16 的 dtype 条件**那一版。
- `rows_per_block` 最优是 2（4.2 vs 4.6 µs，差 0.4 µs × 48 = 19 µs/step，噪声级）⇒ 沿用上游 **4**，不加旋钮。
- **计划自带缺陷 #2（已修）**：fp64 参考写成 `w.double() * x.t().double()` —— `w` 是 `[m,k]`、`x` 是 `[1,k]`，转置后拿 `k` 去广播 `m`，第一次 `check()` 就 `RuntimeError: The size of tensor a (48) must match the size of tensor b (5120)`。正解 `w.double() * x.double()` 后 `sum(dim=1)`。顺带把 `ten` 行那句「segfaults on gfx906」的错误注释改成「incumbent Tensile」。
- **按本表重算收益**（比写计划时的 33.1 ms 略保守）：`48×(309.4−4.6) = 14 630 µs` + `(6637.4−1468.3) = 5 169 µs` = **19 799 µs/step** ⇒ `53.678 − 19.799 = 33.9 ms`（29.5 tok/s，+59%）。用引擎内桶值（318 / 6982 µs）算则得 20 551 µs ⇒ 33.1 ms。⇒ Task 4 的 `ab_on` 门槛取 **≤ 36.0 ms**，区间 33.1–33.9 ms 均在带内。

### Task 2（真实权重数值门）— G3 PASS

`/work/p0/p05_accuracy.json`。权重从 checkpoint 原样取（BF16 存储 ⇒ 顺带覆盖 bf16 分支），输入是 `in_proj_ba` 真实激活与 lm_head 抽样 8192/248320 行：

| case | m×k | gemv rel vs fp64 | Tensile rel | 比 | argmax 翻转 | top1 margin min |
|---|---|---|---|---|---|---|
| `in_proj_ba[in_proj_a]` | 48×5120 | 7.88e-4 | 4.08e-4 | **1.93** | 0/64 | 2.56e-3 |
| `lm_head[8192/248320]` | 8192×5120 | 6.08e-4 | 4.49e-4 | **1.35** | 0/64 | 3.89e-3 |

**G3**：两例都满足「rel ≤ 3×Tensile 且 < 5e-3」（7.88e-4 < 1.22e-3；6.08e-4 < 1.35e-3）。✅ 且 margin 远大于误差比 ⇒ 换内核不会改变贪心解码的 argmax。
- **计划自带缺陷 #3/#4（已修）**：`locate()` 的后缀写成了 `.lm_head.weight`（checkpoint key 是不带前导点的 `lm_head.weight`）⇒ 改为不带点；lm_head 取前 N 行会只采样控制符/空格 token 而让 margin 统计失真 ⇒ 改 `torch.linspace` 全词表等距抽样。

### Task 3（补丁 + 门单元测试）— G4 PASS

权威副本 `docker/patches/layers_utils.py`（md5 `55379cbd3b0284264bfbd4e0cd1dd13e`，基线 md5 固定 `43a4addb9ab34c654baf9616df5b2079`）。谓词独立于上游 `use_skinny` 块（`wvSplitK` 在 LLMM1 之前，且 `_ON_GFX9` 不含 gfx906 ⇒ 上游那条永远不可达），只接管 `n==1 && m%4==0 && k<=8192 && bias is None && fp16/bf16 && 权重连续`，并显式 `x.reshape(-1, x.size(-1)).contiguous()`。

`p05_gate_test.py`：打补丁前 2 条红（两个 M=1 期望 LLMM1=1 实得 0），补丁后 **10/10 ALL PASS** —— M=2、bias、非连续权重、k=12288、m=50（m%4≠0）、非 gfx906、`VLLM_ROCM_USE_SKINNY_GEMM=0` 全部保持 Tensile；bf16 与 N=96 复制形状接管。✅ G4。

### Task 4（引擎内 A/B）— G5 PASS、G6 PASS

单变量 = `VLLM_ROCM_USE_SKINNY_GEMM`（在 opaque custom op 内读，不进编译图）；两臂 `rc=0`，锁频 1425/1000：

| 臂 | per_step_ms | tok/s | prefill_ms | req(256 tok)_ms |
|---|---|---|---|---|
| `ab_off`（=0） | 53.663 | 18.6 | 471.8 | 14 156.0 |
| `ab_on`（unset=True） | **33.008** | **30.3** | 465.9 | 8 882.8 |

**G5**：off 复现 53.7±3 ✅；on 33.0 ≤ 36.0 ✅（落在 Task 1 重算的 33.1–33.9 预测带内）。收益 20.66 ms/step = **-38.5%**。
**G6**：两臂 `p0_step_out.json` md5 相同（`7db6f866e11b8f8886a8f14337bbe3d2`）⇒ **256/256 贪心 token 逐字一致**，远强于「前 32 一致」。✅
- **计划自带缺陷 #5/#6（已修）**：Task 4/5 的 `git add logs/p0/*.log` 会落空 —— `.gitignore` 是「`*.log` 全拒 + `!p0_*.log` 白名单」，而 `p05_*.log` 两条都不匹配 ⇒ 白名单补 `!p05_*.log`，p05 的证据日志随报告入库（数字同时誊在本记录，宿主产物是原始件）。另外采样器 `blackbox.sh` 原本挂在会话级 ssh 管道上，ssh 一断日志就停 ⇒ 改 `nohup setsid bash /home/x/blackbox.sh > .../blackbox_ab.log`。

### Task 5（trace 复测与桶表收口）— G7 PASS

`p0_buckets.py` 新增 `dense_gemv`（`LLGemm1`）桶并读 `OUT`：

| bucket | off µs/step | on µs/step | Δ | launches |
|---|---|---|---|---|
| dense_fp16 → **dense_gemv** | 22 207.2 | **1 707.5** | **−20 499.7** | 3136（不变） |
| w4a16_gemv | 19 823.8 | 19 866.6 | +42.8（0.2%） | 16384 |
| rccl_wait | 7 237.7 | 7 604.4 | +366.7 | 8320 |
| elementwise | 4 035.8 | 4 086.7 | +50.9 | 43456 |
| gdn / attention / memcpy | 768.0 / 515.4 / 127.6 | 847.3 / 513.6 / 126.2 | +79.3 / −1.8 / −1.4 | — |
| UNCLASSIFIED | 7.5（0.0%） | 7.6（0.0%） | — | 64 |

**G7**：UNCLASSIFIED 0.0% < 5% ✅、`w4a16_gemv` 未变（+0.2%，噪声）✅。对账：计算桶（去 rccl_wait）47 485 → 27 156 µs，Δ 20 330 µs 对 wall Δ 20 656 µs，残差 326 µs = 1.6% —— 与 ROUTING 行 0.88 → 0.82 一致。`dense_fp16` 桶整桶消失即本任务的目标达成，#20 的头部空间现在就是 W4A16 GEMV 的 19 867 µs/step 对 6 788 µs 地板。

### Task 6（进交付镜像）— G8 PASS

配方四处改动两条路径都齐：Path B `COPY patches/layers_utils.py`（`:101`）+ 静态断言（`:124`）；Path A 同样 COPY（`:165`）+ 断言（`:182`）；`prepare_build_context.sh` 第 3 步收集；`selfcheck.sh` 运行期 `chk`。

**计划自带缺陷 #7（已修，且是本次最贵的一条）**：Task 6 假设 Path A 的 base `vllm-gfx906:phase3-working` 可用 —— 实测它是 2026-08-31 的 dev-container commit，**早于 B3/B4/C1**，连配方自己的 5 条断言都过不去（缺 `gptq_gemm_out` python+编译、缺 `gfx906_gemv_fix.py/.pth`）；而唯一「能过断言」的 commit（`:c1-entrypoint-working`）带 **16 GB `/root/.cache`**（9.7G pip、3.6G vLLM 编译、1.8G AOT W4A16 且缓存键不含内核选择）。⇒ Path A 的 `SNAPSHOT_IMAGE` 默认改为 **`:rebuilt`**（Path B 产物，`/root/.cache` 不存在），构建顺序改为 **B → A**（A 依赖 B 的 tag），并给 `final` 阶段补上与 Path B 同款的 `/root/.cache` 为空守卫，防止将来有人 `--build-arg SNAPSHOT_IMAGE=<dev commit>` 把 16 GB 缓存静默发出去。
**计划自带缺陷 #8（已修）**：三方一致性只覆盖 python，但 `prepare_build_context.sh` 第 4 步是 `docker cp $CONTAINER:/opt/gfx906/.` ⇒ 只改仓库里的 `selfcheck.sh` 不会进 Path B 镜像。已 `docker cp` 进容器并复验 md5。

一致性链（同一 md5 `55379cbd…`）：仓库 == 容器 `$SP/…/layers/utils.py` == `build-context/patches/layers_utils.py` == 镜像内文件；`grep -c use_gfx906_gemv` = 2（谓词 + `if`）。

构建（`p0/p05_build2.log`，第一轮）：`PATH B OK 08:16:22`（`0236cb297106`，18 GB 解包层命中缓存）→ `PATH A OK 08:18:48`（`a43d33b25658`，69.8 GB）⇒ **BOTH-OK，两阶段静态断言全过**。
第一轮之后又改了随镜像分发的文档（`QUICKSTART.md` 的基线一节 + `BUILD_GUIDE.md` 的两条路径说明）⇒ **必须再走一遍**（构建期断言只 `test -f QUICKSTART.md`，不校验内容）。第二轮（`p05_build3.log`）：`PATH B OK 08:31:56`（`7e5d830de21c`）→ `PATH A OK 08:34:27`（**最终 `:latest` = `1e18c8ea6b8b`**）。
⇒ **固化一条流程规则**：改 `docker/entrypoint/*` 的正确顺序是「仓库 → `docker cp` 进容器 → `prepare_build_context.sh` → 两条路径重建 → 镜像内 md5 复验」，因为第 4 步的流向是容器 → 构建上下文，只改仓库不会进 Path B。最终 md5 三方一致：`QUICKSTART 76787f6c…` / `selfcheck 6796844b…` / `layers_utils 55379cbd…`（仓库 == 容器 == build-context == 镜像）。

开箱自检（真机双卡，`docker run --rm --device /dev/kfd --device /dev/dri … vllm-gfx906:latest gfx906-selfcheck`）：**SELFCHECK PASSED**，20 条 `[ok]`、`grep -c FAIL` = 0，含新增的 `[ok] gfx906 M=1 dense GEMV on skinny LLGemm1`，且 GPU 分支的 `fp32 mv correct`/`fp16 matmul finite`/`W4A16 graph-capture op exposed` 全过（`p0/p05_selfcheck.log`）。
加测（交付镜像内跑门测试，证明不是「字符串在但路径不通」）：`docker run --rm --device … vllm-gfx906:latest python3 /work/p0/p05_gate_test.py` ⇒ **ALL PASS 10/10**。
**Step 7**：镜像 ENV 只有 `HSA_OVERRIDE_GFX_VERSION/VLLM_TRITON_ATTN_NUM_SEGMENTS/VLLM_WNA16_EXLLAMA/PYTORCH_ROCM_ARCH/VLLM_TARGET_DEVICE`，`SKINNY=[unset]` ⇒ 取代码默认 True ⇒ 交付默认即 33.0 ms/step 臂，`serve.sh` 无需改动。✅ G8。
Step 8 的 `git add` 见下（无 git 身份 ⇒ 暂存不提交）。

### Task 7（收口、文档与 #20 输入）— 完成，提交降级为暂存

**Step 1 环境恢复**：`p0_clock.sh unlock` ⇒ 两卡 `auto`（待机 sclk 860 / mclk 350），`torch.cuda.mem_get_info` 空闲 **17.1 GB / 17.1 GB**，`rocm-smi` util 0%，`blackbox.sh` 采样器已停（用 `pkill -f "blac[k]box.sh"`，见缺陷 #10）。
**Step 2 报告**：`docs/superpowers/plans/2026-09-08-gfx906-p05-report.md`（结论先行：18.6 → 30.3 tok/s）。含门谓词一行 diff、为何不放宽 `_ON_GFX9`、G1..G8 全表、两臂 `per_step_ms`/tok·s⁻¹/rc、五桶对照 + 1.6% 对账残差、真实权重 rel 与 `argmax_flips=0/256`、greedy token 零分叉、两条构建路径与 `SELFCHECK PASSED 20/0`、5 条已知偏差、后续（`M=2..5` 的 `wvSplitK` 从未在 gfx906 跑过、`in_proj_ba` 是 12 block / 60 CU 的占用不足、1700 MHz 交付态复测）、复现命令。
**Step 3 #20 输入**：P0 报告（`2026-09-04-gfx906-w4a16-p0-report.md`）末尾追加「P0.5 已完成 ⇒ 本报告的预测表已被实测替换」，归因文档（`...-p0-attribution.md`）执行记录标 P0.5 完成。#20 的头部空间收敛为 **W4A16 GEMV 19 867 µs/step 对 6 788 µs 地板（33% 效率）＝全 step 的 60.2%**。
**Step 4 记忆**：`gfx906-env-pitfalls` 追加第 9、10 条（`_ON_GFX9` 有意排除 gfx906 ⇒ skinny 快路径默认不可达，先查 arch 谓词别怀疑内核没编；`pkill -f` 的自匹配与「会话级 ssh 管道当日志」两处死角）；`final-goal-shippable-image` 记 P0.5 已进镜像 + 前后 tok/s + Path A base 规则 + `prepare_build_context.sh` 的 entrypoint 流向。
**Step 5 提交**：**降级为暂存** —— 仓库 0 提交、`git config` 无 `user.name`/`user.email`（长期约束：永不改 git config）。`git add` 后 `git diff --cached --stat` = **159 files changed, 28726 insertions**，`git status` 无残留 `MM`；暂存区密钥扫描（api_key/passwd/private key/ghp_/AKIA）仅命中 `p0_preflight.sh` 的提示字符串 `(sudo: needs password)`，无泄漏。剩余 11 个未跟踪文件是历史遗留的 `*.md` 草稿与 `.gitattributes`，不属于本任务。

**仍未闭合的验证缺口**（下次接手先看这三条）：
1. **serve 路径在交付时钟（1700/auto）下未复测** —— QUICKSTART 里的 ~36 tok/s 是按 sclk 比例外推，文档中已明确标注「外推值」。
2. **交付镜像冷启动未走一遍** —— 新 `layers/utils.py` 会使 vLLM 源码哈希键的 AOT 缓存失效（一次性 +387 s），只测量过开发容器内的等价路径。
3. **提交与分发（#18）** 仍未落地，镜像 `1e18c8ea6b8b` 目前只存在于宿主本地。
