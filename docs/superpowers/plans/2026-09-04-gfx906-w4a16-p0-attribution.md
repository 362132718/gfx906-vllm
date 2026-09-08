# #20-P0 gfx906 decode-step 归因与开工门 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一天时间产出"batch1 decode 的 45.5ms 到底花在哪"的可信分解，并据此判定 #20（手写 wave64 W4A16 GEMV）是否开工、以及若开工则 P1 计划必须消费的输入。

**Architecture:** 全部是**测量**，不改任何交付路径。五个工具串成一条链：`BW_eff` 纯读探针给出带宽地板 → 桥接层 shape 日志给出内核真实收到的 (K,N,M) 与生效的 graph 配置 → 端到端 torch.profiler 给出五桶/组件分解与 GPU busy ratio → 7 个真实形状的 M=1 基线给出 ExLlama 距地板的效率 → 汇总成报告并按 spec §1(c) 路由表下结论。

**Tech Stack:** ROCm 7.2.4 / torch 2.13.0+rocm7.2 / triton 3.8.0 / vLLM 0.28.0+rocm724（site-packages 非 editable 安装）；主机 `x@192.168.10.11`（2×gfx906，12 核），容器 `vllm-gfx906`（dev）与 `vllm-gfx906-rc`（一次性，当前 Exited）。

**Spec：** `docs/superpowers/specs/2026-09-03-gfx906-w4a16-wave64-kernel-design.md`（v3）。本计划只覆盖其 §1 的 P0；P1–P3 在 P0 结论之上另写计划（见文末 Handoff）。

---

## 环境事实（已实测，不要重新 discovery）

| 事实 | 值 | 出处 |
|---|---|---|
| vLLM 安装形态 | 非 editable：包在 `/usr/local/lib/python3.12/dist-packages/vllm`，源码树在 `/work/vllm-official`，两者**不是同一份** | 实测 `vllm.__file__` 与 `pip show vllm` |
| 源码树版本控制 | **无 git**（tarball）；`git -C /work/vllm-official` 报 not a repository | 实测 |
| C++ 重建周期 | **分钟级**（实测：`q_gemm.cu` 改动 04:49 → 安装好的 `.so` 04:51）。注意只证明"改一处到 `.so` 落地是分钟级"，**不证明是增量**（单个 `.hip` 全量重编也是分钟级）；`build/temp.linux-x86_64-cpython-312/` 里 CMake 对象与 hipify 产物（`.../gptq/q_gemm.hip.o`）确实还在 | 实测 |
| **stale-tar 陷阱（已上膛）** | `prepare_build_context.sh:73-77`：`dist-packages.tar.gz` 存在且无 `FORCE=1` 时**跳过打 tar**，而末尾 `echo "Build context ready"` 无条件执行 ⇒ 新 `.so` 静默不进交付。主机现状：`/home/x/vllm-work/docker/build-context/dist-packages.tar.gz` = 2026-09-02 16:33 / 7.3GB，容器 `.so` = 09-01 → 今天恰好一致，**一旦重编 `.so` 就变成静默带旧库** | 实测脚本 + ls |
| 新算子登记三处 | `csrc/libtorch_stable/quantization/gptq/q_gemm.cu`（实现）、`csrc/libtorch_stable/ops.h`（声明，容器内镜像见 `container_src/ops_stable.h`）、`csrc/libtorch_stable/torch_bindings.cpp`（`ops.def` + `ops.impl`，镜像 `container_src/torch_bindings.cpp:670-683, 865-867`） | 实测 grep |
| CMake 源列表位置 | `CMakeLists.txt:439` 已含 `"csrc/libtorch_stable/quantization/gptq/q_gemm.cu"` | 实测 grep |
| Path B（交付构建）**不编译 C++** | 只解包 `dist-packages.tar.gz` 再叠 `patches/*.py`；所以新 `.so` 必须**先在 dev 容器编好，再重跑 `prepare_build_context.sh`** 才会进交付 | 读 `docker/Dockerfile.rocm_gfx906:66-100` |
| 主机↔容器同一份文件 | `/home/x/vllm-work` 就是容器里的 `/work`（同一 bind mount，两边各 57 个条目实测对齐）⇒ P0 脚本放 `/home/x/vllm-work/p0/`，容器内直接是 `/work/p0/`，**不需要第二次 docker cp，也不会有两份分叉副本** | 实测 ls |
| 模型路径 | 主机 `/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16`；dev 容器挂载点同名 | 实测 |
| 锁频 | `sudo sh -c 'echo manual > /sys/class/drm/card$N/device/power_dpm_force_performance_level'; sudo sh -c 'echo 8 > .../pp_dpm_sclk'` → 1700MHz；恢复写 `auto`。`pp_od_clk_voltage` 不存在 | 实测读回 |
| 卡间链路 | `rocm-smi --showtopo` = 2 hops / **PCIE**（无 xGMI 桥接器） | 实测 |
| 进程清理 | 必须杀 `vllm serve`、`EngineCore`、**`VLLM::Wor`**（否则 worker 变孤儿占 13.4GB 显存） | 既有 `kill_serve.sh` |
| 主机稳定性 | 会静默死机且无内核日志；长任务必须 `docker exec -d` + 日志文件；黑匣子 `/home/x/blackbox.sh` 应在跑 | 记忆 `gfx906-host-instability` |

**为什么计划里没有 git commit 之外的构建步骤**：本地 `D:\gfx906-vllm` 在 Task 0 才建仓库；主机侧不是仓库，其权威副本就是本地 `container_src/`，同步靠 `sync_to_host.sh` + md5 核对（这是本项目抓出过两次"镜像里是旧文件"的手段，继续沿用）。

---

### Task 0: 版本控制与主机同步工具

**Files:**
- Create: `D:\gfx906-vllm\.gitignore`
- Create: `D:\gfx906-vllm\.gitattributes`（强制 LF：否则 Windows 检出会给 `.sh` 塞 CRLF，scp 到 Linux 上就是 `bash\r: command not found`）
- Create: `D:\gfx906-vllm\p0_sync.sh`
- Create: `D:\gfx906-vllm\p0_clock.sh`（`lock|unlock|status`）

- [ ] **Step 1: 建仓库（只跟踪源与文档，排除日志/模型/构建产物）**

```bash
cd /d/gfx906-vllm && cat > .gitignore <<'EOF'
*.log
__pycache__/
*.pyc
build-context/
dist-packages.tar.gz
blackbox_*.log
EOF
git init -q && git add .gitignore docs container_src docker *.py *.sh 2>/dev/null; git commit -q -m "chore: version-control gfx906 adaptation sources and delivery recipe" && git log --oneline -1
```

Run: `git status --porcelain | wc -l`
Expected: `0`（工作区干净；未跟踪的杂项日志已被 ignore）

- [ ] **Step 2: 写同步脚本（本地权威 → 主机，带 md5 核对）**

`p0_sync.sh`：

```bash
#!/usr/bin/env bash
# p0_sync.sh -- push local authoritative copies to the GPU host and verify by md5.
# Usage: ./p0_sync.sh <relative-path-under-repo> [more...]
set -euo pipefail
HOST=${HOST:-x@192.168.10.11}
DIR=${P0_HOST_DIR:-/home/x/p0}
ssh -o BatchMode=yes "$HOST" "mkdir -p $DIR"
for f in "$@"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
  base=$(basename "$f")
  scp -q -o BatchMode=yes "$f" "$HOST:$DIR/$base"
  L=$(md5sum "$f" | cut -c1-32)
  R=$(ssh -o BatchMode=yes "$HOST" "md5sum $DIR/$base | cut -c1-32")
  [ "$L" = "$R" ] && echo "ok   $base  $L" || { echo "MISMATCH $base local=$L host=$R" >&2; exit 1; }
done
```

- [ ] **Step 3: 写锁频脚本（成对：lock / unlock，锁不到就明确报错）**

`p0_clock.sh`：

```bash
#!/usr/bin/env bash
# p0_clock.sh lock|status  -- discrete DPM pinning (arbitrary OD clocks are
# unavailable on these MI50-flashed-as-VII cards: pp_od_clk_voltage is absent).
set -u
act=${1:-status}
for dev in /sys/class/drm/card0/device /sys/class/drm/card1/device; do
  [ -e "$dev/pp_dpm_sclk" ] || { echo "SKIP $dev (no pp_dpm_sclk)"; continue; }
  cur=$(cat "$dev/pp_dpm_sclk" | tr '\n' ' ')
  if [ "$act" = lock ]; then
    sudo -n sh -c "echo manual > $dev/power_dpm_force_performance_level"
    sudo -n sh -c "echo 8 > $dev/pp_dpm_sclk"
    after=$(cat "$dev/pp_dpm_sclk" | tr '\n' ' ')
    case "$after" in *'8: 1700Mhz  *'*|*'* 8: 1700Mhz'*) echo "LOCKED $dev";;
      *) echo "FAILED to pin $dev: $after" >&2; exit 1;; esac
  else
    echo "$dev level=$(cat $dev/power_dpm_force_performance_level) $cur"
  fi
done
```

- [ ] **Step 4: 在主机上验证锁频脚本可用**

Run: `./p0_sync.sh p0_clock.sh && ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh lock && bash /home/x/vllm-work/p0/p0_clock.sh unlock"`
Expected: 两行 `LOCKED ... -> 1700MHz`，随后两行 `unlocked` 且回到 `level=auto active=860Mhz`。
**已踩过的坑（所以别改回手写恢复）**：`sudo sh -c 'echo auto > .../card'$d'/device/...'` 经 ssh 嵌套引号后 `$d` 被吞成 `card/device`，两卡静默留在 1700MHz。`p0_clock.sh unlock` 把恢复动作收进脚本内部，是唯一安全写法。

- [ ] **Step 5: Commit**

```bash
git add p0_sync.sh p0_clock.sh && git commit -q -m "feat(p0): host sync with md5 verification and discrete DPM clock pinning"
```

---

### Task 1: `BW_eff` 纯流式读探针（带宽地板，标称值不作数）

**Files:**
- Create: `D:\gfx906-vllm\p0_bweff.py`
- Create: `D:\gfx906-vllm\p0_bweff.sh`

- [ ] **Step 1: 写探针（Triton grid-stride 纯读；输出流量可忽略；缓冲远大于 4MB L2）**

`p0_bweff.py`：

```python
# p0_bweff.py -- achievable HBM read bandwidth for gfx906. Nothing else in
# #20 is trustworthy until this number is, so the nominal 1024GB/s is never
# used as the floor. Only 1 fp32 is written per program, so traffic ~= pure read.
import os, statistics, sys
import torch, triton, triton.language as tl

@triton.jit
def read_probe(ptr, out, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + offs, mask=offs < n_elem, other=0.0)
    tl.store(out + pid, tl.sum(x.to(tl.float32)))

def bench(gib: float, iters: int = 20) -> float:
    n = int(gib * 2**30 / 2)                      # fp16 elements
    x = torch.zeros(n, dtype=torch.float16, device="cuda")
    out = torch.zeros(triton.cdiv(n, 8192), dtype=torch.float32, device="cuda")
    grid = (out.shape[0],)
    for _ in range(3):
        read_probe[grid](x, out, n, BLOCK=8192, num_warps=8)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); read_probe[grid](x, out, n, BLOCK=8192, num_warps=8); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    del x, out
    return n * 2 / (statistics.median(ts) / 1e3) / 1e9

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("DEV", "0")))
    rows = [(gib, bench(gib)) for gib in (0.25, 1.0, 4.0)]
    for gib, bw in rows:
        print("working_set=%5.2f GiB  read_bw=%7.1f GB/s" % (gib, bw))
    big = rows[-1][1]
    print("BW_eff = %.0f GB/s" % big)
    if big > 950:  print("SUSPECT: above spec peak -- buffer probably L2-resident", file=sys.stderr); sys.exit(2)
    if big < 600:  print("SUSPECT: far below expectation -- check clocks/thermals", file=sys.stderr); sys.exit(3)
```

- [ ] **Step 2: 写运行脚本（单卡、空闲、锁档）**

`p0_bweff.sh`：

```bash
#!/usr/bin/env bash
# One idle card, clocks pinned, probe three working-set sizes.
set -euo pipefail
P0=/home/x/vllm-work/p0
bash $P0/p0_clock.sh lock
docker exec -e DEV=0 vllm-gfx906 python /work/p0/p0_bweff.py
docker exec -e DEV=1 vllm-gfx906 python /work/p0/p0_bweff.py
bash $P0/p0_clock.sh unlock   # restores auto; never hand-write the sudo restore
```

- [ ] **Step 3: 跑，并把三个工作集大小的读数都记下来**

Run: `./p0_sync.sh p0_bweff.py p0_bweff.sh && ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_bweff.sh" 2>&1 | tee p0_bweff.log`
Expected: 0.25 GiB 明显高于 1/4 GiB（L2 命中），**1.0 与 4.0 GiB 收敛**到同一个数；外部先验区间 **720–780 GB/s**；退出码 0。
If it fails: 偏离先验 → 先怀疑探针（缓冲不够大 / 有并发负载 / 未锁档），不要拿这个数往下算。

- [ ] **Step 4: Commit**

```bash
git add p0_bweff.py p0_bweff.sh p0_bweff.log && git commit -q -m "feat(p0): measured BW_eff probe for gfx906 with working-set sweep"
```

---

### Task 2: 内核真实收到的 (K,N,M) + 生效的 graph 配置

**Files:**
- Modify: `D:\gfx906-vllm\container_src\wna16_exllama.py`（`apply_weights`，紧随 `M = x_2d.shape[0]` 之后，约 :233）
- Create: `D:\gfx906-vllm\p0_shapes.sh`

- [ ] **Step 1: 写失败判据脚本（先定义"什么算采到"）**

`p0_shapes.sh`：

```bash
#!/usr/bin/env bash
# Capture the real (K,N,M) the W4A16 kernel sees, and the graph config in force.
set -euo pipefail
export VLLM_WNA16_EXL_SHAPEDBG=/tmp/wna16_shapes.csv
rm -f /tmp/wna16_shapes.csv
docker exec -e VLLM_WNA16_EXL_SHAPEDBG=$VLLM_WNA16_EXL_SHAPEDBG -d vllm-gfx906 \
  bash -lc 'cd /tmp && MODELS_ROOT=/home/x/models/vllm SERVED_NAME=qwen38 PORT=8191 \
    gfx906-serve /home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16 > /tmp/p0_serve.log 2>&1'
for i in $(seq 1 90); do
  sleep 20
  grep -q "Application startup complete" /tmp/p0_serve.log 2>/dev/null && break
done
curl -s localhost:8191/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen38","prompt":"The capital of France is","max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
bash /home/x/vllm-work/p0/kill_serve.sh
sleep 5
echo "--- distinct (M,K,N) the kernel actually saw:"
docker exec vllm-gfx906 bash -lc 'tail -n +2 /tmp/wna16_shapes.csv | cut -d, -f1-3 | sort | uniq -c | sort -rn'
echo "--- graph config in force:"
docker exec vllm-gfx906 bash -lc 'grep -oE "cudagraph_mode[^,}]*|capture_sizes[^,}]*|CUDA-GRAPH[^ ]*" /tmp/p0_serve.log | sort -u | head'
docker exec vllm-gfx906 bash -lc 'grep -c "Graph capturing finished" /tmp/p0_serve.log'
```

- [ ] **Step 2: 跑，确认它失败（还没有日志）**

Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_shapes.sh"`
Expected: 最后一段输出为空 / `tail: cannot open '/tmp/wna16_shapes.csv'`。这就是本任务的"红"。

- [ ] **Step 3: 加日志（纯 host 侧整数，不碰 CUDA → 捕获期安全）**

在 `container_src/wna16_exllama.py` 顶部 import 之后加：

```python
_SHAPE_LOG = os.environ.get("VLLM_WNA16_EXL_SHAPEDBG")
_shape_seen: list[tuple[int, int, int, str]] = []

def _dump_shapes(*_a):
    if not _SHAPE_LOG:
        return
    import csv
    with open(_SHAPE_LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "k", "n", "cudagraph"])
        w.writerows(_shape_seen)

if _SHAPE_LOG:
    import atexit
    atexit.register(_dump_shapes)
```

在 `apply_weights` 里 `M = x_2d.shape[0]` 之后（`if M <= self._mcap:` 之前）插入：

```python
        if _SHAPE_LOG:
            import torch.cuda as _tc
            _shape_seen.append((
                int(M), int(K_dim), int(N_dim),
                "capture" if _tc.is_current_stream_capturing() else "eager",
            ))
            if len(_shape_seen) % 512 == 0:
                _dump_shapes()
```

部署：`docker cp container_src/wna16_exllama.py vllm-gfx906:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py`

- [ ] **Step 4: 再跑，核对与 spec §2 推导表一致**

Run: `ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_shapes.sh" 2>&1 | tee p0_shapes.log`
Expected 三个硬结论：
1. **decode 期存在 `m=1` 的行**（否则 spec 的 M=1 前提不成立 → 立刻停下报告，#20 形态要重议）；
2. 出现的 `(k,n)` 集合 ⊆ spec §2 那 7 个 per-rank 组合（多出来或少了 = 我们对融合/分片的理解有错，按 spec §3 L1 的互校条款处理）；
3. `cudagraph` 列在 prefill 行是 `eager`、decode 行是 `capture`；`Graph capturing finished` 计数 ≥1。

- [ ] **Step 5: 把实测表写进 spec（不许只用推理）**

在 spec §2 的表下加一行"运行时实测：见 `p0_shapes.log`（YYYY-MM-DD）"，若与推导不符则改表而不是改日志。

- [ ] **Step 6: Commit**

```bash
git add container_src/wna16_exllama.py p0_shapes.sh p0_shapes.log docs/ && git commit -q -m "feat(p0): capture real (K,N,M) and graph mode seen by the W4A16 kernel"
```

---

### Task 3: 五桶/组件时间分解与 GPU busy ratio

**Files:**
- Create: `D:\gfx906-vllm\p0_step.py`（引擎内 harness，控制 profiler 边界）
- Create: `D:\gfx906-vllm\p0_buckets.py`（把 profiler 表分类成桶并出报告）
- Create: `D:\gfx906-vllm\p0_step.sh`

- [ ] **Step 1: 写 in-process harness：同一段 prompt 跑 max_tokens=1 与 65，差值 /64 = 每 decode step 的 wall**

`p0_step.py`：

```python
# p0_step.py -- one decode step, five buckets. Uses the same engine flags as
# gfx906-serve so attribution transfers to the delivery path.
import json, os, time
import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16")
PROMPT = "Explain in detail why the sky is blue and what determines its shade at dawn. " * 8
N = 256          # decode-dominant: prefill (~0.3s) is ~2% of a 256-step request

def gen(llm, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    return time.perf_counter() - t0, out[0]

llm = LLM(
    model=MODEL, dtype="float16", tensor_parallel_size=2,
    gpu_memory_utilization=0.90, max_model_len=16384, max_num_seqs=48,
    attention_backend="TRITON_ATTN", disable_custom_all_reduce=True,
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48]},
    limit_mm_per_prompt={"image": 1, "video": 1},
    mm_encoder_attn_backend="TRITON_ATTN",
)
prefill = min(gen(llm, 1)[0] for _ in range(3))
req = min(gen(llm, N)[0] for _ in range(3))
per_step = (req - prefill) / (N - 1)
print("prefill wall = %.1f ms ; request(%d tok) wall = %.1f ms" % (prefill*1e3, N, req*1e3))
print("decode wall/step = %.2f ms  (=> %.1f tok/s)" % (per_step*1e3, 1/per_step))
json.dump({"prefill_ms": prefill*1e3, "req_ms": req*1e3, "per_step_ms": per_step*1e3,
           "n_tok": N}, open("/tmp/p0_wall.json", "w"))

from torch.profiler import profile, ProfilerActivity
REPS = 3
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(REPS):
        gen(llm, N)
    torch.cuda.synchronize()
p.export_chrome_trace("/tmp/p0_trace.json")
rows = [(e.key, e.self_device_time_total, e.count, e.self_cpu_time_total)
        for e in p.key_averages()]
json.dump({"reps": REPS, "rows": rows}, open("/tmp/p0_kernels.json", "w"))
print("profiled %d requests; kernels with device time: %d"
      % (REPS, sum(1 for r in rows if r[1] > 0)))
```

- [ ] **Step 2: 跑一次，把 top-40 内核名导出来（分类器的输入就是这份名单，先看清再写正则）**

Run:
```bash
./p0_sync.sh p0_step.py && ssh x@192.168.10.11 "docker exec -d vllm-gfx906 bash -c 'cd /tmp && python /work/p0/p0_step.py > /tmp/p0_step.log 2>&1'"
# 轮询到结束（27B TP2 起服务要 7-12 分钟，必须 detached）
ssh x@192.168.10.11 "tail -5 /tmp/p0_step.log"
ssh x@192.168.10.11 "docker exec vllm-gfx906 python -c \"import json;d=json.load(open('/tmp/p0_kernels.json'));r=d['reps'];[print('%9.1fus %7d  %s'%(a/r,c,k[:80])) for k,a,c,b in sorted(d['rows'],key=lambda x:-x[1])[:40]]\""
```
Expected: 日志末行出现 `decode wall/step = … ms`，且 40 行里有可辨认的 Tensile `Cijk_*`、Triton `triton_poi/RED/fp16_dot`、`gemv`/`gptq`、`rccl_*`、`allreduce` 类名字。
**把这 40 行贴进 `p0_buckets.py` 的注释里**，Step 3 的正则必须逐条覆盖它们 —— 不允许出现"其余未知"占大头。

- [ ] **Step 3: 写分类器（每个桶一条正则；未分类必须显式列出并占比 <5%）**

`p0_buckets.py`：

```python
# p0_buckets.py -- turn /tmp/p0_kernels.json into the spec section 1(b) table.
import json, re, sys

BUCKETS = [
    ("w4a16_gemv",  r"gptq|gemv|exllama|dequant|WNA16|w4a16|awq"),
    ("attention",   r"triton_(?:decode|fwd|attn)|flash|attention"),
    ("gdn",         r"gdn|fla|chunk|gate_delta|conv1d|scan"),
    ("lm_head",     r"Cijk.*248320|logits|vocab_parallel"),   # bf16 2.54GB matvec
    ("rccl",        r"rccl|nccl|allreduce|all_reduce"),
    ("dense_gemm",  r"Cijk|tensile|hipblaslt|sgemm|hgemm"),
    ("elementwise", r"elementwise|vectorized_|copy|CatArrayBatched|norm|silu|rope|softmax|index"),
]

d = json.load(open("/tmp/p0_kernels.json"))
reps, rows = d["reps"], d["rows"]
w = json.load(open("/tmp/p0_wall.json"))
req_ms, prefill_ms, n_tok = w["req_ms"], w["prefill_ms"], w["n_tok"]
dec_ms = req_ms - prefill_ms                       # wall for the n_tok decode steps

out, un = {}, []
for key, dev_us, cnt, cpu_us in rows:
    if dev_us <= 0:
        continue
    for name, pat in BUCKETS:
        if re.search(pat, key, re.I):
            out[name] = out.get(name, 0) + dev_us
            break
    else:
        un.append((key, dev_us))
tot = sum(out.values())
busy_req = tot / reps / 1e3
print("wall: request %.1f ms (prefill %.1f + decode %.1f over %d steps)"
      % (req_ms, prefill_ms, dec_ms, n_tok))
print("%-12s %10s %9s %9s %8s" % ("bucket", "ms/req", "%busy", "us/step", "%dec*"))
for name, us in sorted(out.items(), key=lambda kv: -kv[1]):
    ms_req = us / reps / 1e3
    # *%dec uses the whole request's bucket time over the decode window; prefill
    # kernels (~2% of the request) bias it upward. Fine for a routing decision.
    print("%-12s %10.2f %8.1f%% %9.1f %7.1f%%"
          % (name, ms_req, us / tot * 100, ms_req * 1e3 / n_tok, ms_req / dec_ms * 100))
un_us = sum(u for _, u in un)
print("%-12s %10.2f %8.1f%%" % ("UNCLASSIFIED", un_us/reps/1e3, un_us / max(tot, 1) * 100))
for k, v in sorted(un, key=lambda x: -x[1])[:10]:
    print("    %.1fus/req  %s" % (v / reps, k[:90]))
print("GPU busy = %.2f ms/req ; busy ratio (vs request wall) = %.2f" % (busy_req, busy_req / req_ms))
print("busy ratio (vs decode wall only) = %.2f" % (busy_req / dec_ms))
gemv = out.get("w4a16_gemv", 0) / reps / 1e3
print("W4A16 GEMV = %.2f ms/req = %.1f us/step = %.1f%% of decode wall"
      % (gemv, gemv * 1e3 / n_tok, gemv / dec_ms * 100))
if un_us / max(tot, 1) > 0.05:
    print("STOP: >5%% unclassified -- extend BUCKETS before trusting this table", file=sys.stderr)
    sys.exit(1)
```

- [ ] **Step 4: 跑出五桶表并检查未分类占比**

Run: `./p0_sync.sh p0_buckets.py && ssh x@192.168.10.11 "docker exec vllm-gfx906 python /work/p0/p0_buckets.py" 2>&1 | tee p0_buckets.log`
Expected: 表格 + `busy ratio = X` + `W4A16 GEMV share`；未分类 <5%，否则脚本以 1 退出，回到 Step 2 的名单补正则（不许放宽阈值）。

- [ ] **Step 5: 按 spec §1(c) 下路由结论并记进报告**

`busy ratio < 0.6` → 结论"不开 #20"，Task 5 出报告后计划到此为止；
`0.6–0.8` → 结论"先吃掉非内核那块"；
`≥ 0.8` → 继续 Task 4 拿逐形状效率，P1 才有输入。

- [ ] **Step 6: Commit**

```bash
git add p0_step.py p0_buckets.py p0_buckets.log && git commit -q -m "feat(p0): five-bucket decode-step decomposition with busy ratio"
```

---

### Task 4: 7 个真实形状的 M=1 基线（eager/graph 双测 + 距地板效率 + launch 代价）

**Files:**
- Create: `D:\gfx906-vllm\p0_shapes_baseline.py`
- Create: `D:\gfx906-vllm\p0_launchcost.py`

- [ ] **Step 1: 写逐形状基线（直接打 `ops.gptq_gemm_out`，形状用 Task 2 实测值）**

`p0_shapes_baseline.py`：

```python
# p0_shapes_baseline.py -- ExLlama fused kernel at M=1 for the real per-rank
# shapes, timed in eager AND inside a captured graph, compared against the
# measured BW_eff floor (bytes the kernel actually pulls, not K*N/2).
import json, os, statistics, sys
import torch
from vllm import _custom_ops as ops

BW_EFF = float(os.environ.get("BW_EFF", "0"))      # GB/s from Task 1; refuse if unset
GROUP, BITS = 128, 4
# (K, N) per-rank from spec section 2 / Task 2 measurement
SHAPES = [(5120, 7168), (3072, 5120), (5120, 17408), (8704, 5120),
          (5120, 5120), (5120, 3072), (3072, 5120)]

def make(K, N, dev):
    return dict(
        x=torch.randn(1, K, dtype=torch.float16, device=dev),
        wq=torch.randint(-2**30, 2**30, (K // 8, N), dtype=torch.int32, device=dev),
        wz=torch.randint(0, 2**30, (K // GROUP, N // 8), dtype=torch.int32, device=dev),
        ws=torch.randn(K // GROUP, N, dtype=torch.float16, device=dev),
        gi=torch.empty(0, dtype=torch.int32, device=dev),
        c=torch.empty(1, N, dtype=torch.float16, device=dev),
        td=torch.empty(0, dtype=torch.float16, device=dev),
    )

def call(t):
    ops.gptq_gemm_out(t["c"], t["x"], t["wq"], t["wz"], t["ws"], t["gi"],
                      True, True, BITS, t["td"])

def timed(fn, iters=60):
    for _ in range(16):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

rows = []
dev = "cuda:0"
for K, N in SHAPES:
    t = make(K, N, dev)
    eager = timed(lambda: call(t))
    g = torch.cuda.CUDAGraph()
    call(t)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(20):
            call(t)
    graph = timed(lambda: g.replay(), iters=20) / 20
    cbytes = K * N / 2 + K * N / 2 * (1/32 + 1/128)
    floor = cbytes / (BW_EFF * 1e9) * 1e3
    rows.append(dict(K=K, N=N, eager_us=eager*1e3, graph_us=graph*1e3,
                     bytes=cbytes, floor_us=floor*1e3,
                     eff_pct=floor/graph*100, waves=N/64))
    del t, g
    print("K=%5d N=%5d eager=%7.1fus graph=%7.1fus floor=%6.1fus -> %.0f%% of BW_eff"
          % (K, N, eager*1e3, graph*1e3, floor*1e3, floor/graph*100))
json.dump(rows, open("/tmp/p0_shapes_baseline.json", "w"), indent=1)
lo = min(rows, key=lambda r: r["eff_pct"]); hi = max(rows, key=lambda r: r["eff_pct"])
print("worst %.0f%% (%d,%d) best %.0f%% (%d,%d)" %
      (lo["eff_pct"], lo["K"], lo["N"], hi["eff_pct"], hi["K"], hi["N"]))
```

Run: `./p0_sync.sh p0_shapes_baseline.py && ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh lock && docker exec -e BW_EFF=<Task1的数> vllm-gfx906 python /work/p0/p0_shapes_baseline.py" 2>&1 | tee p0_shapes_baseline.log`
Expected: 7 行 + worst/best；`BW_EFF` 未设时为 0 会让百分比炸掉 —— 所以必须先有 Task 1 的数（这是刻意的依赖）。

- [ ] **Step 2: 写 launch 代价探针（决定 Split-K 两阶段 vs persistent）**

`p0_launchcost.py`：

```python
# p0_launchcost.py -- per-launch cost inside a replayed graph vs eager. Six of
# seven shapes need Split-K, so ~240 extra reduce launches/token is the default
# path; this is the number that decides whether that is affordable.
import statistics, torch
def timeit(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s,e=torch.cuda.Event(True),torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
one = torch.zeros(1, device="cuda")
EAGER = timeit(lambda: one.zero_())
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(240): one.zero_()
GRAPH = timeit(lambda: g.replay()) / 240
print("eager  per-launch  %.2f us  -> 240/token = %.2f ms" % (EAGER, EAGER*240/1e3))
print("graph  per-launch  %.3f us -> 240/token = %.3f ms" % (GRAPH, GRAPH*240/1e3))
print("verdict:", "two-phase reduce is affordable" if GRAPH*240/1e3 < 0.5
      else "persistent/last-block reduction is required for Split-K")
```

Run: `./p0_sync.sh p0_launchcost.py && ssh x@192.168.10.11 "docker exec vllm-gfx906 python /work/p0/p0_launchcost.py" 2>&1 | tee p0_launchcost.log`
Expected: 两行数字 + 一行 verdict。该 verdict 直接写入 spec §2 被否清单最后一条（把它从"待测"改成实测结论）。

- [ ] **Step 3: 算"完美 GEMV 能换来多少 tok/s"，与 §1(c) 的 1.15× 门对照**

手工一步即可（数据都已在两个 json 里）：
`预期 e2e 增益 = 1 / (1 - gemv占比 × (1 - 1/(1/eff_pct)))`，其中 gemv 占比来自 Task 3、`eff_pct` 来自 Task 4 的加权（按各形状在 304 次里的出现次数加权，不是按形状数）。把算式与结果写进 `p0_report.md`。

- [ ] **Step 4: Commit**

```bash
git add p0_shapes_baseline.py p0_launchcost.py p0_shapes_baseline.log p0_launchcost.log && git commit -q -m "feat(p0): per-shape M=1 baseline vs measured floor and in-graph launch cost"
```

---

### Task 5: P0 报告与门决策（交付物就是这一份）

**Files:**
- Create: `D:\gfx906-vllm\docs\superpowers\plans\2026-09-04-gfx906-w4a16-p0-report.md`

- [ ] **Step 1: 按固定骨架填报告（结论必须在第一段，不能埋在末尾）**

必须包含的六项，每项都是实测数字 + 指向产生它的 log 文件：
1. `BW_eff` = ___ GB/s（三工作集读数）
2. 真实 (K,N,M) 表 + 生效 `cudagraph_mode`/capture sizes + `capture`/`eager` 列
3. 五桶/组件表（含 `lm_head` 与 `rccl` 两行）+ **GPU busy ratio = ___** → spec §1(c) 的哪一行
4. 逐形状基线：ExLlama 达 `BW_eff` 的 ___%（最坏/最好形状）
5. graph 内 per-launch ___µs → 240 次/token = ___ms → Split-K 归约选型结论
6. **决策**：开工 / 转 #21b(GDN) / 转配置与图覆盖 / 转 lm_head —— 一句话，附预期 e2e 增益的数字

- [ ] **Step 2: 若结论是"开工"，把 P1 计划的输入清单写全**

清单固定为：白名单候选形状（含 splitK 档与 eager/graph 差值）、`BW_eff`、VALU 预算（4.36）、epilogue 形式 A、L0 单元测试需求、L2 harness 需求。P1 计划另起文件，不在本计划内展开。

- [ ] **Step 3: 更新记忆（P0 数字属于会变的项目状态，不属于代码可推导信息）**

在 `final-goal-shippable-image.md` 加一条：#20-P0 结论 + busy ratio + GEMV 占比 + 决策，日期 2026-09-04。

- [ ] **Step 4: Commit + 环境复位核对**

```bash
git add docs/superpowers/plans/2026-09-04-gfx906-w4a16-p0-report.md && git commit -q -m "docs(p0): decode-step attribution report and gate decision"
ssh x@192.168.10.11 "bash /home/x/vllm-work/p0/p0_clock.sh status; docker exec vllm-gfx906 bash -lc 'nvidia-smi 2>/dev/null || rocm-smi --showmeminfo vram | head -8'"
```
Expected: 两卡 `level=auto`；两卡显存占用回落到 ~60MB/20MB（无残留 worker）。

---

## Self-Review（写完后自查，已执行）

**1. Spec 覆盖**：本计划有意只覆盖 spec 的 P0（§1 的 (a)(b)(c)(d) 四个产物 + §2 的两个待测分母：VALU 预算里的 launch 代价、Split-K 归约选型）。§3 的 L0/L1/L2/L3、§4 的交付配方改动属 P1–P3，其任务必须等 P0 数字才能不带占位符地写 —— 现在写会违反"No Placeholders"。spec 里被本计划消费的条款：§1(b) 五桶、§1(c) 路由表、§1(d) 真实 M、§2 形状表与 544 launch、§6 的 `BW_eff` 先验区间。**无遗漏项被静默丢掉**，未覆盖项全部登记在 Step 2 的 P1 输入清单里。

**2. 占位符扫描**：Task 4 Step 1 的命令里 `-e BW_EFF=<Task1的数>` 是刻意的跨任务数据依赖（不是 TBD），执行者从 `p0_bweff.log` 取值；除此之外无 TBD/TODO/"类似 Task N"。

**3. 类型一致性**：`/tmp/p0_wall.json` 的键 `per_step_ms` 被 `p0_buckets.py` 读取（同名）；`/tmp/p0_kernels.json` 是 `(key, self_device_time_total, count, self_cpu_time_total)` 四元组列表，`p0_buckets.py` 按同序解包；`/tmp/p0_shapes_baseline.json` 字段名与报告第 4 项一致；`p0_clock.sh lock|status` 两种动作在各脚本里用法一致。

---

## Handoff（本计划跑完后才做的两件事）

1. **P1–P3 另起计划文件** `2026-09-0?-gfx906-w4a16-wave64-kernel.md`，输入 = P0 报告第六项 + spec §2/§3。若 P0 判"不开工"，改为给路由到的那个任务写计划。
2. **交付路径（P1 动手前先落地这 4 条护栏，它们与本计划的测量无关但会决定 #20 成果能不能进镜像）**

   **迭代只走 Path A，Path B 只在最终出交付镜像时跑一次。** Path A = 在 dev 容器里编译 → `cp` 进 site-packages → `docker commit --no-pause`（**不带 `--no-pause` 会在本机挂死**，见 `Dockerfile:52-53` 与记忆 `docker-commit-nopause-kfd`）；Path B 从不编译 C++，只解 `dist-packages.tar.gz`，所以它唯一的输入新鲜度问题就是那个 tar。

   - **护栏 1（改脚本，治 stale-tar）**：`prepare_build_context.sh` 的 [7/7] 跳过条件从"文件存在"改成"tar 比容器内的 `_C_stable_libtorch.abi3.so` 新"，且**跳过时不再打印 `Build context ready`**（改为打印 `STALE: rerun with FORCE=1`）：

     ```bash
     # freshness is decided against the extension the tar must carry, not against
     # the tar's mere existence.
     STALE=$(docker exec "$CONTAINER" python3 -c "
     import os
     tar = '$CTX/dist-packages.tar.gz'
     so  = '$SP/vllm/_C_stable_libtorch.abi3.so'
     print(1 if not os.path.exists(tar) or os.path.getmtime(tar) < os.path.getmtime(so) else 0)")
     if [ "$STALE" = "0" ] && [ "${FORCE:-0}" != "1" ]; then
       echo "   up to date ($(du -h "$CTX/dist-packages.tar.gz" | cut -f1)); skipping"
     else
       [ "$STALE" = "1" ] && echo "   STALE vs container .so -> re-tarring"
       docker exec "$CONTAINER" tar -czf /tmp/dist-packages.tar.gz \
           -C / usr/local/lib/python3.12/dist-packages usr/local/bin
       docker cp "$CONTAINER:/tmp/dist-packages.tar.gz" "$CTX/dist-packages.tar.gz"
       docker exec "$CONTAINER" rm -f /tmp/dist-packages.tar.gz
       echo "   $(du -h "$CTX/dist-packages.tar.gz" | cut -f1)"
     fi
     ```
     并把末尾的 `echo "Build context ready at: $CTX"` 改成只在 `$STALE = 0` 或刚重打过时打印。
   - **护栏 2（补 final stage）**：把 `rebuild` 里那条 `RUN test -z "$(ls -A /root/.cache ...)"` 泄漏检查**照抄到 `final`**。#20 会引入第二个 W4A16 内核模式（ExLlama vs wave64），镜像若带着 ExLlama 模式下建的缓存，用户切开关首次启动即崩，而 spec §3 的双向测试管不到镜像里预置的文件。
   - **护栏 3（新符号断言，与内核同批加，两个 stage 一起）**：**不建议现在只加 Path B 那一行** —— 配方是交付物，别人会照它重放；在内核存在之前把 `rebuild` 弄红几周，代价大于收益。等第一个带 `gfx906_w4a16_gemv_out` 的 `.so` 就位时，`Dockerfile:120` 与 `:172` **同一提交各加一行**：
     ```dockerfile
     && grep -aq gfx906_w4a16_gemv_out ${SP}/vllm/_C_stable_libtorch.abi3.so \
     ```
     stale-tar 那个静默失败由护栏 1 拦，不靠这条断言。
   - **护栏 4（文档）**：QUICKSTART/BUILD_GUIDE 明写"B 不编译 C++；改了 `.cu/.hip` 必须走 A 或用 `FORCE=1` 重打 tar"。

---

## 执行记录（P0，2026-09-04 → 09-08）

**Task 0 完成**（除提交外全部验证）：分支 `p0-attribution`、`.gitattributes`（强制 LF，防 `.sh` 被 scp 到 Linux 后变 `bash\r`）、`p0_sync.sh`（本地=主机=容器三级 md5）、`p0_clock.sh lock|unlock|status`（实测锁到 1700MHz 再解回 `auto`）。

**Task 1 完成**：`BW_eff` = **928 GB/s**（手写 Triton 纯读，两卡、1/4 GiB 收敛）/ **821 GB/s**（`torch.sum` 独立实现，copy/read 时间比 2.09）→ **取 821 作地板**。外部 clpeak 先验 720–780 证伪。证据 `p0_bweff.log`。

**Task 2 完成**（三轮才对，见缺陷 1/2）：证据 `p0_shapes.log`。
- 真实形状集 = **5 个**、每 token **256 次 GEMV**、需 Split-K 的 **192 次**；GDN `in_proj_qkvz` 融合后 N=**8192**；attn `o_proj` 与 GDN `out_proj` 共用 (3072,5120)。
- allreduce = **128 次/token**（o_proj 16 + down 64 + GDN out_proj 48）。
- **M=1 已实测确认**（eager 普查：7 步 × 256，逐形状计数精确）；同一请求还出现 m=14 / 48 / 2048 ⇒ **dispatch 必须严格 `M == 1`**。

**Task 3 / 4 / 5 完成（2026-09-08）**：见 `2026-09-04-gfx906-w4a16-p0-report.md`（结论、六项、日志索引都在那一份里）。

### 我自己脚本里的五个缺陷（都已修，需防复发）
1. `apply_weights` 是 **dynamo 追踪区域**：在里面调 `torch.cuda.is_current_stream_capturing()`（返回非 Tensor）会让引擎初始化失败，报错还指向 GDN 模块。已记进记忆 `gfx906-env-pitfalls` 坑 5。修法：热路径只做纯 Python 整数 append。
2. **只靠 `atexit` 落盘不够**：vLLM 硬杀 TP worker，结果只有 APIServer 那份空 CSV。修法：关停前给每个 vLLM pid 发 `SIGUSR1`（handler 在编译区外，安全），文件名带 pid 以免两 rank 互相覆盖。
3. **探针脚本必须有 `if __name__ == "__main__"` 守卫**：vLLM 用 spawn 起 worker，无守卫时子进程重新导入主模块再次进引擎，报 "current process has finished its bootstrapping phase"。本计划 Task 4 的 `p0_shapes_baseline.py` / `p0_launchcost.py` 同样适用。
4. **前端进程里的 `torch.profiler` 在 TP 下看不见内核**：profile 上下文跑在主进程，内核却在 spawn 出来的 worker 里执行，于是导出的表只有 `hipDeviceSynchronize` 一行、`kernels with device time: 0` —— 一张"看起来像测量结果"的空表。修法：改用 vLLM 自带的 per-worker profiler（`LLM(profiler_config={"profiler":"torch","torch_profiler_dir":...,"delay_iterations":...,"max_iterations":...})` + `llm.start_profile()/stop_profile()`），trace 由新增的 `p0_parse_trace.py` 离线聚合成 `p0_kernels.json`（桶正则因此可以反复调，不必每调一次就再花 10 分钟起引擎）。附带好处：`delay/max_iterations` 把窗口关在纯 decode 步内，桶表不再有 prefill 污染，分母就是被测的那个 decode step。
5. **`p0_clock.sh` 只钉了 sclk，把 mclk 悄悄留在了 DPM level 0 = 350MHz**。为躲热炸把 sclk 锁到 level 4 之后，`pp_dpm_mclk` 停在 `0: 350Mhz *`，HBM 理论带宽 1024 → 358 GB/s，于是**所有访存受限的微基准一律 3-4× 偏慢，且没有任何报错**（当晚 dense 基准读出 ~250 GB/s 才回头查出）。已记进记忆 `gfx906-env-pitfalls` 坑 8。修法：`p0_clock.sh lock [sclk_lvl] [mclk_lvl]` 两个都钉（mclk 默认 2=1000MHz），`status` 同时打印两个时钟。
   **已录 trace 不受影响**，三条证据：(a) 27B 的 wall step 53.68 ms 恰等于交付 45.5 ms × 1700/1425，只按 sclk 缩放；(b) 钉回 mclk=1000 后 dense 两项读数与 350 时逐位相同（318.1↔320.4、6661.8↔6617.9）——它们本就与 mclk 无关；(c) 决定性对照：同批 GEMV 形状在 350 下 33824 µs、在 1000 下 20603 µs，而 trace 是 19992 µs ⇒ 长跑期间 SMU 仍按负载 ramp 了 mclk，只有短脉冲基准拿不到 boost。
   **推论（防复发）**：出任何带宽受限的数之前先 `p0_clock.sh status` 看 mclk；而 latency-bound 项（如 M=1 的 Tensile 小 GEMM）**不能用来判断时钟状态**——"换时钟读数不变"是它不受时钟影响的证据，不是"时钟正常"的证据。

**Task 3 附带确认**：worker 落盘的 `p0_step_shapes.csv.<pid>` 两 rank 各 **256 行、全部 M=130**（= 那条 130-token prompt 的一次 prefill）。这不是"只跑了一步"，而是 **dynamo 把 `apply_weights` 的 Python 副作用烤进了编译产物**：同形状的后续 prefill 与全部 decode 都不再执行 Python。所以 Python 侧探针只能看见每种形状首次被 trace 的那一次 —— 同时反向确认了"256 次 GEMV/step/rank、5 个形状"与 eager 普查逐形状计数一致。

### 计划文本要以实测为准的地方
- Task 4 Step 1 的 `SHAPES` = spec §2 的 **5 个** per-rank 组合（不是 7 个）；加权分母是 **256** 次/token（不是 304）。
- Task 3 的 RCCL 桶预期按 128 次/token 重估（~1.4%，未捕获时 ~3%）。
- **产物路径从 `/tmp` 改到 `/work/p0`**（同一 bind mount，主机看得见、容器重启不丢）：`p0_wall.json` / `p0_window.json` / `p0_kernels.json`（schema `p0-trace-v1`，由 `p0_parse_trace.py` 从 `/work/p0/prof/*.json.gz` 聚合而来）均在 `/work/p0/`，**`p0_buckets.py` 按这个路径读**；前端 `p0_trace.json` 已随缺陷 4 一起删掉。

### 挂起
- **提交**：等 git 身份（不阻塞测量，文件都在盘上）。
- **Path B 符号断言时机**：我建议与内核同批加（理由见护栏 3），外部核验主张现在就加。改成现在加只是 `Dockerfile:120` 一行，代价是重建路径在 #20 完成前对所有人是红的。

### P0.5（本报告派生的前置任务）— 已完成 2026-09-08

`dense_fp16` 桶 22 207 µs/step（41.4%，`in_proj_ba` 48×318 + `lm_head` 6 982）已由一处
门谓词换到 ROCm skinny `LLGemm1_kernel`：step 53.663 → **33.008 ms**（18.6 → **30.3 tok/s**），
256/256 贪心 token 逐字一致，八道门 G1–G8 全过，已进交付镜像（`gfx906-selfcheck` 20 ok / 0 FAIL）。
计划与执行记录：`2026-09-08-gfx906-p05-dense-gemv-skinny.md`；结论报告：`2026-09-08-gfx906-p05-report.md`。
本归因表现在唯一的大项是 `w4a16_gemv` 19 867 µs/step = 60.2% ⇒ #20 的输入见 P0 报告末尾"更新"一节。
