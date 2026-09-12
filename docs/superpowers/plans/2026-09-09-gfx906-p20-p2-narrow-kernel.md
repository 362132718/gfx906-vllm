# #20-P2「窄 P2：8192/7168 两族 V3 内核 + dispatch 白名单进镜像」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 P1 胜出的 V3 结构（逐元素 `v_pk_sub_f16` 减带偏 zp）做成生产内核，按形状白名单只服务 (5120,8192) qkvz 与 (5120,7168) qkv 两族 M=1 decode，过 L2/L3 验收并进交付镜像两路径。收益口径（精确式 `r·p/(1−p)`，分母 = 交付态 32.4 tok/s / 30.86 ms）：现状 V3 直接分派 = 702.7 µs ⇒ **+0.76 tok/s（≈33.16）**；G-P0 要求的 1.3× 线 = 累计 **1 148.7 µs（3.72%）⇒ +1.25 tok/s（≈33.653）**；探针上限 2757 µs ⇒ +3.18。

**Architecture:** 生产内核是**无 torch 依赖的纯 HIP 共享库**（`extern "C"` + ctypes 调用，传 `data_ptr()` 与 torch 当前流——绕开 torch C++ ABI 耦合，图捕获安全），由 Dockerfile 在镜像内用 hipcc 编译；dispatch 走 P0.5 验证过的 custom-op 模式（`direct_register_custom_op`，对 dynamo/图捕获不透明），插在 `wna16_exllama.py:apply_weights` 的 fused 分支前，谓词 = `M==1 && (K,N)∈白名单 && bits==4 && 无 g_idx && env 未关`。out 写入既有 `_cbuf[:1]` persistent slab（覆写语义，无需 zero_，无 atomicAdd ⇒ 确定）；split-K partial 按层预分配（同 established 模式）。

**Tech Stack:** hipcc（镜像内已有，P1 Task 0 实证）、ctypes、`direct_register_custom_op`（`vllm.utils`）、既有 P0.5 A/B 链路（`p0_step.py`/`p0_clock.sh`/`p05_install.sh`）、P1 harness（`p1_l1_bench.py` 的 wiring/numeric 门）。

---

## 前置事实（全部 P1 实测或本仓原文，执行前不必重验）

| 事实 | 出处 |
|---|---|
| 位规则：code i 在位 `(i//2)*4 + (i%2)*16`；qweight [K/8,N]、scales [K/128,N] fp16、qzeros [K/128,N/8] int32 GPTQv2、g_idx 空 | `docs/.../2026-09-08-p1-nibble-map.md`（L0 逐字+one-hot 钉死） |
| V3 数值：对 fp64 参考 RMS **0.024–0.027 ULP**（MAX 1），过 L2-c 0.5 门 18× 余量；**incumbent 自己 RMS 69–219 ULP** ⇒ L2 门按产线基线重标定 | P1 报告 v3「#1 ULP 拆分」 |
| 扫参最优：8192 ⇒ sk8/u8 **67.34 µs**；7168 ⇒ sk8/u8 **65.17 µs**（graph 内、1700/1000） | `logs/p1/p1_l1_rows.json` |
| incumbent-rot（双缓冲、同尺）：8192 **78.99 µs**、7168 **74.14 µs** ⇒ 1.3× 目标线 **≤60.8 / ≤57.0 µs** | `logs/p1/p1_gap_probe.json` |
| 探针上限（mlp/x4 最优）：8192 **35.1 µs**、7168 **33.5 µs** | 同上 |
| 分派收益：现状 V3 直接分派 **702.7 µs/step ⇒ +0.76 tok/s（33.16）**；1.3× 累计 **1 148.7 µs（= 48×18.23 + 16×17.11）⇒ +1.25（33.653）**；上限 2 757（+3.18）。tok/s 一律用精确式 `32.4/(1−µs/30860)`，不用线性近似 | P1 报告 v3 收益表数字 + 本计划复核（P2 评审 P0-5/P0-6） |
| `apply_weights` 插入点：`container_src/wna16_exllama.py:277-301`，`c_out = self._cbuf[:M]`、`c_out.zero_()` + `ops.gptq_gemm_out(...)`、`output = c_out` → `output.reshape(out_shape)` | 本仓原文（P2 计划 Task 2 引用） |
| `_cbuf` 在 `process_weights_after_loading` 分配（`_mcap×N_dim`），capture-safe 固定地址；`_tdq_empty` 同 | 同文件 `:229-237` |
| apply_weights 在 dynamo 追踪区内：**只能纯 python int/list 操作 + torch 调用**；`torch.* 返回非 Tensor` 会杀引擎初始化 | [[gfx906-env-pitfalls]] 第 5 条 + 同文件 `:271-275` 注释 |
| `direct_register_custom_op` 模式（mutates_args、fake impl）是 P0.5 验证过的"被追踪区内的安全出口" | `vllm/utils`（rocm_unquantized_gemm 用同一模式） |
| B3 slab 模式：覆写语义不需要 zero_；atomicAdd 才需要 | [[exllama-fused-kernel-atomic-add]] |
| 镜像内 hipcc 可用（`/usr/bin/hipcc`），JIT 通道 P1 已证明能编 gfx906 | P1 Task 0 执行记录 |
| 镜像配方：Path B（`--target rebuild`，COPY patches/ + dist-packages.tar.gz）先、Path A（FROM :rebuilt）后；`prepare_build_context.sh` 第 3 步从容器抽 `$SP` python 文件 ⇒ **改 python 补丁必须先 `docker cp` 进容器**；`.so` 由 Dockerfile RUN hipcc 现场编译（源 = patches/ 里的 .hip）⇒ 无需改 prepare 脚本 | [[final-goal-shippable-image]]、Dockerfile.rocm_gfx906 |
| 引擎 A/B 链路：`p0_step.py`（OUT/EXPECTED_MS 参数化、greedy token 落盘）+ `p0_clock.sh lock 4 2`（1425/1000，长作业安全档）| P0.5 G5/G6 执行记录 |
| 交付态基线：**32.4 tok/s（step 30.86 ms）**；P0.5 钉频基线 **33.008 ms/step = 18.6→30.3** | P0.5 报告 v3 |

## 硬约束（沿用，覆盖每一步）

1. 中文回复；宿主由我 ssh 驱动，>2 分钟 GPU 作业一律 `nohup setsid … > /abs.log 2>&1 < /dev/null &`。
2. 往容器喂脚本一律 `docker exec -i`；kill 模式带括号防自匹配。
3. **永不改 git config**；commit 用行内 `git -c user.name=362132718 -c user.email=362132718@users.noreply.github.com`；**只有用户明确要求才 commit/push**。
4. 改 `docker/entrypoint/*` 或 python 补丁后进镜像的正确顺序：仓库 → `docker cp` 进容器 → `prepare_build_context.sh` → Path B → Path A → 镜像内 md5/存在性复验 → 真机 selfcheck。
5. GPU 长作业分离跑；`p0_clock.sh lock 4 2` 用于 A/B（安全档），测完 unlock。
6. **P2 的内核只服务 M==1 且白名单两族**；M>1 与其余三族必须原路走 `gptq_gemm_out`——谓词错误 = 全模型输出错误，Task 2 的路由门必须覆盖。
7. 判据写死不许事后调：perf 门 = graph 内 ≤ **60.8 µs（8192）/ 57.0 µs（7168）**（1.3× 线，incumbent-rot 分母）。L2 门**两条同时满足**：① 回归门——对 fp64 参考 RMS ULP **≤0.05**（P1 实测 0.024–0.027，容忍编译差异 2×）；② 产线比较门——**≤ incumbent 基线（69–219）的 1/10（≤6.9）**。门意在"不劣于 P1 原型、远好于产线"，不允许只用其中一条。L3 = greedy 256 token 与 incumbent 的分歧必须能被 teacher-forced margin 归因（P0.5 模式），不允许无归因发散。

## 风险与预处置

| 风险 | 处置 |
|---|---|
| multi-acc 改动后 V3 反而变慢（P1 只测了串行链） | Task 0 是 go/kill 门：≤60.8/57.0 才继续；否则停，报告数据，镜像不动 |
| ctypes 调用被 dynamo 拒绝或 graph-break | 不把 ctypes 裸露在被追踪区——包在 `direct_register_custom_op` 的 impl 里（对 dynamo 不透明）；Task 2 的 capture 冒烟必须过 |
| 图捕获期首次调用触发 lazy 分配 | part/sink 全部在 `process_weights_after_loading` 预分配（capture-free 区），捕获期零分配 |
| TP2 下两 rank 各自加载 .so 失败（路径/权限） | `p2_available()` 启动时探测一次并缓存布尔；失败 ⇒ 自动回退 gptq_gemm_out + selfcheck 标 FAIL |
| .so 的 ABI/运行时缺 libc++ 等 | 生产 .hip 不 include torch 头、不链接 torch 库，只用 hip/hip_runtime + hip/hip_fp16 ⇒ 依赖仅 libhiprtc/libamdhip64（镜像本就有） |
| 交付时钟 auto 档与锁频档数字不可比 | Task 6 用交付态 serve 实测 tok/s（32.4 → **≥33.4** 为过，见 G-P6 口径说明），微基准数字只作开发期门 |

## 文件结构

| 文件 | 职责 |
|---|---|
| `p2_kernels.cu`（仓库根，开发期）→ `docker/patches/p2_w4a16_gemv.hip`（交付源） | 生产内核：V3 **单累加器**（Task 0 实测多累加器是回归）+ **splitK=10**（Task 0 最优）+ reduce，`extern "C"` 入口 |
| `p2_shim.py`（开发期验证）→ 并入 `container_src/wna16_exllama.py`（交付） | ctypes 装载 + `direct_register_custom_op` 注册 + 白名单谓词 |
| `container_src/wna16_exllama.py` | **手写权威补丁副本**：P2 dispatch 分支 + part 预分配 + env 门。原文写的 `docker/patches/wna16_exllama.py` 在仓库里不存在（`git ls-files docker/patches` 只有 `layers_utils.py`），它由 `prepare_build_context.sh:42` 从运行中容器 `docker cp` 生成 ⇒ 不是源（Task 2 落地时更正） |
| `docker/Dockerfile.rocm_gfx906` | 两路径各加：COPY .hip + RUN hipcc 编译 + 存在性断言 |
| `docker/entrypoint/selfcheck.sh` | 两条 chk：.so 存在 + dispatch 谓词 grep |
| `p2_gate0.py` | Task 0 的 go/kill 扫参脚本（wiring + fp64 ULP + graph 计时，sk×u 矩阵） |
| `p2_route_gate.py` | Task 2 的路由矩阵单元门（在真实 vLLM 内核对象上验证谓词） |
| `logs/p2/*` | 证据（.gitignore 现规则 `*.log` + 白名单；需补一行 `!p2_*.log`，与 `!p1_*.log` 同形态） |

---

## Task 0: 原型改进门（multi-acc；G-P0，go/kill）

**Files:**
- Modify: `p1_kernels.cu`（V3 分支加多累加器；V1/V2 不动）

- [x] **Step 1: 改 V3 为 4 独立累加器（打散 4×UNROLL 串行 dot2 链）** —— 已落地并实测，**结果为回归，已回退**（见执行记录）。

把 `gemv<>` 内 `float raw = 0.f;` 起的累加段改为（**一次定义** `raw`，末尾聚合；V1 的 per-word 去偏与 group 尾 `zp*sg` 校正作用在被删的共享累加变量上，本提交后 **V1/V2 数值路径失效——只允许对 VARIANT==3 跑 `check`/计时**，代码里写死防误用）：

```cpp
    // multi-acc (P1 report v3): 4 independent dot2 chains, summed at the end.
    // The old single 4*UNROLL-deep serial chain was V3's most likely
    // non-bandwidth bottleneck. NOTE: VARIANT 1's per-word debias (-1024*x) and
    // group tail (zp*sg) corrected THAT serial `raw`; those corrections are NOT
    // ported to r0..r3, so after this change only VARIANT==3 may be checked
    // or timed (V1 is kill, V2 is lose per P1 report v3 -- no loss).
    float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
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
            const unsigned lo = half_lut[(w >> (4 * p)) & 0xF];
            const unsigned hi = half_lut[(w >> (16 + 4 * p)) & 0xF];
            const __half2 c = __halves2half2(__ushort_as_half((unsigned short)lo),
                                             __ushort_as_half((unsigned short)hi));
            const __half2 d = __hsub2(c, z2);
            codes = *reinterpret_cast<const unsigned*>(&d);
          } else {
            codes = ((w >> (4 * p)) & 0x000F000Fu) + MAGIC;
          }
          if (VARIANT == 3) {
            const __half2 c = *reinterpret_cast<const __half2*>(&codes);
            const __half2 d = __hsub2(c, z2b);
            codes = *reinterpret_cast<const unsigned*>(&d);
          }
          // 4 live accumulators (the point of the change); costs VGPR -- Task 0
          // must confirm the win is not eaten by occupancy loss (P2 review).
          float& a = (p == 0) ? r0 : (p == 1) ? r1 : (p == 2) ? r2 : r3;
          a = dot2(a, codes, xv);
        }
      }
    }
    const float raw = (r0 + r1) + (r2 + r3);
```

（第 123 行 `acc += (VARIANT == 1) ? scale * (raw - zp * sg) : scale * raw;` 保持不动——V1 分支已声明失效，跑不到。）

- [x] **Step 2: 两族 × V3 × sk{4,8,16} × u{4,8,16} 的 wiring+ULP+计时（落成 `p2_gate0.py` 再跑）** —— 实跑网格改成 **sk ∈ {2,4,5,8,10,20,40}（16 非法）× u ∈ {4,8,16}**，计时臂按 `main()` 的双缓冲轮换口径重写；结论见执行记录。下方 fence 是计划原稿，**其中 `*b.tensors(K,N,1)` 写在 lambda 内会把 RNG 烤进图、且单缓冲驻留 L2，不可照抄**。

多累加器改变 MLP 深度 ⇒ 最优 splitK 可能从 8 移走，**sk 维度必须重扫**（P2 评审 P1）；多累加器同时改了 fp32 求和顺序 ⇒ **fp64 ULP 在 Task 0 同批量一次量清**，别等 Task 1 才发现数值退化（P2 评审 P1）。

`p2_gate0.py`（开发期落盘文件，走真文件而非 `python3 -c`——spawn worker 不能 re-import stdin 脚本，P1 教训；`import torch` 必须显式）：

```python
# p2_gate0.py -- Task 0 go/kill sweep: V3 multi-acc, both families, sk x u.
# Gate: graph <= 60.8 us (8192) / <= 57.0 us (7168), any (sk,u) combo,
# wiring strict-ok and fp64 ULP RMS <= 0.05 for the SAME combo.
import torch, json
import p1_l1_bench as b

GATES = {(5120, 8192): 60.8, (5120, 7168): 57.0}
rows = []
for K, N in ((5120, 8192), (5120, 7168)):
    for sk in (4, 8, 16):
        for u in (4, 8, 16):
            ok = b.check(3, sk, u, K, N)          # wiring: strict gate
            us = b.timeit(b.graphed(lambda: b.mod.p1_gemv(
                *b.tensors(K, N, 1),
                torch.zeros(sk * N, device='cuda', dtype=torch.float32),
                3, sk, u, True)))                 # reduce=True: 与 incumbent-rot 同尺
            ulp = b.ulp_rms_fp64(3, sk, u, K, N)  # fp64 ULP RMS (p1_review_fix helper)
            rows.append(dict(K=K, N=N, sk=sk, u=u, wiring=ok,
                             graph_us=round(us, 2), ulp_rms=round(ulp, 4)))
            print(rows[-1], flush=True)
json.dump(rows, open('/work/p1/p2_gate0_rows.json', 'w'), indent=1)
ok_rows = [r for r in rows if r['wiring'] and r['ulp_rms'] <= 0.05]
for (K, N), gate in GATES.items():
    best = min((r for r in ok_rows if r['K'] == K), key=lambda r: r['graph_us'],
               default=None)
    print('GATE K=%d N=%d best=%s gate<=%.1f us -> %s' % (
        K, N, best, gate, 'PASS' if best and best['graph_us'] <= gate else 'FAIL'))
```

（`b.ulp_rms_fp64` 若 P1 的 `p1_review_fix.py` 里叫别的名字，执行时把它抽成同签名的 helper 再 import——不许跳过 ULP。计时臂 `reduce=False`：`p1_gemv` 9 元签名为 `x, qw, sc, zr, scratch, variant, splitk, unroll, reduce`，`tensors()` 返回 4 元组。）

Run（**先锁频**——G-P0 的 60.8/57.0 是 P1 在 1700/1000 下测的，不锁频数字不可比且无法区分"实现不行"和"当时降频了"，P2 评审 P0-4）：

```bash
P0_HOST_DIR=/home/x/vllm-work/p1 P0_CONTAINER_DIR=/work/p1 ./p0_sync.sh p1_kernels.cu p2_gate0.py
ssh x@192.168.10.11 "docker cp /home/x/vllm-work/p1/p2_gate0.py vllm-gfx906:/work/p1/ && ./p0_clock.sh lock 8 2"
ssh x@192.168.10.11 "docker exec -d vllm-gfx906 bash -c 'cd /work/p1 && python3 p2_gate0.py > /work/p1/p2_gate0.log 2>&1'"
# 分离跑完读结果，然后 unlock
```

Expected: wiring 全 ok；fp64 ULP RMS ≤0.05；**graph ≤ 60.8 µs（8192）/ ≤ 57.0 µs（7168）** 任一 (sk,u) 档。

- [x] **Step 3: 判定（写死）**

- 两族各自存在 (sk,u) 档同时满足 wiring + ULP ≤0.05 + graph ≤ 门 ⇒ **go**，把最优 (sk,u) 记入执行记录，进 Task 1。
- 达不到 ⇒ **停**：multi-acc 不是答案，报告实测值（含 ULP 与计时全表），P2 挂起交用户（不进镜像、不动 dispatch）。

**判定 = 停（见下方执行记录）**：8192 单族过门（60.51 ≤ 60.8），7168 最好 58.21 > 57.0 ⇒ 字面门不满足；且 **multi-acc 本身是负收益**，已回退 `p1_kernels.cu`。

**用户裁决（2026-09-09）= 放行**：G-P0 的判据由"逐族 ≤ 门"改为**累计收益 ≥ 1.3× 线承诺（1 148.7 µs）的 99%**，实测 1 142.1 µs = 99.4% ⇒ 通过。固定档位 = **两族 sk=10 + 单累加器**（`_P2_SPLITK = 10`），进 Task 1。代价与限制须一路带到报告：**7168 逐族只有 1.274×，未达 1.3×**，G-P6 仍按 ≥33.4 tok/s 卡不放宽。

- [ ] **Step 4: Commit（用户要求时才做）** —— **未执行**：被提交的改动（multi-acc）实测为回归，工作区已 `git checkout` 复原到 `6e83754` 内容（md5 `9c57e38a…`）。

### Task 0 执行记录（2026-09-09，1700/1000 锁频，graph 内，与 incumbent-rot 同尺；证据 `logs/p2/`）

计时口径按 `p1_l1_bench.main()` 原样复刻（两份权重缓冲逐次轮换 + 单一常驻 scratch + `timeit(graphed(fn))`，warm16/reps64/R24）。**计划 fence 里 `*b.tensors(K,N,1)` 写在 lambda 内的写法不可用**：那会把 RNG 内核烤进图，且单缓冲会让 8192 族的 21 MB 权重驻留 L2。incumbent 分母必须取 `p1_gap_probe.json["incumbent_rot"]`（78.99 / 74.14），**不是** `p1_incumbent.json`（80.05 / 74.07 是更早的单缓冲旧值，用它当分母会把所有加速比整体抬高 ~1.3%——本轮第一遍就踩了，已改脚本）。

四条被实测推翻的前提：

1. **multi-acc 是负收益**：同一 (sk,u) 点上一律变慢，sk=8 时 +1.22 µs（67.34→68.56）、sk=10 时 +1.01 µs（60.51→61.52）、sk=2 时 +3.1 µs。串行 dot2 链从来不是瓶颈——它早已被同 SIMT 上的其它 wave 掩盖。
2. **占用率不是解释**：`-Rpass-analysis=kernel-resource-usage` 一次读出 `p2_gemv_kernel` **VGPRs 21 / TotalSGPRs 31 / Scratch 0 / VGPRs Spill 0 / Occupancy 10 waves/SIMD = gfx906 硬件上限 / LDS 256 B**。四条链根本没花寄存器，也就不可能"被占用率吃掉"。（P1 四次提取 VGPR 失败，这条工具链是有效的那条。）
3. **UNROLL 完全惰性**：u ∈ {4,8,16} 在每一格 sk 上的极差 ≤ 0.10 µs（8192/sk10：60.51 / 60.53 / 60.56）。计划里 u 这一维不携带任何信息，扫它等于白花 3× 时间。
4. **splitK 才是唯一杠杆，且真最优是 10 不是 8**：sk 必须整除 K/GROUP = 40 ⇒ 合法集 {1,2,4,5,8,10,20,40}，**计划网格里的 16 结构性不可能**（40 % 16 = 8，launcher 直接 TORCH_CHECK 拒绝；P1 当年也是静默跳过）。本轮把轴扫尽：191.8 / 107.4 / 90.5 / 67.5 / **60.51** / 62.2 / 67.7 µs（8192，sk=2→40）。若照计划只跑 {4,8,16}，读到的最好成绩是 68.56 µs = 1.15×，会把"内核不行"误判成结论。

实测表（graph µs，**加粗 = 该族最优**；全部 36+6 格 wiring ok、ULP RMS 0.020–0.039 ≤ 0.05、ULP max 1–2）：

| sk | 8192 单累加器 | 8192 multi-acc | 7168 单累加器 | 7168 multi-acc |
|---|---|---|---|---|
| 2 | 191.8 | 195.1 | 190.8 | 194.3 |
| 4 | 107.4 | 108.8–109.3 | 105.7 | 107.4–107.6 |
| 5 | 90.5 | 91.9 | 89.7 | 91.3 |
| 8 | 67.5 | 68.6 | 65.4 | 66.5 |
| **10** | **60.51** | 61.52 | **58.21** | 59.17 |
| 20 | 62.2 | 62.8 | 59.5 | 60.2 |
| 40 | 67.7 | — | 61.9 | — |

门判定（分母 = incumbent_rot 78.99 / 74.14，均取单累加器）：

| 族 | 最优 | 门 | 判 | 加速比 | 每 token 收益 |
|---|---|---|---|---|---|
| 5120×8192（48 次） | 60.51 µs (sk10/u4) | ≤60.8 | **PASS** | **1.306×** | 887.2 µs |
| 5120×7168（16 次） | 58.21 µs (sk10/u8) | ≤57.0 | **FAIL** | 1.274× | 254.9 µs |

**门的悖论（裁决依据）**：G-P0 要保的是"1.3× 线累计收益 1 148.7 µs"，而实测两族都换到 sk=10、**不做 multi-acc** 时累计 = **1 142.1 µs = 承诺的 99.4%** ⇒ `32.4/(1−1142.1/30860)` = **33.645 tok/s（+1.245）**，对 1.3× 计划的 +1.2527 只差 0.008。也就是说：字面的逐族 1.3× 不达标（7168 差 1.21 µs/次），但门真正承诺的聚合收益基本达成，而且比 P1 的"现状分派 +0.755"多出一半以上。可选口径：只上 8192 一族 = +0.959；两族 = +1.245；两族 + multi-acc（更慢）= +1.173。

**裁决（用户，2026-09-09）：放行**。G-P0 判据改按累计口径（≥ 1.3× 线承诺 1 148.7 µs 的 99%，实测 99.4%），两族 sk=10 + 单累加器进 Task 1；G-P6 的 ≥33.4 tok/s 不放宽。

**未测**：真实权重（本轮全合成）、TP2 分片后的 N、fp16 输出的逐位对照产线内核——这些按计划属于 Task 1/2/4。**当前状态：G-P0 已跑并经用户放行改口径，P2 进 Task 1；镜像未动、dispatch 未改、未 commit。**

---

## Task 1: 生产内核 `.hip` + ctypes shim（G-P1：独立于 vLLM 的 wiring/numeric 门）

**Files:**
- Create: `p2_kernels_prod.hip`（开发期源；交付副本 = `docker/patches/p2_w4a16_gemv.hip`）
- Create: `p2_shim.py`（ctypes 装载 + 调用；交付时并入补丁版 wna16_exllama.py）
- Create: `p2_prod_gate.py`

- [x] **Step 1: 写生产 `.hip`（无 torch 依赖；固定 splitK=10；覆写 out；单累加器；host 校验返回错误码）** —— Task 0 的两处修正已并入下方 fence：**多累加器撤掉**（实测回归）、**splitK 8→10**（实测最优，且 10 | 40 与 640 都成立）。执行时又并入了第三处：**两级展开写法**（第一版把 `#pragma unroll 8` 压在运行期边界的 16 次循环上，G-P1 实测慢 4%，见执行记录）。

```cpp
// p2_kernels_prod.hip -- production M=1 W4A16 GEMV for gfx906, V3 structure.
// Scope (P1 report v3): whitelisted (K,N) = (5120,8192) & (5120,7168), M == 1.
// No torch headers: pure HIP + a C API driven by ctypes from the Python shim.
//   x    : fp16 [K]        (one decode row)
//   qw   : int32 [K/8, N]  (bridge layout, L0-pinned nibble order)
//   sc   : fp16 [K/128, N]
//   zr   : int32 [K/128, N/8]  (GPTQv2: stored nibble == true zp)
//   part : fp32 [splitk, N]   (caller-owned persistent scratch)
//   out  : fp16 [N]           (caller-owned capture-safe slab; OVERWRITE, no
//                              zero_ needed -- no atomicAdd anywhere)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>   // NOT <cuda_fp16.h>: that name does not exist on ROCm
                            // 7.2.4 (hipcc: file not found). P1 got away with it
                            // only because torch's load_inline adds shim -I paths;
                            // a standalone hipcc build has none. Container-verified
                            // 2026-09-09.

#define WAVE 64
#define GROUP 128
#define MAGIC 0x64006400u
#define UNROLL 8            // must divide GROUP/8 = 16; see the loop comment

__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
  float r;
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(acc));
  return r;
}

// extern "C" stays on the HOST launcher only (ROCm HIP FAQ pattern). The
// __global__ kernels are launched through hipLaunchKernelGGL by function
// pointer inside this same .so -- they need no C linkage, and marking them
// extern "C" buys nothing (P2 review P1).
__global__ __launch_bounds__(WAVE) void p2_gemv_kernel(
    const __half* __restrict__ x, const int* __restrict__ qw,
    const __half* __restrict__ sc, const int* __restrict__ zr,
    float* __restrict__ part, int K, int N, int words_per_chunk) {
  __shared__ __half xs[GROUP];
  const int lane = threadIdx.x & (WAVE - 1);
  const int n = min(blockIdx.x * WAVE + lane, N - 1);   // clamp: never diverge-exit
  const int kend = min(blockIdx.y * words_per_chunk * 8 + words_per_chunk * 8, K);
  const int g0 = (blockIdx.y * words_per_chunk * 8) / GROUP;
  const int g1 = kend / GROUP;
  float acc = 0.f;

  for (int g = g0; g < g1; ++g) {
    for (int t = lane; t < GROUP; t += WAVE) xs[t] = x[g * GROUP + t];
    __syncthreads();
    const int zz = __ldg(&zr[g * (N / 8) + (n >> 3)]);
    const float zp = (float)((zz >> ((n & 7) * 4)) & 0xF);
    const __half2 z2b = __halves2half2(__float2half(1024.f + zp),
                                       __float2half(1024.f + zp));
    // SINGLE accumulator -- deliberately. Task 0 (2026-09-09) measured 4 chains
    // at the same grid points and multi-acc LOST 1.0-1.5 us everywhere (sk=8
    // 67.34->68.56, sk=10 60.51->61.52) while costing nothing in registers
    // (VGPR 21, occupancy 10 waves/SIMD = gfx906 max, 0 spills). The serial dot2
    // chain was already hidden by the other waves on the same SIMD.
    float raw = 0.f;
    // TWO-LEVEL loop: outer steps by UNROLL, inner has a COMPILE-TIME trip
    // count so `#pragma unroll` fully unrolls it. `#pragma unroll 8` on the
    // single 16-iteration loop instead cost 2.2-2.4 us (3.7-4.0%) on both
    // families, bit-identical output, 10/10 same-signed over 5 alternating
    // rounds per family with a within-arm spread <=0.06 us (p2_unroll_ab.py;
    // p2_body_ab.py is the control that swaps only the function body, which
    // rules out the build flags). Task 0's "UNROLL is lazy" finding was true
    // only of THIS spelling: there UNROLL picked the inner width, not
    // whether the loop got unrolled at all.
    for (int kb = 0; kb < GROUP / 8; kb += UNROLL) {
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const unsigned w = (unsigned)__ldg(
            &qw[((g * GROUP / 8) + kb + u) * N + n]);
        const __half* xp = &xs[(kb + u) * 8];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
          const unsigned xv = *reinterpret_cast<const unsigned*>(xp + 2 * p);
          // pair p = (i2p, i2p+1) via ((w >> 4p) & 0x000F000F) + MAGIC
          // (L0-pinned: code i lives at bits (i//2)*4 + (i%2)*16)
          unsigned codes = ((w >> (4 * p)) & 0x000F000Fu) + MAGIC;
          const __half2 c = *reinterpret_cast<const __half2*>(&codes);
          const __half2 d = __hsub2(c, z2b);      // (1024+c)-(1024+zp) == c-zp
          codes = *reinterpret_cast<const unsigned*>(&d);
          raw = dot2(raw, codes, xv);
        }
      }
    }
    acc += __half2float(__ldg(&sc[g * N + n])) * raw;   // V3: codes are exact c-zp
    __syncthreads();
  }
  part[(long)blockIdx.y * N + n] = acc;
}

__global__ void p2_reduce(const float* __restrict__ part,
                          __half* __restrict__ out, int N, int splits) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.f;
  for (int s = 0; s < splits; ++s) a += part[(long)s * N + n];
  out[n] = __float2half(a);
}

// Returns 0 on success, -1 on a rejected shape or a failed launch. The shim
// MUST surface a nonzero return: a silently dropped launch is the hardest
// failure class to debug post-capture (P2 review P1).
extern "C" int p2_gemv(const void* x, const void* qw, const void* sc,
                       const void* zr, void* part, void* out,
                       int K, int N, int splitk, void* stream) {
  if (K % GROUP != 0 || N % WAVE != 0) return -1;   // whitelisted shapes satisfy both
  const int words = K / 8;
  if (words % (int)splitk != 0) return -1;          // partial tail must not vanish
  const int wpb = words / (int)splitk;
  if (wpb * 8 % GROUP != 0) return -1;              // chunk must end on a group boundary
  dim3 grid(N / WAVE, (unsigned)splitk);
  hipStream_t s = (hipStream_t)stream;
  hipLaunchKernelGGL(p2_gemv_kernel, grid, dim3(WAVE), 0, s,
                     (const __half*)x, (const int*)qw, (const __half*)sc,
                     (const int*)zr, (float*)part, K, N, wpb);
  int nth = 256, nb = (N + nth - 1) / nth;
  hipLaunchKernelGGL(p2_reduce, dim3(nb), dim3(nth), 0, s,
                     (const float*)part, (__half*)out, N, (int)splitk);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}
```

注意四点与原型的差别：① **展开写法本身是 4% 的性能差，不是可读性**：第一版 fence 用 `#pragma unroll 8` 作用在 `GROUP/8 = 16` 次的运行期循环上，G-P1 实测两族各慢 2.30 / 2.16 µs（原型 60.36 / new 60.45 / old 62.75，原型 58.13 / new 58.08 / old 60.24，见 `logs/p2/p2_task1_evidence.log` 的 `===UNROLL_AB===`），输出逐位一致、每轮轮换测量顺序后两族各 5 轮共 10/10 同向、同一臂轮内极差 ≤0.06 µs。已改成原型的两级写法（外层 `kb += UNROLL`、内层编译期 trip count 全展开）⇒ 生产与 `load_inline` 原型差 +0.14% / −0.08%。Task 0 的"UNROLL 完全惰性"结论只在该写法内成立（那里 u 选的是内层宽度，不是"展不展开"）；累加是**单链 32 深**（4 链版本实测更慢，已撤）；② 没有 `z2`/LUT 分支——生产只有 V3 路径，`__hsub2(c, z2b)` 直接内联（P1 报告 v3：V2 判负、V1 判死）；③ host 侧三道形状校验 + `hipGetLastError`，错误码必须被 shim 透传；④ split-K 中间写流量入账（P2 评审漏3），按 **sk=10** 重算：`part` 每 token 写+读 `2×10×N×4` B，N=8192 时 **640 KB ≈ 该族主权重（20 MiB）的 3.0%**、N=7168 时 573 KB。**这条从"可忽略"升级成相关项**：7168 差 1.3× 门 1.21 µs/次，而 573 KB 的写+读在同一带宽预算下折约 0.8–1.7 µs——若将来要摘掉这一档（计划选项"再挖 7168"），第一个动的就是它——但**不能靠降 sk 来省这份流量**：Task 0 已证 sk<10 一律更慢（sk=8 是 65.41 µs），唯一不牺牲并行度的出路是把 reduce 折进同一遍（persistent / 单遍原子或 DSMEM 风格归约，P1 判过 REDUCE 边际 1.887 µs ⇒ 值得算一次账）。

- [x] **Step 2: 编译 + ctypes shim** —— 实跑通过：`hipcc -shared -fPIC --offload-arch=gfx906 -O3` 产物导出 `p2_gemv`（`nm -D --defined-only | grep -qw`），shim 的 `rc != 0 → RuntimeError` 透传被 `verify_p2_blocks.py` 的第 [4] 项实测打到（`p2_gemv rc=-1` 原样冒出）。

```bash
hipcc -shared -fPIC --offload-arch=gfx906 -O3 p2_kernels_prod.hip -o libp2gemv.so
```

`p2_shim.py`：

```python
# p2_shim.py -- ctypes loader for the production M=1 W4A16 GEMV.
# No torch C++ ABI: raw pointers + the CURRENT torch stream (capture-safe).
import ctypes
import torch

_LIB = None


def p2_lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL("/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so")
        lib.p2_gemv.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.p2_gemv.restype = ctypes.c_int   # 0 ok / -1 rejected: MUST be checked
        _LIB = lib
    return _LIB


def p2_gemv(x, qw, sc, zr, part, out, K, N, splitk=10):
    lib = p2_lib()
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.p2_gemv(x.data_ptr(), qw.data_ptr(), sc.data_ptr(), zr.data_ptr(),
                     part.data_ptr(), out.data_ptr(), K, N, splitk, stream)
    if rc != 0:
        raise RuntimeError(f"p2_gemv rejected/failed K={K} N={N} splitk={splitk} rc={rc}")
```

- [x] **Step 3: 独立门（先于 vLLM 集成）**

`p2_prod_gate.py`：对两形状各跑 ① one-hot 16 探针（wiring ≤5e-4 绝对）② 随机 x 的 fp64 ULP（RMS/MEAN/MAX，对照 Task 0 同档实测 0.029 / 0.021——**回归即停**：换编译器 flag 不该改数值）③ graph 内计时（`torch.cuda.CUDAGraph` 包 ctypes 调用——捕获有效性本身就是测试项）④ **生产 vs 原型逐位比对**（同 x 同权重：`p1_kernels.cu` 的 **V3 单累加器**原型在 sk=10 下的输出与 `.so` 输出 `torch.equal`——两条代码路径只看 ULP 上下界会漏静默偏差；逐位不一致时记录并解释差异来源，ULP RMS 仍 ≤0.05 才放行）（P2 评审漏4）。

Run: `ssh … "docker exec -i -w /work/p1 vllm-gfx906 python3 p2_prod_gate.py"`
Expected: wiring ok；ULP RMS ≤0.05（Task 0 实测 0.020–0.039，编译差异容忍 2×）；graph 计时与 Task 0 一致（**60.53 / 58.22 µs @ sk=10 u=8，`logs/p2/p2_gate0_single_acc.log`**，容 ±20%）；④ 逐位一致或差异已归因。

Actual（2026-09-09，1700/1000 锁频，两族四臂全过，门独立跑三次）：wiring 1.22e-04；ULP RMS 0.0292 / 0.0205（与 Task 0 同档同值）；graph 60.40–60.46 / 58.12–58.88 µs（8192 −0.1~−0.2%；7168 −0.2%~+1.1%，散布是机器态双稳而非产线慢，见执行记录第 5 条）；毒化重放写回 True、replay 逐位稳定 True；④ 与原型逐位一致 `n_diff=0`。

- [ ] **Step 4: Commit（用户要求时才做）**

```bash
git add p2_kernels_prod.hip p2_shim.py p2_prod_gate.py
git commit -m "p2: production HIP kernel (extern C, capture-safe) passes standalone gates"
```

### Task 1 执行记录（2026-09-09，1700/1000 锁频；证据 `logs/p2/`：`p2_task1_evidence.log`、`p2_prod_gate_rows.json`、`p2_gate_recheck.log`、`p2_gate_from_fence.log`、`p2_contam.log`、`p2_bistability.log`）

**结论：G-P1 PASS（两族四臂全过）。过程中抓到一个 4% 的真实内核回归并修掉，另给 G-P0 那条 99% 线补了一条必须如实说的噪声带。**

1. **交付物与身份**：`p2_kernels_prod.hip`（md5 `af1535d67012d54e3a525e4686858855`）= 计划 fence 抽取结果（`p2_plan_selfcheck.py` 抽 fence → `md5sum` 对看，逐字节同）；`p2_shim.py`、`p2_prod_gate.py`；诊断件 `p2_unroll_ab.py`、`p2_body_ab.py`、`p2_proto_copy_gen.py`、`p2_gate_time_contamination.py`、`p2_7168_bistability.py`。**最终入账的那次门是"抽 fence → hipcc → 装 → 跑门"一条龙**（`logs/p2/p2_gate_from_fence.log`：60.46 / 58.12 µs、四臂全 ok），另外两次改注释后的重跑（`logs/p2/p2_gate_recheck.log` 60.40 / 58.88、`logs/p2/p2_prod_gate_rows.json` 60.42 / 58.83）同分布——**判据不能用 `.so` 的 md5**，hipcc 把源路径嵌进了产物，注释一改 md5 必变，只能靠"从 fence 现编再跑门"来锁身份。
2. **四臂实测**：wiring 两族 1.22e-04（门 5e-4）；fp64 ULP RMS 0.0292 / 0.0205，与 Task 0 同档**同值**（MEAN 0.0009 / 0.0004，MAX 1 ULP）⇒ 换构建不动数值；graph 计时 + 捕获有效性（毒化后重放必须写回）+ replay 逐位稳定 + 与原型 `torch.equal` 全 True，`n_diff=0`。
3. **P1 欠测的资源数补上**：`p2_gemv_kernel` VGPRs 19、occupancy 10 waves/SIMD（gfx906 上限）、0 spills、LDS 256 B；`p2_reduce` VGPRs 7、LDS 0。
4. **4% 回归的根因链**（拒绝按"±20% 容差内"放过，因为它逐位一致、且量级正好是 7168 离门的距离）：第一版 fence 把 `#pragma unroll 8` 压在运行期边界的 16 次循环上。① 三臂 A/B（`p2_unroll_ab.py`：原型 / 新写法 / 旧写法，每轮轮换测量顺序）→ 旧写法两族各慢 2.30 / 2.16 µs，10/10 轮同向、同臂极差 ≤0.06 µs；② 换体不换构建（`p2_proto_copy_gen.py` 造出"原型函数体 + 生产 head/tail/launcher"，`p2_body_ab.py` 测）→ 与生产 `.so` 差 +0.01 / +0.05 µs，**排除构建 flag 与 launcher**；③ 唯一剩下的自变量就是那一句展开写法。改两级展开后 new−proto = +0.14% / −0.08%。已入记忆（`gfx906-env-pitfalls` #18）。
5. **7168 有 ±0.7 µs（1.2%）机器态双稳，用户批准的 99% 线落在这条带内**：`p2_gate_time_contamination.py` 显示同一进程 CLEAN 阶段 r0–r2 = 58.09 / 58.10 / 58.11、r3–r4 = 58.81 / 58.80，且**两条代码路径逐轮同值**（prod−proto ≤0.04 µs）⇒ 不是产线慢，是机器态。`p2_7168_bistability.py` 12 轮 × 双臂 + **实时** `freq1/2_input`（`pp_dpm_*` 只会重复锁频请求，读它等于自欺）：两态都是 sclk 1670–1676 MHz、mclk 1000 MHz、边温 43–45 °C、160 W ⇒ 既非时钟亦非热降频；8192 在 ~30 次配对观测中从未出现第二态。探针独立跑两次，中位数从 58.78 翻到 58.13，两次都是两个紧簇（簇内极差 ≤0.05 µs、簇间 0.62–0.70 µs）；门自身三次独立运行也分别落在高态（58.83 / 58.88）与低态（58.12）⇒ 与门内的臂序无关，**7168 的单次点测不可用，只能报区间**。**后果必须写清**：固定 8192 = 60.42，7168 取两态 ⇒ 累计收益低态 1 148.0 µs（99.9%）/ 高态 1 136.4 µs（98.9%），Task 0 存档值 1 140.8 µs（99.3%）⇒ 99% 线被跨越，**报告里不得把小数点后一位当分辨率**。折算端到端 33.64–33.65 tok/s（带内只差 0.01），距 G-P6 的 33.4 有 0.24 tok/s ⇒ 真正的仲裁是 G-P6，不是 µs 级算术。另记一笔账：放行时引用的 1 142.1 µs 来自 60.51 / 58.21 那一轮，存档行是 60.53 / 58.22 ⇒ 1 140.8 µs，两值都 ≥99%，裁决不受影响。
6. **eager 与 graph 差 8.5 µs 是尺子属性**：门内 eager 51.92 / 50.00 µs 对上 Task 0 的 51.85 / 49.54（≤0.5 µs），而 Task 0 原型同一行也是 eager 51.85 / graph 60.53 ⇒ 两族两路皆有，不是 ctypes×graph 交互缺陷；产线口径仍按 graphed（decode 在 CUDA graph 内跑）。
7. **现场状态（Task 5 前必须清）**：开发期手装的 `/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so` 仍在（Task 2 的 `_P2_OK` 要用它），时钟仍钉 1700/1000。**在它被清掉之前不得 `docker commit` 这个容器**，否则 `.so` 会被烤进快照并顺着 Path A 进交付镜像。已在 Task 5 Step 1 补 pre-flight。
8. **未动**：镜像、dispatch、`container_src/wna16_exllama.py`；Step 4 的 commit 等用户指令。

---

## Task 2: custom op + 白名单 dispatch + 路由门（G-P2）

**Files:**
- Modify: `container_src/wna16_exllama.py`（**唯一手写权威副本**，`p05_install.sh` 装进容器 dist-packages）。落地时更正：原文指的 `docker/patches/wna16_exllama.py` 在仓库里**不存在**（`git ls-files docker/patches` 只有 `layers_utils.py`），它由 `prepare_build_context.sh:42` 用 `docker cp $CONTAINER:$SP/…` 从运行中容器反向生成 ⇒ build context 自动跟随容器，无需"同步"动作（P2 执行期实测 2026-09-09）。

- [x] **Step 1: 补丁版 wna16_exllama.py 加四样东西**

① 模块级 ctypes 装载与可用性探测（失败 ⇒ 布尔 False，dispatch 回退）。**装载成功不等于符号存在**——CDLL 是懒绑定，缺符号要到首次调用才 `AttributeError`，而那已经在引擎内/捕获期，故必须显式探测 `p2_gemv`（P2 评审 P1）：

```python
# ---- P2: gfx906 M=1 GEMV for whitelisted shapes (P1 report v3) ----
# Shapes are POST-partition (K_dim, N_dim): TP2 sharded widths (same table that
# has 8704 = 17408/2 and 3072 = 6144/2). A wrong assumption here fails SILENTLY
# -- zero hits, zero error, zero gain -- so Task 2 asserts them against the
# engine shape log.
_P2_SHAPES = frozenset({(5120, 8192), (5120, 7168)})
# ONE source of truth, on the Python side: the same constant sizes the `part`
# buffer AND is passed to the kernel, so the two can never disagree (a splitk
# larger than part's row count = out-of-bounds write the .so cannot detect -- it
# only sees a raw pointer). 10 is Task 0's measured optimum (60.51 us at 8192 vs
# 68.56 at the design's 8). Constraint the .hip DOES enforce: splitk must divide
# K/GROUP = 40 -- 16 does not, which is why the P1/P2 "powers of two" grid missed
# the optimum entirely.
_P2_SPLITK = 10
# GROUP is compiled into the .so. It is also the ONLY layout fact the kernel
# cannot check: p2_gemv validates K%128 and N%64, then reads [K/128, N] of
# scales and [K/8, N] of qweight. A per-channel layer (group_size == -1) at a
# whitelisted shape has ONE scales row, so the warm-up call would read 39 rows
# past the end -- the .so only sees a raw pointer and cannot tell. Hence
# _p2_layer_ok below gates on group_size as well as on bits/act_type.
_P2_GROUP = 128
_P2_LIB = None
_P2_OK = False
try:
    import ctypes as _ctypes
    _P2_LIB = _ctypes.CDLL(
        "/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so")
    _P2_LIB.p2_gemv.argtypes = ([_ctypes.c_void_p] * 6
                                + [_ctypes.c_int] * 3 + [_ctypes.c_void_p])
    _P2_LIB.p2_gemv.restype = _ctypes.c_int
    # CDLL binds lazily: a successful load does NOT mean the symbol exists.
    # Without this probe the AttributeError surfaces on the first call --
    # inside the engine, during capture (P2 review P1).
    _P2_OK = callable(getattr(_P2_LIB, "p2_gemv", None))
except (OSError, AttributeError):
    _P2_OK = False       # default-safe: dispatch falls through to gptq_gemm_out
# Pure-python hit log: proves P2 actually fired, per shape. Appended INSIDE the
# op impl, so a hit is a real dispatch at execution time -- not a trace-time
# branch visit, which is what made G-P3's dispatch invisible from Python.
# Same pattern as _shape_seen -- list append only, dynamo-safe.
_P2_HITS: list = []


def _p2_layer_ok(c: MPLinearLayerConfig, K_dim: int, N_dim: int) -> bool:
    """Eligibility for BOTH the split-K scratch and the routing branch.

    Two predicates would be a correctness hazard: allocate-but-not-route wastes
    memory, route-but-not-allocate crashes on a None buffer, and a group_size
    mismatch is silently WRONG, not loud. One function, so they cannot drift.
    """
    return (_P2_OK
            and (int(K_dim), int(N_dim)) in _P2_SHAPES
            and c.weight_type.size_bits == 4
            and not c.has_g_idx
            and c.group_size == _P2_GROUP
            and c.act_type is torch.float16)
```

② custom op 注册（模块级）。**import 路径已在交付容器实测**：`vllm.utils` 顶层**没有** `direct_register_custom_op`（`hasattr` False），必须 `from vllm.utils.torch_utils import`；同目录 `mixed_precision/rdna_hybrid_w4a16.py:434` 是现成模板。形参名 `mutates_args`、无 `**kwargs`、内部第一步 `torch.library.infer_schema` **要求全部入参与返回值带类型标注**（三者 2026-09-09 在容器内 `inspect.signature` + `infer_schema` 第一手复现：缺标注即 `ValueError: Parameter x must have a type annotation`）：

```python
def _p2_w4a16_gemm_impl(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                        w_zp: torch.Tensor, part: torch.Tensor,
                        out: torch.Tensor) -> torch.Tensor:
    rc = _P2_LIB.p2_gemv(
        x.data_ptr(), w_q.data_ptr(), w_s.data_ptr(), w_zp.data_ptr(),
        part.data_ptr(), out.data_ptr(),
        int(x.shape[1]), int(out.shape[1]), _P2_SPLITK,
        torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"p2_gemv rc={rc}")   # loud beats silently-wrong out
    return out


def _p2_w4a16_gemm_fake(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                        w_zp: torch.Tensor, part: torch.Tensor,
                        out: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


if _P2_OK and not hasattr(torch.ops.vllm, "p2_w4a16_gemm"):
    direct_register_custom_op(
        op_name="p2_w4a16_gemm",
        op_func=_p2_w4a16_gemm_impl,
        mutates_args=["part", "out"],
        fake_impl=_p2_w4a16_gemm_fake,
    )
```

（`mutates_args` 里列的名字必须是**形参名**且类型为 Tensor——`part`/`out` 满足，两者都列是对的，这是 op 的真实契约。**更正（M3 变异实测 2026-09-09）**：原文断言"漏掉 `out` ⇒ functionalization 把覆写丢掉、输出恒零"不成立——把 `mutates_args` 改成 `["part"]` 后 21 条断言全绿（含 row G 的 `aliases_slab=True` 与"毒化被重放写回"），在当前 torch/vLLM 版本上该后果不复现。它不是门能保的不变量，别把安全性建立在这条上；`out` 的实际行为由 row B/G 的位相等与别名断言兜底。）

③ `process_weights_after_loading(self, layer)` 末尾（`_cbuf` 分配之后）加 **条件** part 预分配 + 一次 eager 空跑。无条件分配会给**每个** WNA16 层挂 `(10, N_dim)` fp32（sk=10；按 256 次调用/token 的 N 分布估 ≈91 MB/卡），其中只有两族（64/256 次）用得上 ⇒ 按白名单分配（P2 评审 P1）；eager 空跑把 HIP code object 的懒加载钉在捕获期之外，是保证而不是运气（P2 评审漏2）：

```python
        # ---- P2: split-K scratch, only for the whitelisted families ----
        # Unconditional allocation would hang (splitk, N) fp32 off EVERY WNA16
        # layer -- ~91 MB/card at the model's N distribution, of which only two
        # shape families use any of it (P2 review P1). Fixed address, allocated
        # here in the capture-free zone; the warm-up call below forces the HIP
        # code object to load OUTSIDE graph capture, so a first-call lazy load
        # cannot become a capture failure (P2 review missing-2).
        self._p2_part = None
        self._p2_route = False
        if _p2_layer_ok(c, K_dim, N_dim):
            self._p2_part = torch.empty((_P2_SPLITK, N_dim),
                                         dtype=torch.float32, device=device)
            self._p2_route = True
            # (1, N) not (N,): the op reads out.shape[1] for N.
            _warm = torch.zeros(1, K_dim, dtype=torch.float16, device=device)
            _sink = torch.empty(1, N_dim, dtype=torch.float16, device=device)
            w_q, w_s, w_zp, _w_g_idx = self._get_weight_params(layer)
            _p2_w4a16_gemm_impl(_warm, w_q, w_s, w_zp, self._p2_part, _sink)
            torch.cuda.synchronize()
```

落地时的三处更正（都进 fence，不留"计划与代码两套"）：

- `K_dim`/`N_dim` 在该函数作用域**已经存在**（`:150` 就是 `K_dim, N_dim = c.partition_weight_shape`），不需要现取。
- 原 fence 的 `_sink = torch.empty(N_dim, ...)` 是 **1-D**，而 impl 读 `out.shape[1]` 取 N ⇒ 首次预热就 `IndexError`。改成 `(1, N_dim)`，语义不变（临时缓冲，不写进 `_cbuf`，以免预热结果与首次真实调用混淆）。
- 分配条件从 `_P2_OK and 形状在白名单` 换成 **`_p2_layer_ok(c, K_dim, N_dim)`**：与 ④ 共用同一个谓词，并把 `group_size == 128` 也纳进来。这不是加保险，是补一个真实的越界读——`.so` 只校验 `K%128`/`N%64` 就按 `[K/128, N]` 读 scales，若某个 per-channel（`group_size == -1`）8bit 层撞进白名单形状，预热那一次就会越过 scales 末尾读 39 行，而 `.so` 只看裸指针、检不出来。
- `VLLM_WNA16_P2_GEMV` **故意不放进分配条件**、只留在 ⑤ 的 op impl 里：这样 `=0` 的消融臂与开启臂内存布局逐字节相同（`_p2_part` 照旧分配），Task 3 的 A/B 只差那一个分支，不会把分配差异算进性能差。

④ `apply_weights` fused 分支前加路由。**这一处是原计划错得最狠的地方**：原文把 `M == 1 and self._p2_part is not None and env 未关` 三件事直接写进 `apply_weights`，Task 3 的 G-P3 run 1 实测证伪——该函数位于 dynamo 追踪区内，`M == 1` 在 **trace 那一刻**按当次形状折叠、False 分支被烤进编译产物，于是 on 臂 `p2_gemv` launch 数 = 0、两臂慢路径逐位同速（详见 Task 3 Actual）。实形因此改成：`apply_weights` 只留一个**与步数无关的装载期常量** `self._p2_route`（③ 里由 `_p2_layer_ok` 算好），决定"这一层进不进这个 op"；M 判断与 kill-switch 全搬进 ⑤。评审要求的 `bits == 4` / `not has_g_idx` / `group_size == 128` / `act_type is torch.float16` / 白名单四道断言**没有删，是搬进 `_p2_layer_ok`**，这样 ③ 的分配与 ④⑤ 的路由不可能各说各话（act_type 那道零成本，防的是"哪天激活走 bf16 ⇒ P2 输出直接是乱码，而 L0/路由门都用合成 fp16 张量抓不到"，P2 评审漏1）。

```python
        if self._p2_route:
            # No M here on purpose: a shape test in this function is folded at
            # trace time (G-P3). The op decides, at runtime, between the P2 GEMV
            # and the fused incumbent, and writes the result into c_out.
            torch.ops.vllm.p2_gemv_or_fused(
                x_2d, w_q, w_s, w_zp, w_g_idx, self._p2_part, c_out, temp_dq,
                bits)
            output = c_out
        elif os.environ.get("VLLM_WNA16_EXL_DEBUG") == "legacy":
            output = ops.gptq_gemm(
                x_2d, w_q, w_zp, w_s, w_g_idx, True, True, bits)
        else:
            # The fused kernel accumulates into c via atomicAdd; upstream
            # always passes fresh zeros. zero_() on our persistent slab is a
            # capture-legal fixed-address write.
            c_out.zero_()
            ops.gptq_gemm_out(
                c_out, x_2d, w_q, w_zp, w_s, w_g_idx, True, True, bits,
                temp_dq)
            output = c_out
```

（`output = c_out` 与后面的 `bias` / `reshape` 尾部共用。短路顺序把最便宜的判断放最前这条要求没有消失，只是搬进了 ⑤：那里 `int(x.shape[0]) == 1` 先短路，非 decode 调用只多付一次比较。）

⑤ 分派 op 本身（模块级，紧跟 ② 的注册之后）。dynamo **不追踪 custom op 的 impl**，所以 `M` 在这里是真实运行期形状、`VLLM_WNA16_P2_GEMV` 也是真运行期杠杆，捕获期同样有效——这正是 vLLM 自己 M=1 skinny 分派（`rocm_un_quantized_gemm` 的 impl）用的架构口径。op **返回 `None`**：`out` 同时是入参，返回它会让 dynamo 每个 rank 报一次 "output of this custom operator must not also be an input"（G-P3 run 2 实测，且官方声明将来会变成 error），就地写缓冲的 op 一律 `-> None` + `mutates_args`。`_P2_HITS.append` 也在 impl 里 ⇒ 从此它是**真实分派计数**，而不是 trace 时刻的分支访问（Task 3 之前它落在追踪区，等于永远只记一次）：

```python
def _p2_gemv_or_fused_impl(x: torch.Tensor, w_q: torch.Tensor,
                           w_s: torch.Tensor, w_zp: torch.Tensor,
                           w_g_idx: torch.Tensor, part: torch.Tensor,
                           out: torch.Tensor, temp_dq: torch.Tensor,
                           bits: int) -> None:
    """M-based dispatch that SURVIVES torch.compile.

    This `if` cannot live in apply_weights: that function is inside the
    dynamo-traced region, so `M == 1` is resolved at TRACE time against the
    shape the region was first traced with (the prefill M) and the False branch
    is baked into the artifact -- every captured decode graph then replays the
    incumbent. Measured in G-P3 run 1: 0 p2_gemv launches compiled vs
    64/step/rank under enforce_eager. Dynamo does not trace inside a custom op,
    so here M is the real runtime shape, including during cudagraph capture --
    which is exactly how vLLM's own M=1 skinny-GEMV dispatch
    (rocm_unquantized_gemm_impl) stays live in the compiled engine.

    Returns nothing on purpose: `out` is also an input, and an op that returns
    an aliased input makes dynamo warn ("must not also be an input to this
    custom operator ... may not alias any inputs", seen once per rank in G-P3
    run 1) and that is slated to become an error. Declaring `out` mutated is
    what keeps the write visible -- the same shape as ops.gptq_gemm_out and
    prefetch_ops' start_prefetch.
    """
    if int(x.shape[0]) == 1 and os.environ.get("VLLM_WNA16_P2_GEMV", "1") == "1":
        # p2_reduce overwrites out[n] for every n, so unlike the fused kernel
        # (atomicAdd into c) this needs no out.zero_() first.
        _P2_HITS.append((int(x.shape[1]), int(out.shape[1])))
        _p2_w4a16_gemm_impl(x, w_q, w_s, w_zp, part, out)
        return
    out.zero_()
    ops.gptq_gemm_out(out, x, w_q, w_zp, w_s, w_g_idx, True, True, bits,
                      temp_dq)


def _p2_gemv_or_fused_fake(x: torch.Tensor, w_q: torch.Tensor,
                           w_s: torch.Tensor, w_zp: torch.Tensor,
                           w_g_idx: torch.Tensor, part: torch.Tensor,
                           out: torch.Tensor, temp_dq: torch.Tensor,
                           bits: int) -> None:
    return


if _P2_OK and not hasattr(torch.ops.vllm, "p2_gemv_or_fused"):
    direct_register_custom_op(
        op_name="p2_gemv_or_fused",
        op_func=_p2_gemv_or_fused_impl,
        mutates_args=["part", "out"],
        fake_impl=_p2_gemv_or_fused_fake,
    )
```

- [x] **Step 2: 路由矩阵门 `p2_route_gate.py`（先于引擎）**

在真实 vLLM 内核对象上断言。构造方式 = **P1 L0 现成 harness**：`p1_l0_mapping.make_layer_and_kernel(K, N, seed)`（`:129`；内部 `build_layer` 用 `_harness_param` 挂 `input_dim/output_dim/packed_dim`，再 `WNA16ExllamaKernel(cfg,…)` + `process_weights_after_loading`）——P1 已验证这条构造能走通真实内核的权重加载路径，直接复用，不要在 Task 2 另发明一套（P2 评审 P1：原文"Task 1 的 make_layer_and_kernel"是引用错位）。

- (5120,8192)/(5120,7168) × M=1 ⇒ 输出与直接 ctypes 调用逐位一致（op 通路正确）；
- 同形状 × M=2 ⇒ 走 `gptq_gemm_out`（用 `_P2_HITS` 长度断言 P2 未触发，别只靠 `_SHAPE_LOG`——它只记形状不记分支，P2 评审漏9）；
- 非白名单 (5120,17408) × M=1 ⇒ P2 未触发且输出与 incumbent 一致；
- `VLLM_WNA16_P2_GEMV=0` ⇒ 全部回退（`_P2_HITS` 保持空）；
- 五形状 × M=1 后 `_P2_HITS` 计数 = 白名单族各 1、其余 0 ⇒ 谓词精确命中。
Expected: 五行 `ROUTE ok`。

**Actual（2026-09-09，G-P2 PASS）：门最终是 7 行 21 断言，不是 5 行。** 两行是落地时新增的：

- **row A 探针**（`_P2_OK` / op 已注册 / `_p2_part` 在两族分配、三族为 `None`）：不加这行，下面所有"P2 未触发"的负向断言在 `.so` 静默缺失时**全部空过**——绿色测的是空气。
- **row G capture**（一行一步骤 3，见下）。

**首跑是 FAIL，而且 FAIL 的是仪器不是补丁。** C/D 三行报 `n_diff ≈ 71%`、单调位指数 `max_ulpidx = 1.1e4–1.5e4`。没有按"容差内"放过，而是用 `p2_repro_discriminator.py` 三臂完全绕开 `apply_weights`（直调 op 进 `_cbuf[:M]` / 直调 op 进新分配 / slab+新 tdq）复现，加 `p2_incumbent_determinism.py` 对 fp64 零自由度参考比对，结论：

> **`gptq_gemm_out` 在 gfx906 上逐次调用不重现**：6 次相同输入差 ~71% 位置、`max|d| = 3.1e-2 … 6.3e-2`（输出幅度 17–22），与输出缓冲是否同一块无关；它自身对 fp64 参考的误差同量级（`max 2.6e-2…5.5e-2`）⇒ 这个 spread 就是它的噪声地板（融合内核 fp16 atomicAdd 跨 split-K 部分和、顺序随调度变）。**补丁无罪。**

由此定下本任务后续的数值判定规则（[[kernel-swap-acceptance-criteria]] 的细化）：**P2 侧用位相等**（`p2_reduce` 是覆写、无 atomicAdd，实测四次重跑 `n_diff=[0,0,0,0]`），**incumbent 侧只能用"以被测对象自身噪声定标"的绝对带** `tol = max(6·ULP@|out|max, 2·σ_inc实测)`。两个坑写进了门头注释：① 单调位指数"ULP"尺子在抵消区直接炸到 1.1e4，对 incumbent 完全不可用；② 拿张量和"它被写进的那块缓冲的 view"相比是空断言（早期探针就是这么造出"同 slab 就重现"的假结论的）⇒ 门里所有比较对象一律 `.clone()`。

21/21 绿（`logs/p2/p2_route_gate2.log`）：

| 行 | 断言 | 关键数字 |
|---|---|---|
| A 探针 | `_P2_OK`、op 注册、5 形状 part 分配/跳过 | `(10, 8192)`/`(10, 7168)` 有，17408/8704/3072 为 `None` |
| B op 通路 | 两族 × M=1 与裸 ctypes **逐位一致** + P2 四次重跑位稳 | `equal=True n_diff=0`，`hits=1 last=(K,N)` |
| C M=2 | 零命中 + 落在 incumbent 带内 + 与直调 `gptq_gemm_out` 同带 | `hits=0`，`max|d|=3.906e-2 ≤ tol=1.016e-1`（σ=5.078e-2） |
| D 非白名单 | 17408/8704/3072 × M=1：part=None ∧ hits=0 ∧ 带内 | `4.688e-2/3.125e-2/4.688e-2 ≤ 9.375e-2` |
| E env 开关 | `0`/未设/`1` ⇒ 命中 `0/1/2`；开关真的换内核且输出有限 | `n_diff=6378/8192` |
| F 谓词 | 五形状一次扫描，命中计数逐族精确 | `want==got`，总命中 2 |
| G capture | 捕获+两次 replay 位稳、毒化被写回、与 eager 逐位同、捕获期命中 +1、返回确实别名 `_cbuf` | `rewrote_poison=True ==eager=True aliases_slab=True` |

**门有牙齿——四臂变异实测**（`p2_gate_teeth.sh`/`p2_gate_teeth2.sh`，每轮 `cp` 备份 + sed 注入 + 跑门 + 恢复 md5 收尾）：

| 变异 | 注入 | 结果 |
|---|---|---|
| M2 | 白名单谓词 `... in _P2_SHAPES` → `and True`（形状全放行） | 8 行转红、G-P2 **FAIL** ⇒ 白名单断言有效 |
| M1 | `w_s`/`w_zp` 传参互换 | **不是静默乱码，是硬崩**：`Memory access fault by GPU node-1 … Reason: Page not present or supervisor privilege.`，进程在 row A 就死（③ 的 eager 预热先撞上换过的实参）——内核按 `[K/128,N]` fp16=655 KB 去读只有 163 KB 的 qzeros。副产物：**预热调用顺带成了传参顺序的边界检查**。原计划"传反会产出看着合理的乱码、没有数值带能抓"这句**作废** |
| M1′ | 给内核传 `K//2`（内存安全、结果仍是"像样的点积"） | row B 精确转红：`n_diff=8190/8192 max=1.4e+01`（7168 同形），其余行 C/D/F/G 保持绿 ⇒ row B 抓的正是这一类静默错 |
| M3 | `mutates_args` 去掉 `"out"` | 门**仍全绿** ⇒ 步骤 ② 里"输出恒零"的断言被证伪（已在该处更正） |

七个日志与首跑失败样本均已归档（统一 `p2_` 前缀，否则被 `*.log` 忽略规则吞掉）：`logs/p2/p2_route_gate.log`（仪器错版 FAIL）、`p2_route_gate2.log`（PASS）、`p2_inc_det.log`/`p2_inc_det2.log`、`p2_discrim.log`、`p2_teeth.log`、`p2_teeth2.log`（七文件 md5 互不相同）。

引擎侧白名单形状回读（Task 3 首次起服务后核）：白名单是 **partition 后**形状，写错 ⇒ 静默零命中零报错零收益（P2 评审 P1/漏10）——从 `VLLM_WNA16_EXL_SHAPEDBG` 的 shape CSV 里断言 `(5120,8192)`、`(5120,7168)` 真实出现过；**每 step ≈64 次（48+16）这个数只能从 per-worker trace 数 `p2_gemv_kernel` 的 launch 得到**，`_P2_HITS` 无导出路径且位于 dynamo 追踪区内，给不出计数（见 Task 3 Step 2）。TP2 下 `_p2_part` 是 per-rank 本地张量，两 rank 各分配各的、无跨 rank 可见性问题（vLLM 的 kernel 对象本就 per-rank），但仍要在两 rank 日志里各看到一次 shape 命中才放行（P2 评审漏8）。

- [x] **Step 3: capture 冒烟**

`torch.cuda.CUDAGraph()` 包一次 whitelisted 调用 + 两次 replay，输出逐位一致 ⇒ dynamo/custom-op/capture 三者相容。

**Actual（并入 row G，两族全绿）**：字面版"两次 replay 逐位一致"**会空过**——一次泄漏到图外的 launch 同样"位稳定地重放"，只不过稳定地放着旧数据。故实现加了三个可证伪的补强：① 先 `_cbuf.zero_()` **且** `out.zero_()` 两处一起毒化（若 functionalization 返回图私有池里的副本，只毒 slab 会空过），再要求 replay **把毒写回来**；② 断言返回张量 `data_ptr()` 确实别名 `kern._cbuf`；③ 统计捕获期 `_P2_HITS` 增量（eager 1 次 + 捕获 1 次 = 2）。两族结果：`replay_stable=True rewrote_poison=True ==eager=True hits_in_capture=2 aliases_slab=True` ⇒ ctypes 传入 `torch.cuda.current_stream().cuda_stream` 在捕获下确实进的是捕获流。

- [x] **Step 4: 同步到开发容器并装 patches 副本**

Run: `P0_HOST_DIR=/home/x/vllm-work/p1 P0_CONTAINER_DIR=/work/p1 ./p0_sync.sh … && ./p05_install.sh container_src/wna16_exllama.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py`（FORCE=1 视守卫）。hipcc 产物放 `/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so`（`docker cp` 或容器内 hipcc 编译）。

**Actual**：`container_src/wna16_exllama.py` 319 → **431 行**，md5 `37325e2aa3da0dcdcf3317111e098883` 与容器内已安装副本**完全相同**；旧版留档 `…wna16_exllama.py.pre-p2.bak`（md5 `d3f023dc…`）；语法门在容器内 `python3 -m py_compile` 过（本机 Windows 无可用 python，`python`/`python3` 静默无输出）。步骤标题里"装 patches 副本"这半按 Files 行的更正作废——patches 目录由 build-context 脚本反向生成。

- [ ] **Step 5: Commit（用户要求时才做）**

```bash
git add container_src/wna16_exllama.py p2_route_gate.py
git commit -m "p2: whitelisted M=1 dispatch via custom op (route gate green)"
```

---

## Task 3: 引擎内 A/B（G-P3）

**Files:**
- Create: `p2_ab.sh`（P0.5 `p05_ab.sh` 同款双臂改单臂 P2）

- [ ] **Step 1: 双臂 A/B（1425/1000 锁频，分离作业）**

两处先行修正（P2 评审 P0-7 + P1/Task5 全节）：
1. `p0_step.py:94` 的 `SamplingParams(temperature=0.0, max_tokens=...)` **没有 `ignore_eos`** ⇒ 提前停则 tok/s 被低估、greedy 长度不足 256，两臂都受影响。加 `ignore_eos=True`（P0.5 链路同样受益），并断言落盘 token 数 == max_tokens。
2. **每臂一个全新的编译缓存根**（`p2_ab.sh`：`export VLLM_CACHE_ROOT=/tmp/p2cache_ab_$arm`）。`VLLM_WNA16_P2_GEMV` 在 dynamo 追踪期被求值成常量，磁盘编译缓存命中时 env=0 **不生效** ⇒ off 臂可能静默仍在跑 P2，A/B 整个失真（Dockerfile 注释里已记过同类坑："a cache built under one kernel mode crashes another"，[[gfx906-env-pitfalls]] 第 6 条）。**这段理由已在执行期被改写**（Task 3 Actual）：真因不是缓存串台、也不是 env 被烤成常量，而是 `apply_weights` 里那个 `M == 1` 在编译态根本不生效；修复把 M 判断与 env 读取一起搬进 custom op 的 impl（dynamo 不追踪 op 内部）⇒ 两臂编出**同一个**产物，env 是真正的运行期杠杆（off 臂 trace 里 `p2_gemv` launch = 0 直接证伪了"off 臂静默跑 P2"这一担忧）。独立缓存根仍然保留，理由换成缓存卫生：pre-fix 代码留下的产物不能被复用，而缓存键不覆盖内核选择分支。**计划原稿这里写的是 `rm -rf /root/.cache/vllm`，两处错**：① 目录错——`p0_step.py:31-33` 的 root 是**按 W4A16 模式分键**的 `$HOME/.cache/vllm-w4a16-exllama`（dev 容器实测 1.7 GB），而 `/root/.cache/vllm` 只有 12K，照字面执行等于"清了个空目录、缓存照旧串台"；② 不必删——把 root 指到一次性 scratch 即可达到同一目的，且不销毁任何东西（代价是每臂各付一次冷编译，本来就要付）。

- 臂 off：`VLLM_WNA16_P2_GEMV=0` ⇒ 必须复现 33.008 ms/step（锚点漂移 ⇒ 先查条件）。
- 臂 on：默认 ⇒ EXPECTED_MS=**31.9**（= 33.008 − 1.15，即 G-P0 的 1.3× 线承诺兑现 ~90%）；**perf 门：step ≤ 31.9 ± 0.3 ms 且 tok/s ≥ 30.6**（step 门是承诺，tok/s 门是"确实快于 off 臂 30.30"的下限健全性检查——旧稿的 18.9 比 off 臂自己还低，无判别力）。
- 每臂 `p0_step.py` 落 `p0_wall.json` + greedy token；两臂 token 逐位对比（P0.5 G6 模式：分歧必须能被 teacher-forced margin 归因——V3 与 incumbent 数值不同是预期内的，**重点是没有无归因的语义跑偏**）。

- [ ] **Step 2: 判定**

- on 臂 step ≤31.9 ms 且 greedy 分歧可归因（或逐位一致）⇒ **G-P3 过**。
- step 未降 ⇒ 先查 dispatch 是否真触发。**注意 `_P2_HITS` 在这里帮不上忙**：落地文件里它只有定义与 append 两处（无任何导出路径，`_dump_shapes` 只写 `_shape_seen`），而且那次 append 和 `[7]` 的 `_shape_seen` 一样位于 dynamo 追踪区内 —— 同形状后续执行走编译产物/graph replay，Python 根本不跑（[[gfx906-env-pitfalls]] 第 7 条）。**状态已随修复改变**（Task 3 Actual）：append 现在位于 custom op 的 impl 内，而 impl 不被追踪 ⇒ `_P2_HITS` 从此是**真实分派计数**而不是 trace 时刻的分支访问。但它仍然不落盘，所以每步计数的权威证据仍是 device 侧：`p0_step.py` 已经自带 per-worker trace（`DELAY_ITERS=8` + `ACTIVE_ITERS=64` ⇒ 窗口就是 64 个干净 decode step，两 rank 各一份文件），数 `p2_gemv_kernel`/`p2_reduce` 的 launch 数 ÷ 64 应当 ≈64/step（48+16）；形状覆盖则看 `{OUT}/p0_step_shapes.csv.<pid>` 两份里 `(5120,8192)`/`(5120,7168)` 是否真实出现。
- 语义跑偏且不可归因 ⇒ **停**，回退 env 门，报告。

**Actual（2026-09-09）：本门跑了两遍才过，中间换过一次实现位置。**

**run 1 = FAIL（11/15）**。off 臂锚点复现良好（32.893 ms），但 on 臂 **32.890 ms，delta 0.003 ms**——两臂慢路径逐位同速；device 侧证据是决定性的：**`p2_gemv` launch 数 = 0**（两 rank 皆 0），incumbent 仍是 **256/step/rank**（该掉到 192）。四条 FAIL 全指向同一件事：**编译态 P2 从不被分派**，而不是"P2 慢了"。

定位过程（严格走 systematic-debugging，先证仪器再改码）：
1. 先用 `enforce_eager=True` 的引擎对照（`p2_eager_probe.py`）：**eager 下 64 launch/step/rank、incumbent 精确降到 192、两 rank 一致** ⇒ 内核、ctypes shim、custom op 注册、白名单形状（partition 后）、装载期 scratch **全部正确**。缺的只是"这段判断在编译态有没有地方跑"。
2. 这一步的探针自己翻车两次，都值得记住：**(a)** 第一版把类补丁写在 `main()` 里 ⇒ spawn 出的 TP worker 以 `__mp_main__` 重跑脚本、只执行模块级代码，两 worker 全无补丁、落盘 0 行；反过来若整份脚本没有 `__main__` 守护，worker 会照着模块级代码再建一个引擎（`Engine core initialization failed`）。**(b)** 第二版误把 `VLLM_WNA16_EXLLAMA` 的 setdefault 覆盖掉 ⇒ `can_implement()` 直接 False、引擎静默落到 `TritonW4A16LinearKernel`，整轮测的是"另一个内核"，表现只是 incumbent=0 / 0 行，与"没接管"完全同形。修法：脚本开头打印 `env:`、结尾打一条可区分两种失败的解释（"0 rows means WNA16ExllamaKernel was never selected"）。见 [[gfx906-env-pitfalls]] 第 28、29 条。
3. 结论：**`M == 1` 写在 `apply_weights` 里等于写在追踪区内**（本文第 5、7 条的同一族），运行期形状只在 trace 那一刻被看到；`VLLM_WNA16_P2_GEMV` 同命。

**修复（改实现位置，不改判据）**：新增 `torch.ops.vllm.p2_gemv_or_fused`——把 `M == 1` 与 env 读取放进它的 **impl**，impl 内部再走 `_p2_w4a16_gemm_impl` 或 `out.zero_()+gptq_gemm_out`；`apply_weights` 只留一个**与步数无关的装载期常量** `self._p2_route`（= `_p2_layer_ok(...)`，在 `process_weights_after_loading` 算好）决定"这一层进不进这个 op"。dynamo 不追踪 custom op 内部 ⇒ 运行期 M 与 kill-switch 都活。**这不是自创旁门**：vLLM 自己的 M=1 skinny 分派（`rocm_unquantized_gemm_impl` 的 `n == 1 and m % 4 == 0`）就是同一架构口径。
**诚实边界（必须带进报告）**：独立复现 harness（`p2_dynamo_probe.py`，`specialize_int` 两个 regime、把 pre-fix 路由逐字复刻成 arm[2] 并自带计数器）**没有复现出"折叠"**——两臂在其下都照常命中。所以"追踪区内形状测试被折叠"这一机制的证据只有**引擎内实测 + eager 对照**两条，不能说已在受控条件下复现。同一轮还顺手证伪了一处计划断言（arm[2] 最初复用 `_P2_HITS` ⇒ 结构上永不命中，见第 30 条）。
改码后 G-P2 全套重跑：**21/21 绿**（含 M=2 回退、off-list、捕获毒化重放 `rewrote_poison=True aliases_slab=True`），dynamo 探针 arm[3] 两 regime 均 `compiled@M=1 == eager@M=1 bit-for-bit: True`。

**run 2 = PASS（15/15）**：off **32.835 ms**（锚点 33.008 ± 0.3 内）、on **31.934 ms**、**delta 0.902 ms = 31.31 tok/s**（门 ≤32.2 / ≥30.6）；`p2_gemv` 两 rank 各 **64.0/step**、incumbent 精确 **256 → 192**；off 臂 launch=0（env 是真运行期杠杆）；shape CSV 两族 **48+16** 复现；**两臂 greedy 256 token 逐位一致、0 处分歧**。
**为什么兑现的是 0.90 ms 而不是承诺的 1.1487 ms（−78.5%，不是 −100%）**：把两臂 trace 的内核账拆开看——off 臂 256 次 incumbent 共 19 733 µs/step；on 臂剩 192 次共 15 120 µs ⇒ **被替换的那 64 次平均只值 72.0 µs**，而 G-P0 的分母是微基准的 78.99 / 74.14 µs。P2 侧实测 `p2_gemv` 54.3 + `p2_reduce` 5.1 = **59.4 µs/调用，比承诺的 60.5/58.2 更快**。⇒ **差额全在分母**：引擎内权重长期驻留 L2，被替换的 incumbent 比"两份缓冲逐次轮换"的微基准便宜，与 [[gfx906-env-pitfalls]] 第 15 条（分母必须与被测数字同源同尺）同源同因。设备侧算得 812 µs/step，与 wall delta 902 µs 相符（同一次改动内自洽）。

**run 2 暴露的一处契约问题，已在 run 3 前改掉**：dynamo 每个 rank 报一次 `UserWarning: vllm::p2_gemv_or_fused ... The output of this custom operator must not also be an input to this custom operator ... This is deprecated and will become an error in a future version of PyTorch`——根因是路由 op 把 `out` 既当入参又当返回值。vLLM 自己写"就地写缓冲"的 op 一律 `-> None` + `mutates_args`（`ops.gptq_gemm_out`、`prefetch_ops.py` 的 `start_prefetch` 连注释都写明"declare mutated to prevent torch.compile from reordering"）⇒ 已把 impl/fake 改成 `-> None`、调用点用 `output = c_out`，`aliases_slab` 性质由 G-P2 row G 继续守住。

**run 3（终版代码 = 要进镜像的那份）= PASS（15/15）**：off **32.776 ms** / on **32.120 ms**、delta **0.656 ms = 31.13 tok/s**（门 ≤32.2 ⇒ **余量只有 0.08 ms**）；`p2_gemv` 两 rank 各 **64.0/step**（窗口内 8 192 次）、incumbent 两 rank 精确 **256 → 192**、off 臂 launch **0**；shape CSV 两族 48+16；**别名警告 off/on 两臂均 0 次**（run 2 是每臂 2 次）⇒ 重构达到了目的；greedy 两臂 **256/256 逐位一致、0 处分歧**。
**两遍 PASS 的读数为什么差 0.19 ms（重要，别读成"重构变慢"）**：把 W4A16 家族的设备侧账并排看——off 臂 19 733 vs 19 742 µs/step、on 臂 18 921 vs 18 916、**净收益 811.7 vs 826.1 µs/step**，单次耗时逐项吻合到 0.02 µs（incumbent 77.08/77.12 与 78.75/78.73、`p2_gemv` 54.34/54.32、`p2_reduce` 5.05/5.05）⇒ **设备侧完全没有差别，0.19 ms 全在 W4A16 家族之外**（off 臂三次读数 32.893/32.835/32.776 也各有 0.12 ms 漂移）。⇒ 本门的可信读数应引**设备侧 delta 812–826 µs/step**，wall step 自带 ±0.2 ms 噪声。

- [ ] **Step 3: Commit（用户要求时才做）**

```bash
git add p2_ab.sh logs/p2/
git commit -m "p2: in-engine A/B at 1425 MHz -- step -0.7ms, greedy parity attributable"
```

---

## Task 4: L2/L3 验收（G-P4）

方法沿用 [[kernel-swap-acceptance-criteria]] 与 spec §3/§8 的既有形式化判据，**不新造标准**（P2 评审漏5/漏6：原稿两处只写了门没写怎么量）。

- [x] **Step 1: L2（引擎内真实权重的 fp64 ULP）**

**怎么取真实 x 与权重（Task 3 修复后已改写）**：原稿写"monkeypatch `apply_weights`、`_P2_HITS` 非空时 `torch.save`"——**两处都失效了**：`_P2_HITS` 的 append 已搬进 op impl（不在 `apply_weights` 里），而在追踪区内放 `torch.save` 会当场炸引擎初始化（[[gfx906-env-pitfalls]] 第 5 条）；就算能跑，编译/graph-capture 路径上拿到的 `x` 也可能是捕获用的哑输入而非真实激活。改法：monkeypatch **普通函数** `_p2_w4a16_gemm_impl`（由 op impl 在运行期直接调用），在**独立的一次 `enforce_eager=True` 引擎**里做（不是 A/B 那两臂），每族首个白名单形状命中时落盘 `(x, w_q, w_s, w_zp, part, out)` 到 `/work/p2/l2_dump_{K}x{N}.pt`；**存 clone**——`out` 是常驻 slab、`part` 是复用的 splitK scratch，下一步就被覆写。
**离线判定**：加载 dump，用零自由度 fp64 参考（int4 解包 → fp64 逐乘积 → fp64 求和 → 末次 RNE cast 到 fp16）复用 P1 `p1_review_fix.py` 的 ULP 拆分器，算 RMS/MEAN/MAX，并同表算 incumbent 基线（同 x 同权重走 `ops.gptq_gemm_out`）。
**门（两条同时满足）**：① 回归门 RMS ≤0.05（P1 实测 0.024–0.027）；② 产线比较门 RMS ≤ incumbent 基线的 1/10（基线 69–219 ⇒ ≤6.9–21.9）。同时记录 MAX 与 big-only 分项。

**Actual（2026-09-09，G-P4 L2 PASS，8/8；证据 `logs/p2/p2_l2_gate.log`、`p2_l2_rows.json`、`p2_l2_dump1.log`、`p2_l2_dump2.log`、`p2_l2_dump_md5.txt`）**

1. **取样**：`enforce_eager=True` 单引擎、TP2、两族 × 2 rank × 2 份不同权重 = **8 份 dump**（各 19.4/22.1 MB，宿主 `/work/p2/l2/`，md5 已入账）。钩子按 `w_q.data_ptr()` 去重 ⇒ 每 rank 每族拿到的是**两个不同模块**；每 rank 每族过滤掉 16 次 `x.abs().amax()==0` 的加载期预热（正是 `_warm = torch.zeros(1, K)` 那一步）。**口径限制**：激活取自 eager 引擎 ⇒ 对交付路径"有代表性"，不是逐位相同。
2. **第一次采集死于探针自身**：8192 一族 4 份已落盘后，钩子里的 `json.dump` 因 tuple 键抛 `TypeError`，而我 `except` 里打印完又 `raise` ⇒ 异常穿透 op 调用路径 ⇒ 两个 TP worker 一起倒下。修法：键转 str、**删掉 raise**（钩子在 op 调用路径上，任何异常都是 worker 崩溃，而此时张量已落盘），另加 `FAMS_RUN` 续跑开关只补缺的 7168 一族 —— 门不必为了"看仪器行为"重跑第二遍。**入坑记忆：探针只许吞异常，不许外抛。**
3. **两条门都过**：P2 在真实权重/真实激活上 **RMS 0.0118–0.0331**（门 0.05）、MEAN ≤7e-4、**MAX = 1 ULP**、偏离参考的占比 0.01–0.07%；同操作数 incumbent **RMS 29.7–94.1**（每份跑 3 次）、MAX 高达 6592 ULP、**71–74% 位置非零误差** ⇒ INC/P2 **≥1301×**（门 10×）。真实权重下的 incumbent 基线（30–94）比 P1 合成基线（69–219）低一档，但结论方向不变。
4. **必须一起读的两条限制**：① 那个 MAX 数千 ULP 全部落在 `|ref| ≤ 0.01` 的近零尾部（绝对误差 ~1e-4，是抵消误差，单调距离在近零处尺度过细），big-only 分项（`|ref|>0.01`）是 **P2 0.0119–0.0206 vs INC 4.13–4.34 ⇒ 仍 ~200×** ⇒ 换尺子不改判；② 确定性不对称：P2 重跑 **8/8 逐位相等**，而 incumbent 自己重跑就有 **71–72% 位置变动（MAX 1360–6592 ULP）**（atomicAdd 顺序，见 [[exllama-fused-kernel-atomic-add]]）——**被换掉的那条路径才是不可复现的那条**。

- [x] **Step 2: L3（teacher-forced margin 归因，非逐 token 等值）**

1. **噪声地板先测**：同一路径（全 off 臂）prefill-vs-decode 的翻转率——引擎自身就有百分之几的翻转，这是背景噪声不是 bug。
2. **机制**：`prompt_logprobs` + top-k `logprobs`，**上限须显式抬到 32**（`EngineArgs.max_logprobs` 默认 20，不抬则 top-32 静默截断）。**执行期改为离线 `LLM.generate` 同一套 SamplingParams**（`p2_l3_run.py` 传 `LLM(..., max_logprobs=32)`）而不是起 `/v1/completions`：两臂各起一个引擎要 2×7 min，起 serve 只会再叠一层 HTTP 与批调度噪声，而 L3 要的是"同一 token 序列上两臂的 top-2 margin"，离线口径完全等价；serve 数值归 G-P6。
3. **判据（形式化，与 P0.5 同）**：**存在 M₀ 使 margin > M₀ 处零翻转**。高 margin 处一次翻转 = 真 bug ⇒ 停；低 margin 处翻转 = 两条浮点路径的正常分叉。总翻转率**不是**判据。
   **这条按字面写是空洞的**：取 M₀ = "最大翻转 margin" 永远成立。让它成为门的是**同一台引擎自己付的地板**（`p2_l3_gate.py` 实现口径）：
   - `M0_swap` = 两臂在**同一强制序列**上 top-1 不一致处的 min-margin 最大值；
   - `M0_floor` = **同一臂内**"prefill 路由 vs decode 路由"在同一位置不一致处的 min-margin 最大值（同权重、同引擎、同 token，只差评估路由——引擎与自己分歧的固有幅度）；
   - **门 = `M0_swap ≤ M0_floor`**，即换内核引入的全部分歧都落在引擎已经自发分歧的 margin 带内。min-margin 取两侧 margin 的**较小者**（只有两边都自信才算"高 margin 翻转"）。
   **两臂必须评分同一条序列**：teacher-forced 的序列**恒取 off 臂**的自由运行结果（off 先跑并落盘 `l3_seqs.json`，on 读它；缺文件即失败）。两臂自由运行的差异是**结果**（记为 `free_run_divergence`），不允许它反过来改变输入。
   **落地更正（仪器实现时发现，必须记在门前面）**：上面那句"同一强制序列"如果只走 `prompt_logprobs`，测到的是 **prefill 路由（M>1）**——而 P2 只在 M==1 生效，两臂在那一行代码上跑的是同一个 incumbent ⇒ 跨臂翻转 ≈ 0，**门会绿着测空气**。所以 `p2_l3_run.py` 加 **pass 2b**：每个被评位置单独发一个请求（`prompt = seq[:t-1]`, `max_tokens=2`，取**第二个** token 的分布 = 真正 M=1 decode step 在位置 t 的分布），跨臂 `M0_swap` 用 2b、`M0_floor` 用"同臂 2a vs 2b"，2a 跨臂只留作**对照**（~~应为 0 翻转，非 0 说明有内核之外的东西动了~~ **执行期证伪**：实测 4 处，来源就是 incumbent `gptq_gemm_out` 的 atomicAdd 不可重现，见 Step 2 Actual 第 2 条）。三个附带约束：
     - **2b 必须串行提交**：`M` 是整步展平后的 token 数，多序列并发时"decode step"是 M=k，on 臂同样不会走 P2 分支——并发跑等于两臂都关。
     - ~~**两臂位置集合必须相同**，否则 swap 集是在比不同位置 ⇒ 直接 FAIL。~~ **执行期作废**（Step 2 Actual 第 2 条）：drop 的成因是"该臂 prefill 路由在 t-1 的贪心 ≠ 强制序列"，而 on 臂的对照物正是 P2 的解码输出 ⇒ 两臂解码路线一分歧，该位置就只有一臂保留。要求集合相等 = 要求换内核不改变任何行为。改为**比较交集**，只有交集 < min(两臂行数)×90% 才 FAIL，不对称逐 prompt 入档。
     - **地板为空不许白过**：有跨臂翻转却没有地板时，比较失去参照（同一引擎 prefill-vs-decode 本就有百分之几自发翻转），判 FAIL 而不是让 `M0_floor` 默认 0 蒙混。
     - 自由运行行只在**两臂各自轨迹仍一致的前缀**上跨臂比较（行 j 的条件是 `comp_own[0..j-1]`）：分歧点之后比的是两个不同前缀。分歧点本身 j0 可比且最有趣，保留；之后的位置计数记为 skipped。
4. **自由运行** greedy 256 × 多 prompt 只做 top-N(8) 包容检查，且**把上游默认的 warn 改成 fail**（我们没有"换 GPU 就 skip"的退路）。
5. **无归因发散 ⇒ 停**，回退 env 门，报告。

**Actual（2026-09-10，G-P4 L3 PASS，四条件全绿；证据 `logs/p2/p2_l3.log`、`p2_l3_{off,on}.log`、`p2_l3_gate.log`、`p2_l3_ctrl.log`、`logs/p2/l3/`（含 `l3_{off,on}.json`、`p2_l3_verdict.json`、8 个 `p2hits_*.json`）**

1. **跑法与量级**：`p2_l3.sh` off→on 串行、每臂独立 `VLLM_CACHE_ROOT`、宿主钉频 1425/1000（manual，起跑前 `p0_clock.sh status` 核过）。off 臂 18:26:30→18:48:24Z（torch.compile 389.12 s），on 臂 18:51:24→19:13Z（402.89 s），臂间 `sleep 180`。4 prompt × 256 位置，pass 2b 探针 **279–596 ms/位置**（越到后面越慢 = 热降频，两臂同形不成对照差异）；共享 decode 位置 **1015**、共享 prefill 位置 **1090**。
2. **三条写在门"前面"的前提被真实数据证伪，全部就地改正**（本次执行最重要的产出，且都改在判据侧不是数据侧）：
   - **"两臂位置集合必须相同 ⇒ FAIL"结构性不可能**。drop 的判据是"该臂 prefill 路由在 t-1 的贪心 ≠ 强制序列"，而 on 臂的对照物正是 P2 的解码输出 ⇒ 两臂解码路线一在 t-1 分歧，该位置就只有一臂保留。**要求集合相等 = 要求换内核不改变任何行为**，是 hollow gate 的镜像。实测不对称只有 3 个位置/1021（p0 off_only=[218] on_only=[171]，p1 off_only=[152]）。改为：比较**交集**，交集 < min(两臂行数)×90% 才 FAIL，不对称逐 prompt 打印并写入 verdict（`forced_track_asymmetry`）——它本身就是行为分歧的证据，不该被静默吸收。
   - **"2a 跨臂对照应为 0 翻转"错了**：实测 **4 处**（/1090 = 0.37%），worst min-margin **0.0469 nats**，最差一处 **0.0**。这条对照两臂走的都是**未改动的** incumbent prefill 路由，翻转来源就是 `gptq_gemm_out` 的 atomicAdd 不可重现（L2 已量化：同臂重跑 71–72% 位置变动）。**没换的那条路翻得比换了的还多（4 vs 2）**。这不放松门，反而给 M0_swap 多了一把独立尺子：`swap_worst 0.0469 ≤ max(floor, ctrl) 0.0469` ⇒ 无论噪声带按"同臂 prefill-vs-decode"还是按"跨臂纯未变路由"定义，判定不变（`p2_l3_ctrl.py`）。对照依旧不进 `passed`。
   - **dispatch proof 第一版判据（`off == 0 且 on > 0`）会 FAIL 掉一对完全健康的臂**：实测 off 每 rank **60** 次命中。根因在 `wna16_exllama.py:395`——加载期预热**按模块名直调 `_p2_w4a16_gemm_impl`，完全不读 `VLLM_WNA16_P2_GEMV`**（它存在的意义就是把 HIP code object 拉到捕获区外加载）。正确的判别量是**轮数**：off = 1 轮 × 64 层 = 64（快照 60），on = 3 轮 × 64 = 192（预热 + size-1 图懒捕获自带的 warm-up + capture；快照 190），两臂族比例都精确是 **48:16**，而 60/64、190/192 正是"每 10 次 flush 一次"的截断。判据改为 `off > 0 且 on ≥ 2×off 且两臂各 ≥2 pid`：空转（on=off）与开关泄漏（两臂同时 3 轮 ⇒ on=off）**两种空洞情形都过不了**。
   - **顺带钉死记忆 #34 的口径**：warm-up+capture 之外确实不再产生任何命中——1024 次 pass 2b 探针全程没把 190 推高，文件 mtime 停在 03:02（自由运行中段的懒捕获时刻）。"Python 计数器看不见每步"成立，错的是我推论出"off 臂应为 0"。
3. **判定**：`[cross-margin] M0_swap 0.0469 <= M0_floor 0.0469 -> PASS`、`[floor established] yes`、`[top-8 inclusion] 0 violations -> PASS`、`[dispatch proof] off=120 on=380 -> PASS` ⇒ **G-P4 L3 PASS**。跨臂翻转 **2/1015 = 0.20%**，两处 min-margin **0.0469 / 0.0156 nats**（= 3/64 与 1/64，硬币级平票）；自由运行 4 条中 **1 条**（p0）在 token 98 发散，其后 157 位置按"前缀已不同"剔除，另 3 条 256 token 逐位相同。
4. **必须一起读的限制**：① **门是等号过的，不是余量充裕**——M0_swap 恰等于 M0_floor，带再窄一点就翻；缓解证据是 ctrl 里那处 0.0-margin 的未变路由翻转，根因是 margin 落在 1/64 的 fp16 量化格点上、样本只有 4 prompt ×256；② 判据修改发生在**数据落盘之后**，两条数值条件（cross-margin、top-8 包容）一字未动，动的是一条不可能成立的前提和一条对对照的误解，archive 里的 verdict 已用最终 gate 重生成（CPU-only 读 json，不碰 GPU）；③ `p2hits_*.json` 的 mtime 证明懒捕获在自由运行中段才发生 ⇒ 若将来把 pass 1 改成并发，dispatch proof 的轮数仍成立但**每步**证据只能回 G-P3 的 device trace；④ 一次 instrument-only 的错账：`p2_l3_run.py` 注释写"上限 ~128/rank"，真实 192——我就是被这行错注释引导写出 `off==0` 的。注释已改，**未为此重跑 GPU**（数值臂与计数器无关）。

- [ ] **Step 3: Commit（用户要求时才做）**

```bash
git add logs/p2/
git commit -m "p2: L2/L3 acceptance -- V3 two orders cleaner than incumbent baseline"
```

---

## Task 5: 进镜像两路径（G-P5）

- [x] **Step 1: 交付四件套**

⓪ **pre-flight（Task 1 现场遗留）**：开发期为跑门手装的 `/usr/local/lib/python3.12/dist-packages/gfx906/libp2gemv.so` 必须先 `rm -rf` 再 `docker commit` / 建镜像——交付路径会自己从 `patches/p2_w4a16_gemv.hip` 编一个同路径产物，留着的隐患不是重复而是**快照里那份可能与 fence 不同源**（Task 1 就发生过"失败同步 + 用旧源重建却打印成功标记"，见记忆 #19）。核验手段（**Task 2 收口时更正**）：比对的是**源 md5**（`patches/p2_w4a16_gemv.hip` vs fence 抽出的 `.hip`），**不是 `.so` md5**——hipcc 给每个编译单元打一个随构建路径变化的 `__hip_cuid_*` 戳，实测两份源 md5 完全相同（`af1535d6…`）的 `.so` 逐字节不同，只有 `__hip_cuid_*`/`__hip_gpubin_handle_*` 四行字符串不一样；拿 `.so` md5 当"同源判据"会把正常状态读成"不同源"而引发一次假排查。真要比产物，用 `diff <(strings a|sort) <(strings b|sort)` 看除 cuid 外是否还有别的差异。

① `cp p2_kernels_prod.hip docker/patches/p2_w4a16_gemv.hip`；② **作废**（Task 2 落地更正）：`docker/patches/wna16_exllama.py` 由 `prepare_build_context.sh:42` 从运行中容器 `docker cp` 生成，手写源只有 `container_src/` ⇒ 唯一要求是"容器里装的 = `container_src/` 版"，Task 2 Step 4 已用 md5 钉死；③ `.gitignore` 在白名单组补一行 `!p2_*.log`（现规则是 `*.log` 全局忽略 + `!p0_*.log`/`!p05_*.log`/`!p1_*.log` 放行，同形态补即可）；④ Dockerfile 两路径各加（rebuild 阶段 `COPY patches/gfx906_gemv_fix.pth` 之后）——编译后必须验符号，`.so` 编译成功但缺 `p2_gemv` 会让首次调用在引擎内炸（P2 评审 P1）。**落地形态与两处原稿差异**：路径写成 `${SP}` 变量（与同阶段其余层一致），且**先 `rm -rf ${SP}/gfx906` 再编**——`build-context/dist-packages.tar.gz` 会把打包那一刻容器里 `dist-packages/gfx906` 的全部内容带进 Path B，Path B 若不清就可能把**旧的/手装的** `.so` 当成"已存在"，与本小节 ⓪ 防的是同一件事（`Dockerfile:129`）：

```dockerfile
# always a build of patches/p2_w4a16_gemv.hip.
COPY patches/p2_w4a16_gemv.hip /tmp/p2_w4a16_gemv.hip
RUN rm -rf ${SP}/gfx906 && mkdir -p ${SP}/gfx906 && \
    hipcc -shared -fPIC --offload-arch=gfx906 -O3 /tmp/p2_w4a16_gemv.hip \
      -o ${SP}/gfx906/libp2gemv.so && rm -f /tmp/p2_w4a16_gemv.hip && \
    nm -D --defined-only ${SP}/gfx906/libp2gemv.so | grep -qw p2_gemv
# -w is load-bearing: without extern "C" the mangled _Z14p2_gemv_kernel... and
# _Z29__device_stub__p2_gemv_kernel... still contain the substring, so a plain
# `grep -q p2_gemv` passes on a .so whose ctypes lookup would fail (container-verified).
```

final 阶段（Path A）加同一存在性断言（.so 从 :rebuilt 继承，且下面两条静态自检链会复验），并清掉任何预烧的编译缓存。**理由已在 Task 3 执行期改写**：原稿写"env 开关被 dynamo 求值成常量，脏缓存会让 `VLLM_WNA16_P2_GEMV=0` 不生效"——修复后该开关在 custom op 的 impl 里被读取，是运行期杠杆，与缓存无关。仍然要冷缓存的真实理由是：镜像不能带着为**别的**模型/并行度/图配置烧出来的产物出厂，而缓存键又不覆盖 W4A16 内核选择（[[gfx906-env-pitfalls]] 第 4 条）：

```dockerfile
RUN rm -rf /root/.cache/vllm /root/.cache/vllm-w4a16-*
```

⑤ **build-context 陷阱（Task 2 执行期 2026-09-09 现查，别到 Task 5 才发现）**：`COPY patches/…` 只存在于 `rebuild` 阶段（`Dockerfile:77` 起），而该阶段的构建上下文是 **`docker/build-context/`**（`BUILD_GUIDE.md:92-93`），**不是 `docker/`**——`docker/patches/` 里放一份 `.hip` 并不会自动进上下文。`Path A`（`--target final`，`FROM vllm-gfx906:rebuilt`）根本不 COPY patches，所以这条只在重建配方路径上咬人。故 ① 必须配一行改动：在 `prepare_build_context.sh` 的 `[3/7]` 组里补 `cp -f "$HERE/patches/p2_w4a16_gemv.hip" "$CTX/patches/"`。另注意该脚本会用容器内容**覆写** `$CTX/patches/` ⇒ 复用旧 build-context 时，陈旧 `.hip` 可能静默顶掉新源，重跑后必须 `md5sum` 比对 `docker/patches/p2_w4a16_gemv.hip` 与 `docker/build-context/patches/` 那份（与 ⓪ 同源不同影，同一类失败）。

- [x] **Step 2: selfcheck 三条 chk**

```bash
chk "p2 GEMV .so present" test -s "$SP/gfx906/libp2gemv.so"
chk "p2 GEMV symbol exported" sh -c "nm -D --defined-only '$SP/gfx906/libp2gemv.so' | grep -qw p2_gemv"
chk "p2 dispatch wired" grep -q _P2_SHAPES "$SP/vllm/model_executor/kernels/linear/mixed_precision/wna16_exllama.py"
```

- [x] **Step 3: 流程**：`docker cp` 补丁进容器 → `prepare_build_context.sh` → Path B → Path A → 镜像内 `test -s` + md5 复验 → 真机 `gfx906-selfcheck`（含新 chk，全绿）→ 纯净容器 A/B 冒烟。

**Actual（2026-09-09）：六环全过 ⇒ G-P5 PASS。**

1. **pre-flight 五道 fence 全绿**：`p2_plan_selfcheck.py plan.md . wna16_exllama.py` ⇒ `loader / registration / warm-up / route / dispatch` 五个 **ok**（要求是"产线文件逐字子串"）。这一环是本轮补的债：Task 3 两次改码（`_p2_route` 装载期常量、M 判断搬进 op impl、op 改 `-> None`）之后 fence 没回写，第一次重跑就报 `DRIFT ['loader','warm-up','route']`。诊断工具 `p2_fence_diff.py`（按 fence 首末行在产线定位锚点再出 unified diff）留下的现场账在 `p2t5/fence_diff.txt`。**处置原则：改文档不改产线**——漂移是"计划不能当构建输入"的缺陷，不是代码缺陷。选择器同步改掉（route 现在用 `torch.ops.vllm.p2_gemv_or_fused(` 定位，`_P2_HITS.append` 已不属于它），并把 dispatch fence 拼进生成的 `p2_patch_blocks.py`，`verify_p2_blocks.py` 加一条 `hasattr(torch.ops.vllm, "p2_gemv_or_fused")`：让 G-P3 修复的契约（`-> None` + `mutates_args` 过 `infer_schema`）也进可执行检查。
2. **⓪ 已做且可逆**：开发容器 `dist-packages/gfx906` 先核完源 md5 才 `rm -rf`（现在 GONE），复原只需 `cp /work/p2t5/libfresh.so`。这样 Path B 里那层 `rm -rf ${SP}/gfx906 && hipcc …` 是镜像内 `.so` 的**唯一**来源。
3. **溯源闭合**：fence 抽出的 `.hip` ≡ `docker/patches/p2_w4a16_gemv.hip`（源 md5 `af1535d6…`，`cmp` 判 IDENTICAL）≡ 过 G-P0…G-P4 那批 `.so` 的源（滤掉 `__hip_cuid_*`/`__hip_gpubin_handle_*` 后 `strings` diff 只剩编译单元文件名）。即**进镜像编的就是过门的内核**，而 `.so` 逐字节不同是预期（cuid 随构建路径变），不是"不同源"。
4. **两路径都真跑**：Path B `[rebuild]` 执行了新 COPY+hipcc 层与含 P2 断言的自检链 ⇒ `vllm-gfx906:rebuilt` = `3db9747fd931`；Path A `[final 1..8]` rc=0 ⇒ `vllm-gfx906:latest` = `e806793fd238`。`prepare_build_context.sh` 首跑 rc=1（`cp: …/docker/patches/p2_w4a16_gemv.hip: No such file`——① 的那份只 scp 到了 `p2t5/`），`set -euo pipefail` 让它**响**而不是静默留下半成品上下文；补齐后 rc=0，并按 ⑤ 复验了 `$CTX/patches/` 与 `docker/patches/` 两份 md5 相同。
5. **全新容器（`p2-rc`，只挂 `/work` + 模型目录）内复验**：`wna16_exllama.py` md5 `0a146c02`（= `container_src/`）、`/opt/gfx906/selfcheck.sh` `b9cb6aa8`、`libp2gemv.so` 26 536 B、`nm -D --defined-only` ⇒ `0000000000005fb0 T p2_gemv`、`/root/.cache` 为空、`IMAGE_SO_MATCHES`；`gfx906-selfcheck` **23 ok / 0 FAIL**（含三条新 chk）。
6. **纯净容器 A/B 冒烟 = 15/15 PASS**（`p2_t5_ab.sh`，与 `p2_ab.sh` 同构、唯一变量仍是 `VLLM_WNA16_P2_GEMV`，跑在只挂 `/work` + 模型目录的 `p2-rc` 里，各臂一个独立 `/tmp/p2t5cache_ab_$arm`；判据复用 `p2_ab_gate.py`，`BASE=/work/p2t5`）。off 臂 **32.765 ms**（锚点 33.008±0.3 内 ⇒ 条件未漂移、比较有效）、`p2_gemv` launch **= 0**、incumbent **256.0/step/rank**；on 臂 **31.976 ms / 31.27 tok/s**（门 ≤32.2 / ≥30.6）、`p2_gemv` **64.0/step/rank**（窗口内 8 192 次）、incumbent 精确降到 **192.0**、两臂 shape CSV 各 **48+16**、**greedy 256/256 逐位一致 0 处分歧**。⇒ 本门真正买到的两条结论是"**kill-switch 在交付镜像里是运行期杠杆**"（不是被烤进产物）与"**交付镜像的分派率/收益不比开发态差**"：delta 0.789 ms vs G-P3 run 3 的 0.656 ms，两次设备侧账同档（wall 自带 ±0.2 ms 噪声，可信读数仍是设备侧 ~812–826 µs/step）。日志归档 `logs/p2/p2_t5_ab.log`（含门块）、`p2_t5_selfcheck.log`、`p2_t5_fence_diff.txt`。
7. **未做**：未 `docker commit` 开发容器（交付只走两路径配方），未 commit 任何文件。

- [ ] **Step 4: Commit（用户要求时才做）**

```bash
git add docker/ container_src/ .gitignore p2_gate0.py p2_route_gate.py \
        p2_plan_selfcheck.py verify_p2_blocks.py logs/p2/
git commit -m "p2: ship the whitelisted M=1 GEMV on both image paths"
```

---

## Task 6: 交付态复测 + 收口（G-P6）

- [x] **Step 1**: 交付态 serve（auto 时钟）batch1 tok/s：**门 ≥33.4**。口径：G-P0 写死 1.3× 线才 go ⇒ 承诺收益 = 1 148.7 µs ⇒ 理论 **33.653**（精确式给 +1.2527），门 33.4 = 容 **20%** serve 折损。旧稿的 33.2 **高于纯分派上限 33.16**（702.7 µs 口径），完美实现也过不了 ⇒ 已废弃；实测多少写多少。
  **Actual（2026-09-10）**：交付镜像 `:latest e806793fd238` + **全新容器** + **auto 时钟**（脚本硬守卫：`p0_clock.sh unlock` 后必须读到两卡 `level=auto` 才开测）+ 开箱 `gfx906-serve` + 与 `p05_serve_probe.sh` 同探针 ⇒ **on 臂 REQ2_WARM 33.3 tok/s**（DECODE256 三次 33.5/33.4/33.5 ⇒ mean 33.5）；就绪 690 s、`torch.compile took 387.51 s`、KV 67 356 tok、并发 4.11×。**门 FAIL 0.1**（用户裁定不放宽 ⇒ 照报）；33.3 落在该收益的算术顶 **33.18–33.28** 之上（`32.4/(1−819/30860)=33.283`；分母换成同机 off 32.3 则 33.178）⇒ 缺陷在门的定法（诚实账第 9 条由预测升为实测），**但"兑现 100%"这句话只能说在"设备侧收益全额转成吞吐"这个意义上**——相对 1.3× 承诺它是 71–72%（µs 口径）/ 80%（tok/s 口径）。
  **off 臂已补齐（08:20 续跑，一次成功 ⇒ 可重入缓解被验证）**：`READY_AFTER=430s`、`torch.compile took 63.68 s`（同容器 AOT 缓存命中）、KV 64 322 tok（3.93×）、`wna16_exllama_lines=2` ⇒ **REQ2_WARM 32.3 / DECODE256 32.5·32.4·32.3 ⇒ mean 32.4** ⇒ **同机双臂 delta = +1.0 tok/s = +3.1% = 930 µs/step**，比设备侧 812–826 µs 还多 13% ⇒ serve 侧无隐藏折损；off 复现历史 32.4 差 0.1 ⇒ 基线可复现，且"812–826 µs 从 1425 平移到 auto 档"这条遗留假设由**基线复现 + A/B 差值**两处同时验证。一处不对称必须随数字带走：**热态**（on 探针前 43 °C 冷卡，off 紧随 junction 106 °C 的 capture 后余温 65–70 °C），吃亏的是 off 臂 ⇒ 只会让 +1.0 偏保守。
  **主机两次静默掉电**（04:31 compile 期、05:12 off 臂启动后）：`journalctl -b -1` 无 shutdown 记录、`user.journal corrupted or uncleanly shut down`、全部容器同秒 `Exited (255)`，黑匣子（5 s sysfs）取到 **junction 峰值 107 °C > `temp2_crit` 105 °C、每卡 218 W 越过 190 W 读数**，与 [[gfx906-host-instability]] 同型 ⇒ 非 P2 因果；run 2 的黑匣子又读到 **106 °C / 265 W** 只是这次没跳闸，且 `power1_cap` 已等于 `power1_cap_max`=190 W ⇒ 宿主侧无旋钮可拧，交付态计时只能在冷卡窗口取。工具自伤三条见 [[gfx906-env-pitfalls]] #40/#41/#42：续跑时用**同名** `> p2_t6_serve.log` 把 run 1 的 driver 日志整份覆盖（on 臂 `READY_AFTER=690s` 只剩现场观察），记报告诚实账第 11 条 = 坑 #42。
- [x] **Step 2**: 报告 `docs/superpowers/plans/2026-09-09-p2-report.md`（结论先行：tok/s 前后对照、两族 µs、验收门逐条、遗留）。
  **Actual**：已写并随双臂收口修订（逐门表、G-P6 双臂表、**11 条**诚实账、遗留节里"交付时钟平移是假设"已升级为实测、现场账含两个 `:latest` ID 与全部 Task 6 取证文件）。
- [x] **Step 3**: 记忆 + QUICKSTART 更新，**触发条件写死**（P2 评审漏11）：**无论 G-P6 过与不过**，QUICKSTART 基线节与 [[final-goal-shippable-image]] 都记录实测 tok/s 与"是否含 P2"；只有**过门**才把新数字写成"交付基线"，未过门则记忆条目明确标注"P2 已进镜像/未进镜像 + 实测值 + 挂起原因"，不许留白也不许只在报告里。
  **Actual**：走"未过门"分支——`docker/entrypoint/QUICKSTART.md` §6 的 P2 条写成**同机双臂 A/B**（on 33.3 / off 32.3 ⇒ +1.0 tok/s = +3.1% = 930 µs/step，与设备侧 812–826 µs 对得上），**32.4 仍是引用基线**（因为门未过，33.3 只记作"含 P2 的实测值"），并写明默认开启 + `VLLM_WNA16_P2_GEMV=0` 退回；§5 另补一行**内核级 env 开关**说明（`VLLM_WNA16_P2_GEMV=0`、`VLLM_WNA16_EXLLAMA=0` 不走 `gfx906-serve` 参数而是直接透传给 python）；KV 那一条补进第三个读数 64,322（3.93×，warm 侧）。记忆条目 [[final-goal-shippable-image]]（Task 5 已进镜像 / Task 6 双臂 + 门字面 FAIL + 掉电取证）+ [[gfx906-env-pitfalls]] #40/#41/#42。**QUICKSTART 改了 ⇒ 已重跑 Path A**：新 `:latest` = `2fba84a62576`（`[final 1..8]` 全过，含 `/root/.cache` 泄漏断言），全新容器内复验 `/opt/gfx906/QUICKSTART.md` md5 `fb388833…` ≡ 仓库、`gfx906-selfcheck` **23 [ok] / 0 FAIL**（三条 p2 chk 全绿）、`libp2gemv.so` 符号 `T p2_gemv` 在。Path B 未重跑（`patches/`、`.hip`、dist-packages 都没变）。
- [ ] **Step 4: Commit + push（用户要求时才做）**。

---

## 门（任一不过即停）

| 门 | 判据 |
|---|---|
| G-P0 | multi-acc V3 在 graph 内 ≤60.8（8192）/ ≤57.0（7168）µs，**且同档 wiring 严格过 + fp64 ULP RMS ≤0.05**，1700/1000 锁频下测——不达 ⇒ P2 挂起，报告数据。**已跑（2026-09-09）：FAIL** —— multi-acc 全线 +1.0~+1.5 µs 是回归（已回退）；单累加器换 sk=10 后 8192 = 60.51 **PASS**、7168 = 58.21 **FAIL**（门 57.0）。聚合收益仍达 1.3× 承诺的 99.4%（1 142.1 / 1 148.7 µs ⇒ +1.245 tok/s）⇒ **用户 2026-09-09 批准改口径放行**：判据改为累计收益 ≥ 1.3× 线承诺的 99%，固定两族 sk=10 + 单累加器。限制须一路带进报告：**7168 逐族只有 1.274×，字面 1.3× 未达**。详见 Task 0 执行记录 |
| G-P1 | 生产 .so 独立门：wiring ≤5e-4 绝对 + fp64 ULP RMS ≤0.05（回归门）+ graph 计时与 Task 0 一致 + 生产 vs 原型逐位一致（或差异已归因）。**已跑（2026-09-09）：两族 PASS** —— wiring 1.22e-04、ULP RMS 0.0292/0.0205（= Task 0 同值）、graph 8192 = 60.40–60.45（−0.2%）、7168 = 58.09–58.88（**±0.7 µs 机器态双稳**，与代码路径无关：同轮 prod−proto ≤0.04 µs）、毒化重放写回 + replay 逐位稳定 + 与原型逐位一致全 True。附带修掉一个 4% 回归（`#pragma unroll 8` vs 两级展开，10/10 轮同向）。详见 Task 1 执行记录 |
| G-P2 | 路由矩阵全过 + 引擎 shape CSV 白名单形状真实出现 + `_P2_HITS` ≈64/step + capture 冒烟逐位一致。**已跑（2026-09-09）：单元侧 PASS** —— 实际形态是 **7 行 21 断言**（比原文五行多"探针行"与"capture 行"：没有探针行，所有负向断言在 `.so` 缺失时空过），21/21 绿；门有牙齿由四臂变异证实（M2 白名单丢 ⇒ 8 行红；M1′ 传 `K//2` ⇒ row B 精确红；M1 传参互换 ⇒ 硬 GPU fault；M3 去 `mutates_args="out"` ⇒ **门仍绿，证伪了本计划两处原文断言**）。incumbent 行改判据的原因见 Task 2 Step 2 Actual（`gptq_gemm_out` 自身不重现）。**未闭合的一半已闭合（Task 3）**：shape CSV 两 rank 各 48+16 复现、per-step 分派由 device trace 数得 `p2_gemv` **64.0/step/rank** 且 incumbent 精确 256→192。**门本身在 G-P3 两次改码后各重跑一遍，仍 21/21 绿**（含 M=2 回退、off-list、捕获毒化重放 `rewrote_poison=True aliases_slab=True`） |
| G-P3 | 引擎 A/B（两臂各一个独立 `VLLM_CACHE_ROOT=/tmp/p2cache_ab_$arm`、`ignore_eos=True`）：on 臂 ≤31.9 ± 0.3 ms 且 tok/s ≥30.6，且 greedy 分歧可归因（或逐位一致）。**已跑（2026-09-09）：run 1 FAIL ⇒ 修复 ⇒ run 2/3 各 PASS 15/15** —— run 1 的 device 侧证据是 `p2_gemv` launch **= 0**、incumbent 仍 256/step、两臂 step 差 0.003 ms，真因是 `apply_weights` 里的 `M == 1` 在编译态从不生效（eager 对照 64/step 证明内核与 wiring 无恙）；修复＝M 判断与 env 读取一起搬进 custom op 的 impl。终版代码（op 改 `-> None`）复跑：**off 32.776 / on 32.120 ms、31.13 tok/s、`p2_gemv` 64.0/step 两 rank、incumbent 256→192、greedy 256/256 逐位一致、别名警告 0**。可信收益 = **设备侧 812–826 µs/step = 承诺 1 148.7 µs 的 71–72%**（差额全在分母：被替换的 incumbent 在引擎内只值 72.0 µs/调用，微基准分母是 78.99/74.14），wall step 自带 ±0.2 ms 噪声。**后果见 G-P6 行** |
| G-P4 | L2 两条同时满足（RMS ≤0.05 且 ≤ incumbent 基线 1/10）；L3 满足"存在 M₀ 使 margin>M₀ 零翻转"，top-N(8) 包容改 fail。**已跑，两半都 PASS**。**L2 8/8**：P2 真实权重/真实激活 RMS 0.0118–0.0331、MAX 1 ULP、重跑 8/8 逐位相等；同操作数 incumbent RMS 29.7–94.1、MAX 6592、自身重跑 71–74% 位置变动 ⇒ INC/P2 ≥1301×。**L3 四条件全绿**：`M0_swap 0.0469 ≤ M0_floor 0.0469`（**等号，落在带边缘不是余量充裕**）、floor 已建立、top-8 包容 0 违例、dispatch proof off=120/on=380；跨臂翻转 2/1015=0.20%，自由运行 4 条中 1 条在 token 98 发散。**执行期改正三条写死在门前面的前提**：① "两臂位置集合必须相同"结构性不可能（on 臂的 drop 正是 P2 换 token 的结果）⇒ 改比较交集 + <90% 才 FAIL；② "跨臂 prefill 对照应为 0 翻转"被证伪（实测 4 处、worst 0.0469、最差 0.0，全部来自未改动路由的 atomicAdd 抖动，**没换的比换了的翻得多**）⇒ 它反过来成为 M0_swap 的独立尺子；③ dispatch proof 的 `off==0` 会 FAIL 掉健康臂，因加载期预热不读 `VLLM_WNA16_P2_GEMV` ⇒ 判据改为**轮数**（off 1×64、on 3×64，族比恒为 48:16）。详见 Task 4 Step 1/2 Actual |
| G-P5 | 两路径镜像 selfcheck 三条 chk 全绿 + 纯净容器 A/B 冒烟（env=0 在交付镜像里真的回退）。**已跑（2026-09-09）：PASS** —— Path B `:rebuilt`（`3db9747fd931`）的 hipcc 层与 Path A `:latest`（`e806793fd238`）的清缓存层都真执行；全新容器 `p2-rc` 内 `gfx906-selfcheck` **23 ok / 0 FAIL**（含 `p2 GEMV .so present` / `p2 GEMV symbol exported` / `p2 dispatch wired`），`.so` 由镜像自己编出（`nm -D --defined-only` ⇒ `T p2_gemv`）、`/root/.cache` 为空；同源链闭合：fence `.hip` ≡ `docker/patches/p2_w4a16_gemv.hip`（源 md5 `af1535d6…`）≡ 过 G-P0…G-P4 那批 `.so` 的源（滤 `__hip_cuid_*` 后 strings diff 只剩 TU 名）；**五道 fence 逐字对齐产线文件**（Task 3 改码后曾 3 处 DRIFT，已回写并给 route/dispatch 各立选择器），且 `verify_p2_blocks.py` 在**交付镜像内** `PLAN-BLOCKS-OK`（含新增的 `p2_gemv_or_fused` 注册断言）。**纯净容器 A/B 15/15**：off **32.765 ms**（锚点 33.008±0.3 内、`p2_gemv` launch **= 0**、incumbent 256.0/step/rank）⇒ on **31.976 ms / 31.27 tok/s**、`p2_gemv` **64.0/step/rank**、incumbent 精确 256→192、两臂 shape 48+16、**greedy 256/256 逐位一致**。收益 0.789 ms 与 G-P3 run 3 的 0.656 ms 同类（wall ±0.2 ms 噪声，交付镜像不比开发态差）。详见 Task 5 Step 3 Actual |
| G-P6 | 交付态 serve tok/s ≥33.4（承诺口径 = 1.3× 线累计收益 1 148.7 µs ⇒ 33.653，容 20% serve 折损；旧稿 33.2 高于纯分派上限 33.16，完美实现也过不了）。**⚠ Task 3 之后本门已高于实测上限**：按同一精确式，33.4 需要 **924 µs** 真实收益（`32.4/(1−924/30860)=33.400`），而引擎内实测只有 **812–826 µs/step** ⇒ 理论顶 **33.28–33.29**，即**零 serve 折损也过不了**（这与评审当初抓 G-P6 33.2>33.16 是同一类缺陷，只是这次是实测把分母压低造成的）。口径保留两点诚实注记：① 812–826 µs 是在 1425 MHz 钉频下测的，平移到交付时钟（auto，访存受限项基本不随 sclk 变）是**假设**而非测量；② **用户已裁定 G-P6 不放宽** ⇒ Task 6 照跑照报，实测值 + 本条归因一起进报告，不事后改门。**已跑（2026-09-10，双臂）：字面 FAIL 0.1 入档** —— 交付态同容器同探针 **on 33.3 / off 32.3 ⇒ delta +1.0 tok/s = +3.1% = 930 µs/step**（比设备侧 812–826 µs 还多 13% ⇒ serve 无隐藏折损），off 复现历史 32.4 到 32.3；注记 ① 的"平移是假设"由这对数字升为实测。明细见 Task 6 Step 1 Actual 与报告"交付态"一节 |

## 自查（作者，2026-09-09）

1. **范围**：M>1、其余三族、persistent kernel、FP8/attention 均明确不在本计划。**白名单外零行为变化**由 Task 2 路由门保证。
2. **原"执行时先读源码"两处已就地核实**（2026-09-09，交付容器 0.28）：`direct_register_custom_op` 签名为 `(op_name, op_func, mutates_args, fake_impl, target_lib, dispatch_key, tags)`、无 `**kwargs`、内部第一步 `infer_schema` 要求全标注（`inspect.signature` + `infer_schema` 复现）；import 路径必须是 `vllm.utils.torch_utils`（`vllm.utils` 顶层没有该符号，同目录 `rdna_hybrid_w4a16.py:434` 为模板）。**仅剩一处留给执行时**：`p05_install.sh` 守卫对"新目标路径"的登记方式（P1 缺陷 #1 的教训：守卫可能假阴性，FORCE=1 是 documented 逃生门）。
3. **类型一致性**：`p2_gemv(...) -> int`（0 成功 / −1 拒绝）在 .hip 的 `extern "C"` host launcher、shim 的 `argtypes`+`restype=c_int`、补丁 impl 三处一致（**内核 `__global__` 不带 `extern "C"`**，见 Task 1 Step 1 注）；splitK 现在只有 `_P2_SPLITK = 10` 一个定义点（Python 侧同一常量既把 `_p2_part` 分配成 `(10, N_dim)` 又传给 impl，故"缓冲行数 vs 传参"不可能不一致；`.hip` 拿到的是运行期参数，只能校验整除关系、不校验它等于 10），Task 0 之前这里写死三处 8 且互相必须一致——已消除该漂移面；白名单两形状与 P1 实测一一对应（partition 后口径）。
4. **已知薄弱处**：① ~~自定义 op 漏 `out` 会输出恒零~~ **已被 M3 证伪**（删掉 `out` 后门仍全绿）⇒ 列全 `["part","out"]` 是契约要求而非门保护的不变量，真实兜底是 row B/G 的位相等 + `aliases_slab`；② ctypes 在 capture 下的调用是普通 host 调用，launch 进 capture 流依赖传入的 stream 参数正确——**已由 row G 证实**（毒化被重放写回 + replay 位稳定 + 与 eager 逐位同），2026-09-09；③ `p2_prod_gate` 的 graph 计时若与 Task 0 差 >20%，先查 ctypes 路径每次调用的 Python 开销（host 端 ctypes ~µs 级，graph 内不计）；④ **依赖运行期形状的分支不能写在 dynamo 追踪区内**——G-P3 首跑 FAIL 的真因就是 `apply_weights` 里的 `M == 1` 在编译态从不成立（详见 Task 3 Actual）。缓解不是清缓存（原稿这条指错了机制），而是把判断写成 custom op impl 内的代码：impl 不被追踪，运行期 M 与 `VLLM_WNA16_P2_GEMV` 都照常生效。缓存卫生仍要，但对象是按模式分键的 root（`VLLM_CACHE_ROOT` 指一次性 scratch），且它保护的是"复用旧代码的产物"，不是"开关失效"。
5. **评审处置（P2 计划评审 2026-09-09）**：P0 七条全采纳并落改（含 P0-5/P0-6 数字重算：+0.76/+1.25/+3.18、1.3× = 1 148.7 µs、门 33.2→33.4、18.9→30.6）；P1 十二条全采纳；漏 11 条中 1/2/3/4/5/6/7/8/9/10/11 分别落进 Task 1 ④、Task 2 ③、Task 1 注④、Task 1 Step 3 ④、Task 4 两步、Task 5 Step 1 ③、Task 2 Step 2、Task 2 Step 2、Task 2 ① `_P2_HITS`、Task 2 Step 2、Task 6 Step 3。评审自身一处口径偏离已修正：它把 G-P6 按"纯分派收益 702.7 µs"定 33.15，但 G-P0 写死 1.3× 才 go，故按 1.3× 累计口径定门（33.4）。**另记一笔自我更正**：本文前一版把 1.3× 收益写成 +1.26 / 33.66，并称评审的 +1.25 只是"舍入边界差（精确 1.2557）"——容器内重算（`32.4/(1−1148.7/30860)`）得 **33.6527 ⇒ +1.2527**，评审的 +1.25 才是对的，那个 1.2557 是我算错的。
6. **计划可执行性实证（2026-09-09，交付容器内真跑，不是读码）**：把本文当构建输入——`p2_plan_selfcheck.py` 从 md 抽 fence（`.hip` + 三段模块级补丁 + 两段函数内片段，片段补 `pass` 后过 `compile()`），`hipcc -shared -fPIC --offload-arch=gfx906 -O3` 真编译，`verify_p2_blocks.py` 在容器内 import 抽出的补丁块、真注册 `torch.ops.vllm.p2_w4a16_gemm`、调一次并对照 fp64 零自由度参考。五臂全绿：`_P2_OK=True` / op 注册成功（P0-1、P0-2 的修法在真实 0.28 里成立）/ **ULP RMS=MEAN=MAX=0.0**（门 ≤0.05）/ 非法形状 rc 透传成 `RuntimeError: p2_gemv rc=-1` / 捕获重放**先毒化 out 再要求重放写回**且逐位一致。**这一轮抓到两条评审漏掉的 P0 级缺陷**：① `#include <cuda_fp16.h>` 在 ROCm 7.2.4 下 hipcc 直接 `file not found`（P1 能过是因为 torch `load_inline` 自带 shim `-I`，独立 hipcc 构建没有）⇒ 已改 `hip/hip_fp16.h`；② `nm -D | grep -q p2_gemv` 对 C++ mangled 的 `_Z14p2_gemv_kernel…`/`_Z29__device_stub__p2_gemv_kernel…` 同样命中 ⇒ `extern "C"` 丢了也照样"验符号通过"，已收紧为 `grep -qw`（Dockerfile 与 selfcheck 两处）。任何 fence 代码改动后重跑这条链即可复现（本机无可用 python，抽取与编译都必须在容器内）：

```bash
scp p2_plan_selfcheck.py verify_p2_blocks.py docs/superpowers/plans/2026-09-09-gfx906-p20-p2-narrow-kernel.md \
    x@192.168.10.11:/home/x/vllm-work/p2check/
ssh x@192.168.10.11 'docker cp /home/x/vllm-work/p2check/. vllm-gfx906:/work/p2check/ && \
  docker exec vllm-gfx906 bash -lc "cd /work/p2check && mv plan.md x.md && python3 p2_plan_selfcheck.py x.md /work/p2check && \
    hipcc -shared -fPIC --offload-arch=gfx906 -O3 p2_kernels_prod.hip -o libp2gemv.so && \
    nm -D --defined-only libp2gemv.so | grep -qw p2_gemv && \
    mkdir -p /usr/local/lib/python3.12/dist-packages/gfx906 && cp libp2gemv.so \
      /usr/local/lib/python3.12/dist-packages/gfx906/ && \
    HIP_VISIBLE_DEVICES=0 python3 verify_p2_blocks.py | tail -6; \
    rm -rf /usr/local/lib/python3.12/dist-packages/gfx906"'
```

末行须见 `PLAN-BLOCKS-OK`，且**最后那句 `rm -rf` 不能省**——临时 `.so` 若留在 `dist-packages/gfx906/`，`prepare_build_context.sh` 会把它 tar 进交付镜像。

**复跑记录（Task 2 收口，2026-09-09）**：fence 换成落地原文后重跑整条链，并给 `p2_plan_selfcheck.py` 加了第三个入参 = 落地文件，要求**四段补丁 fence 逐字是产线文件的子串**。这条漂移检查立刻抓出一处真实漂移：白名单注释里 fence 写 "so **Step 2** asserts them"，落地代码是 "so **Task 2** asserts them"（改 fence 侧，不动已验收的产物）。改后 `loader/registration/warm-up/route` 四段全 ok，抽出 `.hip` 与 Task 1 那份源 md5 相同（`af1535d67012d54e3a525e4686858855`），hipcc 现编符号齐（`grep -qw p2_gemv`），`PLAN-BLOCKS-OK`。**一处读数变化必须记账**：`[3] fp64 ULP` 现在打印 `RMS=0.0167 MEAN=0.0003 MAX=1.0`，而上面 §6 记的是 `0.0/0.0/0.0`——不是回归，是 §6 那次跑在 **sk 还是 8** 的 fence 上，sk=10 改了 `p2_reduce` 里部分分的相加顺序，代价是少量元素 1 ULP。门（RMS ≤0.05）仍 PASS。归因用的是排除法：脚本 md5 两次相同（`501f720b…`）、`.so` 的**源** md5 相同且两份产物只差 `__hip_cuid_*` 这类不参与数值的编译单元戳、输入由 `torch.manual_seed(0)` 钉死 ⇒ 唯一剩下的自变量是 `_P2_SPLITK`（8→10，它同时改变 `p2_reduce` 里部分分的个数与相加顺序）。日志：`logs/p2/p2_fence_recheck.log`。
