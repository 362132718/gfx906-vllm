# 交付镜像 v3：三源整合 · 性能最大化 + MoE 兼容（计划）

**日期**：2026-09-10 · **基线**：`vllm-gfx906:latest 2fba84a62576`（vLLM 0.28.0+rocm724 / torch 2.13.0+rocm7.2 / triton 3.8.0）
**参照**：① 本仓 `D:\gfx906-vllm` ② z100 作者三篇实录 ③ `David-Lzy/vllm-gfx906` fork（本地只读副本 `D:\gfx906-ref\`）
**核实标记**：`[实测]` 本轮核过；`[文档]` 来自仓库文档；`[未决]` 未定价，不可当事实。

---

## 0. 范围裁定（2026-09-10 已定，锁定工作量档位）

三个分叉**全部走推荐档**，因此本计划锁定在「**推荐档 ≈ 4–5 个工作日 / ≈12 h GPU**」。

| # | 分叉 | **裁定** | 含义 |
|---|---|---|---|
| D1 | MoE 做到哪一档 | ✅ **A：只做兼容性冒烟** | 主目标 = **`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`**（用户 2026-09-10 指定），跑通 TP1/TP2 + 贪心生成，走原生 **Triton WNA16 MoE**、**禁用 EP**，写进 QUICKSTART。**不承诺 MoE 性能**。z100 的 INT4 专家内核**不碰**。 |
| D2 | 1.34× 线索怎么追 | ✅ **A：只做零重编的 python A/B** | 搬 fork 的 `gfx906_gptq_wna16.py`（162 行）+ GDN 布局保持（15 行），env 门控 A/B。**不重编 csrc**。若复现不出 1.34×，记录为「不可归因（六混淆项）」结案。 |
| D3 | 大 M prefill 立不立项 | ✅ **A：先跑探针 → 已出结论：不立项** | 探针实测：候选路径比现役 ExLlama 内核 **慢 16%**（均值 0.84×）⇒ **关闭**，见 §4 P6。原「<15 TFLOPS」判据作废（那只是代理指标）。 |

> 被排除的档位（D1-B / D2-B / D3-B）不写进执行步骤，只在对应 P 段保留「重启条件」。
> 三个裁定都不阻塞 **P0–P2 必修/零成本项**，可以现在就动手。

---

## 1. 目标与验收定义

**「可交付使用的镜像」= 满足下列全部**：

1. `docker save` 一份 ~22 GB 归档，对方 `docker load` 后 `gfx906-selfcheck` 全绿。
2. 开箱 `gfx906-serve <model>` 一条命令起服务，**自动探卡数/TP、自动 backstop 关键 ENV**。
3. `QUICKSTART.md` 覆盖：起容器 / 自检 / 起服务 / 调用 / 默认参数理由 / 实测基线 / **模型兼容矩阵** / 排查。
4. **性能不回归**：2K decode ≥ 33.4 tok/s、8 并发 ≥ 66 tok/s `[实测]` 基线。
5. **模型兼容矩阵**（本轮新增，见 §5）：dense 混合线性注意力（现状）+ **MoE** + **MLA** 三类至少 smoke 可跑。

---

## 2. 三源资产总账（这是「三源取长补短」的全部内容）

### 2.1 源①：我们（不可回归的底座）

| 资产 | 内容 | 位置 |
|---|---|---|
| 交付配方 | 两路径 `Dockerfile`（Path B rebuild → Path A final）+ `prepare_build_context.sh` 七步 | `docker/` |
| **九个必要条件** | Tensile 156 文件 / 自编 RCCL 2.27.7 / TRITON_ATTN / NUM_SEGMENTS=64 / W4A16→ExLlama 桥 / capture-safe `gptq_gemm_out` / P0.5 skinny GEMV / P2 wave64 W4A16 / `.pth` rocBLAS 修复 | 见 handover §4 |
| 性能优化 | P0.5（+62.6%）、P2（+3.1%，env 可退） | `container_src/wna16_exllama.py`、`docker/patches/p2_w4a16_gemv.hip` |
| 开箱件 | `gfx906-selfcheck`（23 项）、`gfx906-serve`、`QUICKSTART.md` | `docker/entrypoint/` |
| 纪律 | 门/尺分离、L1/L2/L3 三级数值验收、「内核级数字不作验收」 | `docs/superpowers/plans/` |
| **缺口** | **无任何 MoE 适配**（全仓 MoE 关键词仅命中 `layers_utils.py` 的注释）`[实测]`；63 项未提交；`is_gfx906` 雷 | — |

### 2.2 源②：fork（0.27.2rc 基线 ⇒ **补丁不可直搬，只能搬机理**）

| 物件 | 载体 | 可搬性 | 成本 |
|---|---|---|---|
| `gfx906_gptq_wna16.py`（**162 行纯 python** MPLinearKernel） | `vllm/.../mixed_precision/` | ✅ 零重编 | 半天 A/B |
| GDN `_output_projection` 三维布局保持（15 行） | `vllm/.../mamba/gdn/` | ✅ 零重编 | 半天 A/B |
| SplitKV（~**546 行纯 Triton**，我们 507 → fork 1053） | `vllm/v1/attention/ops/` | 🟡 纯 Triton 但要实测 | 半天 |
| `csrc/rocm/skinny_gemms.cu`（gfx906 wvSplitK）、`csrc/.../q_gemm.cu`（**168 行 delta**，含 gfx906 INT8 FP32 累加） | `csrc/` | ❌ 要**重编** | 2–3 天 |
| prefill 重写 / 反量化→稠密 GEMM / 启动工程 / Tensile / RCCL | — | ❌ **不存在**（grep 确认）`[实测]` | — |

**fork 白送的三条结论（本轮最有价值的部分）**：

- **MoE 在 gfx906 上能跑**：`Qwen3.5 35B-A3B BF16 TP4` 的 text/image/JSON smoke 在 v0.28 上通过 `[文档]`。
- **EP 被否决**：`TP4+EP` 比 `TP4` 慢 **11.5–16.1%**，且 KV 容量 860,714 → 355,102（−58.7%）；`DP4+EP` 在 100K 契约下最紧的一卡算出 **负 1.6 GiB** 可用 KV ⇒ 装不下 `[文档]`。⇒ **MoE 走 TP，不走 EP**，省掉一整条死路。
- **方法学可直接抄**：`benchmark-protocol.md`（口径契约）、`positive-gain-retention.md`（正收益保留政策）、`evidence-lifecycle.md`、脏树拒绝构建。

### 2.3 源③：z100 三篇（**只有方法，无源码**）

| 方法 | 我们的状态 | 裁决 |
|---|---|---|
| **大 M「先反量化 → 稠密 GEMM」**（作者 M=8192 **3.22–3.58×**、8K 模型级 **+96.5%**） | 完全没做，prefill 分母已有（434/338/250） | 🟡 **先探针**（D3） |
| W4A16 wave64 内核（LDS 复用 / INT4 解包 / 小 M 权重复用） | 我们已独立实现 = P2 | ✅ 已对齐 |
| FP8 KV（容量 +58.8% 已实测） | 质量未定价 | 🟡 定价后进文档 |
| GDN 融合（scatter_add → 直写、变长因果卷积单内核） | 部分由我们 P0.5/GDN 回退覆盖 | 🟡 顺带 |
| 手写 HIP attention | 我们 16K 无塌方 ⇒ **已关单** | ❌ 不做 |
| MTP 投机 / custom all-reduce / FP8 权重 | 已判决关单 | ❌ 不做 |
| **MoE 局部内核**（`moe_align_block_size` 回退、小 top-k router、W4A16 专家 GEMV、packed INT4 专家路径） | 无源码，作者自称**未闭环** | 🟡 仅作 D1-B 的参考；验证模型 = 本地 `Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`（~7.9 GB） |

---

## 3. 避免重复造轮子：裁决表

| 来源 | 物件 | 裁决 | 理由 |
|---|---|---|---|
| 我们 | 底座 + 九条件 + P0.5 + P2 | ✅ **保留，不得回归** | 已验收 |
| 我们 | `is_gfx906` 雷（3 处断言 + 1 处病码） | 🔴 **必修** | 交付用户跑 MLA 必踩 |
| fork | `gfx906_gptq_wna16.py`（162 行 python） | ✅ **进 v3，做 A/B** | 零重编；是 1.34× 的候选解释之一 |
| fork | GDN `_output_projection`（15 行） | ✅ **进 v3，做 A/B** | 零重编；GDN 只占 2.6% ⇒ 顺带项 |
| fork | SplitKV（546 行 Triton） | ⏸ **挂起**，重启条件 = P1 实测 ROCM_ATTN 在 16K 赢 | 先测后搬 |
| fork | `q_gemm.cu` / `skinny_gemms.cu` | ❌ **二期**（D2-B） | 在 `csrc/` ⇒ 实质是再走一遍构建通道 |
| fork | EP 相关一切 | ❌ **不做** | 有硬证据（−12~16% + KV 为负） |
| fork | M-RoPE / KVarN / Triton 3.6 补丁 | ❌ 不做 | 只为其 Qwen3.6 形状 / 算术错（0.55× 非 0.7×） |
| Z100 | 大 M prefill 反量化→稠密 GEMM | ❌ **已关闭**（P6 实测 0.84×，比现役慢） | 他们治的是 **Triton** 基线；我们早有 ExLlama 桥，已比其基线快 3× ⇒ 无收益可取 |
| Z100 | 手写 HIP attention / custom-AR / MTP / FP8 权重 | ❌ 已关单 | 16K 无塌方 / PCIe 0 收益 / 两轮负收益 / 缺矩阵指令 |
| Z100 | MoE INT4 专家内核 | ⏸ 仅 D1-B 参考 | 作者自认未闭环 |

**一句话**：**fork 给「结论和方法学」，z100 给「方法」，我们给「可回放的配方」。三者交集里唯一能零成本直接搬的就是 fork 那两处纯 python。**

---

## 4. 执行阶段

### P0 必修：拔 `is_gfx906` 雷 —— ✅ **已完成并验证（2026-09-10）**

**改法（比原计划更规范）**：把该文件从「B 类·反向抽取」升格为 **A 类·正向安装**，即仓库侧留权威副本：

```
docker cp a2b-t1:$SP/.../triton_decode_attention.py  →  container_src/triton_decode_attention.py
                                                        （新增；md5 9ac1dc66… 与容器原文件一致）
   ↓ 在仓库里改
CONTAINER=a2b-t1 ./p05_install.sh container_src/triton_decode_attention.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_decode_attention.py
   ↓（注册门 L22–31 通过：Dockerfile:111 的 COPY 行含该路径；md5 双向校验 6f234fec…→a5a0ac3f…）
a2b-t1 容器内生效
   ↓  P7 时重跑 docker/prepare_build_context.sh（[3/7] L46–47 反向抽取）→ Path B COPY（L111）→ :latest
```

**内容改动**（`_decode_grouped_att_m_fwd`，原 506–510 行）：

```python
    BLOCK = 32
    if is_hip_:
        from vllm.platforms.rocm import on_gfx906   # 延迟导入：import rocm 会探测设备
        if not on_gfx906():
            BLOCK = 16
```

- ⚠ **注释里不能出现 `is_gfx906` 字面量** —— 第一版把病名写进注释解释，被验证脚本抓出
  （会让「不得含该串」的新断言失败）。此坑已写进代码注释本身。
- ✅ `on_gfx906()` 真名在**我们镜像**的 `vllm/platforms/rocm.py:367`
  （handover 写的 `:323` 是 fork 副本行号，已过期）。

**三处断言同步翻新（已改）+ 加强**：

| 位置 | 原 | 现 |
|---|---|---|
| `docker/Dockerfile.rocm_gfx906:152`（Path B） | `grep -q "is_gfx906"` | `grep -q "on_gfx906"` **且** `! grep -q "is_gfx906"` |
| `docker/Dockerfile.rocm_gfx906:221`（Path A，原 L220） | 同上 | 同上 |
| `docker/entrypoint/selfcheck.sh:42` | `grep -q is_gfx906` | `grep -q on_gfx906`，**新增 :43** `sh -c "! grep -q is_gfx906 …"` |

> 断言语义由「该串存在」升级为「新串存在 **且** 病串不存在」——这才真正证明补丁生效。

**新增 selfcheck 通用守卫**（`docker/entrypoint/selfcheck.sh`，`--no-gpu` 下也跑；导入失败降级为 `[warn]`）：
扫描 4 个被补丁 python 文件里全部 `current_platform.<attr>`，用 `set(dir(cp)) | set(vars(type(cp)))` 判存在性。
**禁 `hasattr`** —— 实测：`is_gfx906` 的 `in_dir=False` 但 `hasattr=True`（兜底 `__getattr__` 恒真）。

#### P0 验证证据（真机实跑，容器 `a2b-t1`）

`p0_verify.py`（已入库）：

| 步 | 结果 |
|---|---|
| ① 旧写法 | `current_platform.is_gfx906()` → **`TypeError: 'NoneType' object is not callable`** ⇒ 雷真实存在 |
| ② 新谓词 | `on_gfx906()` → **`True`**（gfx906 上 BLOCK 保持 32） |
| ③ 源码扫描 | `'is_gfx906' in source: False`、`'on_gfx906' in source: True` |
| ④ **端到端** | `decode_attention_fwd(...)` 以 GQA 形状（B=1,H=4,KVH=1,D=128,SEQ=8,SPLITS=2）**执行成功**，`o` 有限（norm 11.99）、`lse` 有限 ⇒ **`P0_VERIFY_PASSED`** |
| ⑤ Dockerfile 断言链模拟 | `DOCKERFILE_ASSERT_OK` |
| ⑥ 新版 `gfx906-selfcheck` | **全绿**（含新增两条守卫），`torch 2.13.0+rocm7.2 / vllm 0.28.0+rocm724 / triton 3.8.0` |

**注意（P7 要用）**：加断言行后行号位移 ⇒ Path B 断言块 **L146 起**、Path A 断言块 **L215 起**（原 L146 / L214）。

- `docker/fix_gfx906_block.py`（孤儿文件）本轮裁定：**保留不动**。
- `docker/entrypoint/QUICKSTART.md` 未提及该雷，**不需改**。

### P1 #63：attention 后端冒烟 —— ✅ **已结案（2026-09-11）· 结论：保持 TRITON_ATTN**

**机制级证据（代码 + 运行时，未再花 GPU）**：

| 后端 | prefill 内核 | decode 内核 |
|---|---|---|
| `TRITON_ATTN`（`backends/triton_attn.py`） | `context_attention_fwd`（:802） | `unified_attention`（:718） |
| `ROCM_ATTN`（`backends/rocm_attn.py`） | **同一个** `context_attention_fwd`（:351） | `chunked_prefill_paged_decode` |

arm B（`:latest` 全新容器，auto 时钟）实测日志给出运行时证据：
```
[chunked_prefill_paged_decode.py:433] WARNING Cannot use ROCm custom paged attention kernel,
falling back to Triton implementation
```
⇒ 我们的几何（head 256 / block 784 / gqa 5）下 `ROCM_ATTN` 的 decode **自己退回 Triton**；
prefill 两后端共用同一个 op。**没有理由期待 ROCM_ATTN 赢 ⇒ 交付默认不动**（与 handover §4 条件 3 一致）。

**arm A（TRITON_ATTN）decode 基线（auto 时钟，全新容器，`engine ready after 691.1 s`）**：

| ctx 请求 / 实际 | decode tok/s | ms/step |
|---|---|---|
| 2048 / 1822 | 33.83 | 29.6 |
| 8192 / 7282 | 32.41 | 30.9 |
| 16256 / 14450 | 30.37 | 32.9 |

与 handover §3.1 的 33.36 / 31.95 / 28.56 吻合 ⇒ 交付基线可复现。

> ⚠ **arm A 的 prefill 数字作废**（我踩了 Prefix Cache 陷阱，见下）。**decode 部分有效**。

#### P1-α ⚠ 本轮踩的坑（已修，纪律见 §6）

**Prefix Cache 陷阱**：同一 prompt 跑 `REPS=2` 取 `min()`，而 vLLM prefix cache 默认开 ⇒ 取到的是
「缓存命中、prefill 免费」那次 ⇒ prefill 虚高 **6–32×**（实测 2569 / 7739 / 7914 vs 基线 434.8 / 338.1 / 249.9）。
**这正是 z100 文章第三篇明确写过的坑，也是本项目第 4 次犯。**
修正版 `p1_prefill_ab.py`：`enable_prefix_caching=False` + 每请求唯一 nonce + 从 `prompt_token_ids` 读真实长度。
**note**：同一次跑里 decode **不受影响**（`dec=(req-prefill)/(n-1)`，两者同为缓存那次，相消）。

#### P1-β auto 档桶表（**未取，已降级为可选**）

本轮未产出 auto 时钟 16K 分桶 trace（原计划的顺带项）。理由：P1 已在机制层结案，
且期间发生一次**整机掉电**（见 P1-γ）。**重启条件**：若日后要重排 prefill/attention 优先级，再跑一次。

#### P1-γ ⚠ 整机掉电（09:07:48）—— **不是过热**（修正 handover §7 风险 1）

宿主重启：`uptime` 归零、全容器 `Exited (255)`、`dmesg` 启动序列、journal「uncleanly shut down」。
死时 arm B 正在 graph capture 之后的 mm warmup。

**但驱动采样推翻「热跳闸」假说**：死前最后 20 秒 **44–48 °C、857MHz 空闲时钟**（仅一次 1564MHz 尖峰）；
全会话最高 84 °C（那是 08:47 编译期）。⇒ **与持续负载/温度无关**，更像电源瞬态。

**结论**：`clockcap.sh` 钉频**不是防掉电手段**；真正的保险是「分离运行 + 日志落绑定挂载」
（本次 arm A 数据与 arm B 诊断日志全部幸存）。

- **参考头寸**：16K 的 attention = **15.9% 步时、只用到 9.0% HBM 峰值带宽 ⇒ 打满 = +16.8% 上限** `[实测]`。
  这是**上限**而非可达收益；机制证据表明换后端拿不到它。

### P2 #62：GDN readout 幅值探针（1 h，与 P1 同容器排队）

16K 满 ctx 采 cast 前 `max|·|`，**覆盖 prefill/chunk 段**；与 `--mamba-ssm-cache-dtype fp16` 能回收的 **1.81 GB/卡** 一起读，不许单独采纳容量收益。出数后决定是否装上游 #54146。

### P3 搬 fork 的两处纯 python（半天，零重编）

- (a) `gfx906_gptq_wna16.py` → 落 `$SP/vllm/model_executor/kernels/linear/mixed_precision/`；**必须 env 门控 + 默认关**起步。
  ⚠ 它的 `apply_weights` 也是 `ops.gptq_gemm(..., use_exllama=True)` ⇒ 与我们的桥**同机制**，A/B 才是唯一判据。
- (b) GDN `_output_projection` 保持三维布局 → 落 `$SP/vllm/.../mamba/gdn/qwen_gdn_linear_attn.py`（我们 820/833）。
- **验收 = 引擎级 tok/s + 逐 token 贪心一致**；不接受「离线张量对拍全快」（作者踩过 packed-view stride）。

### P4 FP8 KV 定价（半天）

容量已实测 **+58.8%**（非翻倍：hybrid 页对齐把 block 784 → 1568）。补质量（贪心 teacher-forced margin）+ 16K decode tok/s。
过关 ⇒ 进 QUICKSTART 作**显式可选项**，**不改默认**。⚠ 改 block=1568 会变 attention 几何 ⇒ 重跑 P1 门谓词。

### P5 **MoE 兼容性**（D1-A；半天）— 本轮新增

**主目标模型已定（2026-09-10 用户指定）**：`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`（ModelScope 同名仓库）。
理由：与在跑的 27B **同族**（`model_type=qwen3_5_moe`，沿用 Qwen3.5 MoE 架构类）、原生多模态、262K 上下文。
`Qwen1.5-MoE-A2.7B-GPTQ-Int4` 降级为**备选**（若 35B 因内存/内核起不来，用它单独隔离 MoE 内核路径）。

#### P5.0 已在交付镜像上**实测**的结论（2026-09-10，容器 `a2b-t1` / `vllm-gfx906:latest`）

| 检查项 | 实测结果 | 结论 |
|---|---|---|
| 架构是否认得 | `$SP/vllm/model_executor/models/qwen3_5.py` 含 `Qwen3_5MoeForConditionalGeneration`，vLLM `0.28.0+rocm724` | ✅ 不需新模型类 |
| AWQ 落在哪个 MoE 后端 | `TritonWNA16Experts._supports_current_device()=True`、`MarlinExperts._supports_current_device()=False`；优先级 `FLASHINFER_TRTLLM→MARLIN→BATCHED_MARLIN→**TRITON**→HUMMING→EMULATION`，前两项被 zero_point / 设备挡掉 | ✅ **选 TRITON（非 CUDA-only 的 Marlin）** |
| `moe_align_block_size` | 在 gfx906 上实跑通过（返回 3 元组） | ✅ 不需要像 z100 那样加回退 |
| `moe_sum` | 不是独立算子，是 `TritonExperts.moe_sum` 方法（`triton_moe.py:561`） | ✅ 无缺失算子 |
| `awq_gemm` / `awq_dequantize` | `vllm._custom_ops` 两个属性都在 | ✅ 符号存在 |
| group_size 兼容性 | 该模型 `group_size=32`；`AWQ_TRITON_SUPPORTED_GROUP_SIZES=[-1,32,64,128]` | ✅ 在支持表内 |

**量化结构（来自 config.json，对本项目极有利）**：`modules_to_not_convert` 把**全部** attention / GDN(`linear_attn.*`) / `shared_expert*` / `lm_head` / vision / MTP 都排除在 AWQ 之外 ⇒ **只有 256 个路由专家是 4-bit**，其余保持 fp16。
⇒ 交付镜像的两条路各管一半：**专家 → Triton WNA16 MoE；其余 fp16 线性层 → 我们的 P0.5 skinny GEMV**。

#### P5.1 仍未证明的（必须真模型实跑才知道）

1. **Triton WNA16 MoE 主内核**（`invoke_fused_moe_wna16_triton_kernel`）在 gfx906 上 JIT 编译 + 数值正确 + **速度**（这是本轮最大的未知量）。
2. **显存是否放得下**：权重 24.32 GB ÷ TP2 ≈ 12.2 GB/卡；16 GB × 0.90 = 14.4 GB 预算 ⇒ 只剩 ~2.2 GB/卡给
   KV + GDN 状态 + 图池。（粗估：全注意力只有 10 层、每卡 1 个 KV head ⇒ ~10 KB/token；GDN fp32 状态 30 层。）
3. **GDN 几何**：该模型 GDN 是 16 QK / 32 V heads、head_dim 128；我们补丁是**按 config 泛读**的
   （`linear_num_key_heads` / `linear_num_value_heads` / `linear_key_head_dim` / `linear_value_head_dim`），但只在 27B 几何上验证过。
4. **mRoPE [11,11,10]** 与视觉塔（我们已验证 vision，但 mRoPE section 不同）。
5. **MTP**：fork 实测负收益；本轮**默认不启用**，仅记录。

#### P5.2 必测项与产出

- ⚠ **勘误（2026-09-11）：TP1 不可行。** 24.3 GB 权重装不进 16 GB 单卡 ⇒ **必须 TP2 起步**。
  （原计划「① TP1 + `--max-model-len 8192` + `--enforce-eager`」这条做不到，已改。）
- 顺序：① **TP2 + `--max-model-len 8192` + `--enforce-eager`** 冷启动（最短暴露路径）→ ② 贪心生成连贯文本
  → ③ TP2 吞吐 → ④ `FULL_DECODE_ONLY` 图捕获（`--max-model-len 16384`）→ ⑤ 一次带图问答。
- 驱动已入库：`p5_moe.py`（分步 marker：load / backend_info / generate / decode，含**退化检测**
  `len(set(token_ids)) <= max(2, n//8)`）+ `p5_moe.sh`（全新容器 `p5-moe`，两档 spec：
  `eager8k:EAGER=1,MAX_LEN=8192` / `graph16k:EAGER=0,MAX_LEN=16384`）。
- 每次用 `docker exec -d … > /绝对路径.log 2>&1`（宿主有静默掉电史）；日志带 run 号。
- 探测脚本已入库：`p5_moe_probe.py`（复跑那 6 项镜像侧实测）。
- **产出**：QUICKSTART 加一节「MoE 模型」，写明**支持 / 不支持 / 推荐参数 / 实测 tok/s**。
- ⚠ **不承诺性能**：fork 在 **4×32 GiB** 上跑 35B-A3B **BF16** 只有 C1 36.04 tok/s；我们只有 2×16 GB 且要走 Triton INT4 MoE
  ⇒ **默认预期是「能跑、可能不快」**。若实跑 < 20 tok/s，如实写进 QUICKSTART，不硬凑。

#### P5.4 实测结果（2026-09-11）—— ✅ **MoE 在交付镜像上跑通了**

第一臂 `eager8k`（`EAGER=1, MAX_LEN=8192`，**时钟钉 level 4 / 1425MHz**，容器 `p5-moe` from `:latest`）：

| 步 | 结果 |
|---|---|
| `load` | ✅ **578.2 s**（TP2；6 个分片全过，`Loading weights took 34.89 s`）⇒ **24.3 GB 在 2×16 GB 上装得下** |
| `backend_info` | ✅ `{"moe_ternary_ok": true, "is_rocm": true}` ⇒ **Triton WNA16 MoE 后端确实生效**（不是 Marlin/CUTLASS） |
| `generate` | ✅ 32 token，`degenerate=False`，**输出连贯**：`"\n\n<think>\nHere's a thinking process:\n\n1. **Analyze User Input:**\n   - **Question:** Explain why the sky is blue."` |
| `decode` | ✅ **17.73 tok/s（56.4 ms/step）**，ctx ≈ 1.9K，**pinned 1425MHz** |
| 首 token 慢 | 3.65 tok/s（8.76 s / 32 token）—— 含 MoE Triton 内核首次 JIT，属预期 |

**对照**：现役 27B dense 在 auto 时钟是 33.4 tok/s ⇒ 35B-A3B **约慢一半**，与「能跑、可能不快」的预判一致。
按 1425→1690MHz 折算，auto 档大约 **21 tok/s** 量级（估算，非实测）。

`graph16k`（`FULL_DECODE_ONLY`）与 `novis8k`（`MM=0`）两臂随后自动跑，用来回答
「MoE + 图捕获是否兼容」与「视觉塔是否真的可用」。

#### P5.5 ⚠ 第一次尝试的失败 = **探针 bug，不是模型 bug**（已修）

首次跑 `eager8k` OOM：`Tried to allocate 128.00 GiB`。真实 traceback 指向**视觉编码器的 profile run**：

```
gpu_worker.py:519 profile_run → gpu_model_runner.py:6606 embed_multimodal
→ qwen3_vl.py self.visual(...) → qwen2_5_vl.py self.attn(...)
→ mm_encoder_attention.py _forward_sdpa(...)          ← 这里要 128 GiB
```

**根因**：`p5_moe.py` 漏了 `serve.sh:106–108` 的两个 flag ——
`--limit-mm-per-prompt {"image":1,"video":1}` 与 **`--mm-encoder-attn-backend TRITON_ATTN`**。
漏掉后者 ⇒ mm 编码器落到 `_forward_sdpa`，在 dummy profile 序列上要 128 GiB，引擎启动即死。
**教训：MoE/多模态冒烟的任何参数都必须逐条对齐 `serve.sh`，不能凭记忆写。**

#### P5.6 已知风险（2026-09-11 补）

- **编译缓存可能不按 attention backend 分键**：本项目已记录「vLLM 的缓存哈希不含内核选择」
  （`p0_step.py` 注释 + `serve.sh:24-30` 因此按 W4A16 模式分 `VLLM_CACHE_ROOT`）。换 attention backend 时
  可能复用错误的 AOT 产物。第一次运行时 `TRITON_ATTN` 的 `torch.compile` 耗时 **385.3 s**
  （日志 `Compiling a graph for compile range (1, 8192) takes 357.70 s`）⇒ 若第二臂**没有**重新编译，
  必须在独立 `VLLM_CACHE_ROOT` 下重跑该臂，否则数字不可信。
- 编译缓存跨臂共享只有在 `VLLM_CACHE_ROOT` 同一时才发生；P1 驱动没给每臂单独设 root，故需事后核对。

### P6 大 M prefill 探针 —— ✅ **已测完 · 结论：不立项（2026-09-10）**

> **原判据作废**。计划写的门是「QKV 形状 < 15 TFLOPS ⇒ 不立项」，但那只是**代理指标**。
> 实测后发现真正该问的是「候选路径 vs 现役内核，谁快」——所以补测了 incumbent，结论随之**改变**。

**探针**：`p6_tflops.py`（候选路径）+ `p6b_incumbent.py`（现役对照）。M=8192，三次独立测量波动 <0.3%。
负荷时钟口径：**hwmon `freq1_input` 峰值 1289 MHz**，最高结温 52 °C（`rocm-smi --showclock` 全程误报 860 MHz，**不可用作口径**）。
理论 fp16 峰值按 1289 MHz 折算 ≈ **19.9 TFLOPS/卡**。

#### 决定性对照（M=8192，单位 ms）

| 投影 | 形状 | **现役**（ExLlama W4A16） | 候选（反量化 + rocBLAS fp16 GEMM） | 加速比 |
|---|---|---|---|---|
| QKV | 8192×8192×5120 | **36.66** | 67.32 | **0.54×** |
| Gate | 8192×17408×5120 | **88.95** | 84.50 | 1.05× |
| Down | 8192×5120×8704 | **39.74** | 42.43 | 0.94× |
| | | | **均值** | **0.84×** |

⇒ **候选路径比现役慢 16%。不立项。**

#### 为什么 z100 报 3.22–3.58×、我们却是 0.84×

z100 的基线是 **Triton** 融合 W4A16（他们 M=8192 是 125.7 / 262.6 / 130.8 ms），
而我们手上是 **ExLlama 桥**（交付镜像九条件之 #5，36.7 / 89.0 / 39.7 ms）——
**我们已经比他们的基线快 3.4× / 2.95× / 3.29×**。他们的「反量化→稠密 GEMM」是治**他们**的瓶颈，
而我们早就绕过去了。这一项从「本项目最大单项」降级为**已关闭**。

#### 附带发现（都有独立价值）

1. **BF16 GEMM 在 QKV M=8192 上是 5.73 TFLOPS，比 FP16 的 10.37 慢 1.81×，甚至比 FP32 的 8.45 还慢。**
   ⇒ 交付镜像强制 `--dtype float16` 有了 GEMM 级独立佐证（Z100 报 2.8× 悬崖，我们在 GEMM 层量到 1.81×）。
2. rocBLAS 覆盖率不均：8192³ = **16.58 TFLOPS**（83% of 1289MHz 峰值），4096³ 只有 **9.90**，
   而 (8192,8192,5120) 只有 **10.39** ⇒ 现役 W4A16 内核在这几个形状上**反而比 rocBLAS 快**。
   ⇒ 想再压 prefill 也不该走「改用稠密 GEMM」这条路。
3. INT4→fp16 朴素 torch 反量化 1.2–2.6 ms（带宽下限 0.13–0.28 ms，差 ~9×），
   但相对 GEMM 的 41–89 ms **可忽略** ⇒ 「一次性反量化很贵」这个担心不成立，只是**没有收益可取**。
4. 负荷配置：sclk 1136–1289 MHz，结温 52 °C ⇒ 这台机器跑这个负载热余量充足。

**若要重启这一项**：重启条件 = 「现役 ExLlama 内核在某个 prefill 形状上被证明慢于 rocBLAS」。
按上面第 2 条，目前没有任何一个形状满足。

### P7 一次批量进镜像（用户裁定：**只发一次**）

- **纪律**：`entrypoint/` 每改一次 = 一整轮 Path B→A + 一次 22 GB 重导 ⇒ **所有文档 + 所有补丁攒成一次构建**。
- 登记链两类，**漏一处即构建失败**：
  - **A 类·正向安装**（源在 `container_src/`）：`./p05_install.sh`（**仓库根目录**，注册门 L22–31 在最前）→ `prepare_build_context.sh:41–58` → Path B `COPY`（L109–116）→ 断言 + `chk`。
  - **B 类·反向抽取**（源在活容器）：改 `$SP` → 重跑 `prepare_build_context.sh:46–47` → 同上。
  - ⚠ **Path A 不需要为新补丁加 COPY**（`FROM ${SNAPSHOT_IMAGE}` 继承 `:rebuilt`；只有 `layers_utils.py` 被额外 re-COPY，L197）。
  - 断言位置：Path B **L146 段** / Path A **L215 段**（P0 加行后位移，原 L214）。

#### ⚠ P7.1 链路陷阱（2026-09-11 实测发现，此前所有文档都没写）

**`dist-packages.tar.gz` 才是「其余一切」的入镜像通道。**

`Dockerfile:94–97`：`COPY dist-packages.tar.gz` → `tar -xzf -C /`。也就是说
**容器 `dist-packages/` 里的任何文件都会进镜像**，哪怕 Dockerfile 从不提它 ——
`qwen_gdn_linear_attn.py`（2054 行的 GDN 补丁）、`causal_conv1d.py`、`fla_ops/` 全走这条路。
L108–122 那 6 个 python 文件是**额外覆盖**（repo 权威/更新），L124–133 的 P2 `.so` 则是**刻意不用 tar 里的**（改成构建期重编，注释说明理由）。

由此产生两个必须记住的后果：

| # | 后果 | 处置 |
|---|---|---|
| 1 | **`p05_install.sh` 的注册门对 tar 覆盖的文件会误拒**（它按「Dockerfile 没 COPY ⇒ 进不了镜像」判断，对 tar 沿线的文件这个前提不成立） | 装这类文件用 `FORCE=1`，并在提交说明里写明理由（本次装 GDN 文件即如此） |
| 2 | **`[7/7]` 有陈旧陷阱**（`prepare_build_context.sh:84`）：**若 `build-context/dist-packages.tar.gz` 已存在就跳过重打**。此时容器里新装的改动**不会进镜像**，而 `[3/7]` 却会刷新 ⇒ 得到一个**半新半旧的构建上下文**，且不报错 | **P7 必须 `FORCE=1 bash docker/prepare_build_context.sh`**，然后验证新文件真的在 tar 里（见下） |

**P7 收尾硬检查（加到现有复验清单里）**：

```bash
# 1) 强制重打 tar（否则上面第 2 条会静默吃掉本次所有容器侧改动）
FORCE=1 bash docker/prepare_build_context.sh <container>

# 2) 证明改动真进去了：从 tar 里取该文件，比 md5
tar -xzOf build-context/dist-packages.tar.gz \
  usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
  | md5sum          # 必须等于 container_src/qwen_gdn_linear_attn.py 的 md5

# 3) 或更省事：起构建后的容器直接 md5sum 那份文件
```

> **可选改进（未做，留给 P7 决定）**：把 `qwen_gdn_linear_attn.py` 提升为 **A 类**
> （加 Dockerfile `COPY` + `prepare_build_context [3/7]` 各一行），就能让注册门与 md5 校验重新覆盖它，
> 也彻底摆脱「tar 是否新鲜」的不确定性。代价是 COPY 清单多一项、断言行号再位移一次。
> 本次**没有**动它（保持变更最小），改用 FORCE=1 + 上面的硬检查兜住。

### P8 交付收尾

- **#18 分发**：~22 GB（`docker save` 流字节）/ 对端解包 ~70 GB。
- **#24 干净机验收**：在**对方那台机**跑开箱 + selfcheck + 一次多模态 + 一次长 ctx + **一次 MoE**。
- **commit**：`docker/patches/p2_w4a16_gemv.hip` 仍未跟踪（`??`），是 Path B 编 `.so` 的**唯一源**，漏提交 = 配方不可回放。⏸ **需你明确授权**（一次授权不覆盖下一次）。

---

## 5. 模型兼容矩阵（交付镜像承诺什么）

| 层级 | 模型族 | 依据 | 交付承诺 |
|---|---|---|---|
| **T1 已验收** | Qwen3.8-27B AWQ W4A16（dense，48 GDN + 16 full attn，head_dim=256） | `[实测]` 端到端 | decode 33.4 / prefill 434.8 |
| **T2 本轮冒烟** | **MoE** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`（40 层 = 30 GDN + 10 full attn；256 专家 top-8+1 共享；**只有路由专家是 AWQ-4bit**，其余 fp16） | P5.0 已完成 6 项实测；P5.1/5.2 待真模型 | 「能跑」+ 实测 tok/s + QUICKSTART 一节 |
| **T2 备选** | MoE `Qwen1.5-MoE-A2.7B-GPTQ-Int4`（~7.9 GB） | 仅当 35B 起不来时用于隔离 MoE 内核路径 | — |
| **T2 本轮冒烟** | **MLA**（import-only，不需权重） | P0 修复后 | 不再 `TypeError` |
| **T3 文档声明** | 其他 W4A16 / GPTQ-Int4 dense | 结构同构 | 「结构同构可试，未定测」 |
| **T3 不支持** | FP8 / MXFP4 / NVFP4 **权重**；EP 拓扑 | gfx906 无矩阵指令 / fork 硬证据 | 明确写「不支持」，防用户试 |

---

## 6. 默认值分档（用户开箱拿到什么）

| 项 | 默认 | 依据 |
|---|---|---|
| `is_gfx906` 修复 | **开** | 纯 bug 修复，只可能变好 |
| P0.5 / P2 既有内核 | **开** | 已验收，各自 env 可退 |
| attention 后端 | **不动 TRITON_ATTN**，除非 P1 实测赢 | 交付默认必须有实测分母 |
| fork 的 `gfx906_gptq` / GDN 布局保持 | **关**，A/B 赢才开 | python 层但改了数值路径 |
| MoE / MLA | **可跑即写进 QUICKSTART**，不改默认拓扑 | 兼容性 ≠ 性能 |
| FP8 KV / `mamba-ssm-cache-dtype fp16` | **文档化可选项**，默认不开 | 质量未定价 |

---

## 7. 验收判据（v3 出厂，五条）

1. 全新容器 `gfx906-selfcheck` 全绿（23 项 + 新增守卫），**无 `/root/.cache` 泄漏**。
2. 纯净容器 A/B：每个新 env 开关 off 臂 **launch 计数 = 0**、on 臂符合预期，且**贪心输出逐位一致**。
3. 开箱 `gfx906-serve`：2K decode **≥ 33.4**（不得回归）、8 并发 ≥ 66、一次带图问答、一次 16K 长 ctx。
4. **MoE 模型能在同一镜像上起服务并生成连贯文本**（TP，非 EP）。
5. 对方干净机上一次成功拉起 + selfcheck ⇒ #24 结单。

---

## 8. 风险与回滚

| # | 风险 | 处置 |
|---|---|---|
| 1 | **宿主静默掉电**（双卡 init 期实测 102/106 °C · 255 W 后整机断电）`[实测]` | 一切 GPU 作业必须 `docker exec -d … > /绝对路径.log 2>&1`；日志名带 run 号 |
| 2 | **无回滚点**：63 项未提交 + `.hip` 未跟踪 `[实测]` | P8 提交（需授权）；这是唯一高风险项 |
| 3 | 满载降频（>100 °C） | 取性能数等冷卡 |
| 4 | `is_gfx906` 雷已在交付镜像 `[实测]` | P0 修 |
| 5 | **fork 的 1.34× 无法解释** `[未决]`：混淆项六个（模型 3.6/3.8、torch 2.11+7.2.1 vs 2.13+7.2.4、4×32 GiB vs 2×16 GB、100K vs 16K、128 vs 64 token、**他们不记录时钟**） | P3 先做零重编 A/B；B 只在 A 复现不出 1.34× **且** prefill 探针合格时才立 |
| 6 | MoE 走通用路径可能很慢 | P5 结论如实写进 QUICKSTART，不承诺性能 |
| 7 | KV 容量 −3.6% 漂移未定位 `[未决]` | 别把 KV 当常数做容量规划 |

---

## 9. 工作量与时间线

| 阶段 | 墙钟 | GPU | 风险 | 档位 |
|---|---|---|---|---|
| P0 地雷 | 0.5 h | 0 | 无 | 必修 |
| P1 #63 冒烟 + trace | 半天 | ~3 h | 低 | 必修 |
| P2 #62 探针 | 1 h | ~1 h | 无 | 必修 |
| P3 fork 两处 python A/B | 半天 | ~2 h | 低 | 零成本 |
| P4 FP8 KV 定价 | 半天 | ~3 h | 低 | 顺带 |
| **P5 MoE 冒烟** | **半天** | **~2 h** | **低** | **本轮新增（D1-A）** |
| P6 prefill TFLOPS 探针 | ~0.5 h | ~0.2 h | 无 | ✅ 已测完：**不立项**（0.84×） |
| P7 批量进镜像 | 1 天（含构建等待） | — | 中 | 必修 |
| P8 分发 + 干净机验收 | 看对方 | — | 中 | 必修 |
| **合计（D1-A / D2-A / D3-A）** | **≈ 4–5 个工作日** | **≈ 12 h GPU** | — | 推荐档 |
| 追加 D1-B | +3–5 天 | +10 h | 高 | 可选 |
| 追加 D2-B | +2–3 天 | +4 h | 中 | 可选 |
| 追加 D3-B | +1–2 天 | +6 h | 中 | 可选 |

---

## 10. 关键文件索引

- 配方：`docker/Dockerfile.rocm_gfx906`（Path B L81–176 / **Path A L182**–238；hipcc L129–133；COPY L109–116；断言 **L146 段 / L214 段**）
- 登记：`docker/prepare_build_context.sh`（**L41–58 = [3/7] 补丁登记**）、`./p05_install.sh`（**仓库根**，注册门 L22–31）
- 入口：`docker/entrypoint/{selfcheck.sh, serve.sh, QUICKSTART.md}`（selfcheck `chk` 习语 L15；**L42 是需翻新的断言**）
- 目标（容器内 `$SP=/usr/local/lib/python3.12/dist-packages`）：`vllm/v1/attention/ops/triton_decode_attention.py`、`vllm/model_executor/kernels/linear/mixed_precision/gfx906_gptq_wna16.py`、`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- 参照（只读）：`D:\gfx906-ref\vllm-gfx906-main`、`D:\gfx906-ref\gfx906docs`
- 证据：`logs/p0/p05_buckets_ab_on.log`（**首行 `pin=1425.0MHz`**，勿当 auto 分母）
- 前序计划：`docs/superpowers/2026-09-10-handover.md`、`plans/2026-09-10-gfx906-v2-three-source-integration-plan.md`、`plans/2026-09-10-gfx906-research-verification-and-replan.md`
- 历史遗留（保留不动）：`docker/fix_gfx906_block.py`（孤儿文件）
