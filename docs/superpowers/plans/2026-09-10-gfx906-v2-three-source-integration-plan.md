# 整合版交付镜像 v2：三源取长补短（勘误版）

> 本文是桌面版《整合版交付镜像 v2：三源取长补短.md》的**勘误版**。
> 勘误依据 = 2026-09-10 对工作区（`D:\gfx906-vllm`）的**逐条实仓核验**。
> 正文结构、裁决结论、排序与时间线**均不变**，只修正 4 处与仓库实况不符的事实描述（见 §0.0）。
> 代码未做任何改动。

## 0.0 修订记录（2026-09-10 核验）

核验方式：对 `docker/`、`container_src/`、`docs/` 逐条比对行号与内容（不依赖任何文档转述）。

| # | 原文档说法 | 仓库实况 | 影响 |
|---|---|---|---|
| E1 | §3 P0／§5：改 `container_src/` 里的 `triton_decode_attention.py` | **`container_src/` 内不存在该文件**（9 项：`_custom_ops.py` / `causal_conv1d.py` / `fla_ops/` / `ops_stable.h` / `q_gemm.cu` / `qwen_gdn_linear_attn.py` / `torch_bindings.cpp` / `wna16_exllama.py` / `wna16_exllama_repo_old.py`）。该文件走「活容器反向抽取」链 | 🔴 P0 按原文做 = 改了不进镜像 |
| E2 | §3 P0：`selfcheck.sh` **加**一条守卫 | 除新增守卫外，**三处既有断言把 `is_gfx906` 当作成功条件**：`selfcheck.sh:42`、`Dockerfile:152`、`Dockerfile:220`。改名后三处**必须同步翻新** | 🔴 漏改 = P5 构建 `rc≠0` + selfcheck FAIL |
| E3 | §5：`docker/p05_install.sh`（L22–27 注册门） | 实际在**仓库根目录** `./p05_install.sh`（注册门在 L22–27，内容正确） | 🟡 路径错一档，接手者找不到 |
| E4 | §3 P5：登记链「Path B COPY **和** Path A COPY（L197，**A 不继承新文件**）」 | Path A 是 `FROM ${SNAPSHOT_IMAGE} AS final`（L182），**继承** `:rebuilt`；`Dockerfile:194–196` 注释明说 base 的其它 python 文件来自 build-context。**只有 `layers_utils.py` 被额外从 repo re-COPY**（L197） | 🟡 说法与实况相反，会误导登记动作 |

附带发现（未改动，仅登记）：`docker/fix_gfx906_block.py` 是**孤儿文件**——全仓 `grep` 零引用（`Dockerfile*` / `*.sh` / `*.py` 均无），而它第 23 行写入的正是 `if is_hip_ and not current_platform.is_gfx906():`。
本轮裁定：**保留不动**，仅加注释标注为历史遗留，防止接手者误把它当成生效路径。

**核验通过（原文可信，不需改）**：`docker/patches/p2_w4a16_gemv.hip` 确为未跟踪（`??`）；未提交变更 = 8 modified + 55 untracked = **63**；`on_gfx906()` 真实存在（`vllm/platforms/rocm.py:323`，fork 副本）；`current_platform.is_gfx906` 确未定义（谓词不存在，`__getattr__` 兜底 ⇒ 调用即 `TypeError`）；只读副本 `D:\gfx906-ref\{vllm-gfx906-main, gfx906docs}` 在位；`docs/superpowers/plans/` 下两份 2026-09-10 文档在位；`handover.md` 尚未生成（只有 `plans/`、`specs/`）；`docker/patches/` 确只有 `layers_utils.py` + `p2_w4a16_gemv.hip` 两个真文件；断言行号 **L146 / L214** 与原文一致；`selfcheck.sh:15` 确为 `chk` 习语。

---

## 0. 交接文档（先读这节，可独立看懂；批准后另存为 `docs/superpowers/2026-09-10-handover.md`）

### 0.1 现在手上有什么

**一个已经能用的交付镜像**：`vllm-gfx906:latest`（ID `2fba84a62576`，~22 GB 压缩 / 解包态约 3×）+
构建中间态 `vllm-gfx906:rebuilt`（ID `3db9747fd931`）。配方在 `docker/Dockerfile.rocm_gfx906`（两条路径），
开箱命令是容器内 `gfx906-serve`，出厂自检 `gfx906-selfcheck`（最近一次全新容器全绿、脚本报 23 项通过）。
**它今天就可以发给同显卡（gfx906 / MI50 / Radeon Pro VII）的用户**，不需要等 v2。

跑通的东西：Qwen3.8-27B AWQ W4A16（64 层 = 48 GDN 线性注意力 + 16 全注意力，`head_dim=256`，
KV head 4 ⇒ TP2 每卡 2）、TP2、fp16、`--max-model-len 16384`、CUDA Graph（`FULL_DECODE_ONLY`，
capture sizes 1/2/4/8/16/24/32/48）、TRITON_ATTN（`VLLM_TRITON_ATTN_NUM_SEGMENTS=64`）、图文视频多模态、
贪心可复现。

**未进镜像的两样**：本轮的文档改动（QUICKSTART 的长上下文基线、`!` 病因行）只在工作区；
以及 **63 项工作区变更全部未 commit**（实仓核验：8 modified + 55 untracked），其中
`docker/patches/p2_w4a16_gemv.hip` 是未跟踪状态——它是 Path B 编 `.so` 的唯一源 ⇒
**这台机器一坏，配方不可回放**。这是当前唯一的高风险项。

### 0.2 性能基线（**口径必须一起读，跨口径比较是本项目最容易犯的错**）

| 指标 | 数 | 口径 |
|---|---|---|
| decode 2K / 8K / 16K | **33.36 / 31.95 / 28.56** tok/s | 交付镜像 + 全新容器 + 开箱 serve + **auto 时钟** + 贪心 64/256 token，每档 3 次取多次均值 |
| decode 64 token 单档（P2 A/B） | on **33.3** / off **32.3** ⇒ +3.1% | 同机同镜像同探针；P2 前是 32.4；再往前（P0.5 前）是 22.0；eager 时代 14.6 |
| prefill 2K / 8K / 16K | **434.8 / 338.1 / 249.9** tok/s | serve 口径（首次）；eager 老基线 442.8 / 346.6 / 235.0 ⇒ **graph 对 prefill 无系统性贡献** |
| 8 并发 decode 合计 | **66.3** tok/s（每流 8.3） | serve |
| KV 容量 | fp16 **67,356**（16K × 4.11 并发）/ fp8 **106,977**（× 6.53） | 同容器同配置；fp8 只 **+58.8%，不是翻倍**（hybrid 页对齐把 block 从 784 顶到 1568） |
| 内核分桶归因（W4A16 60.2% / RCCL 23.0% / elementwise 12.4% / GDN 2.6% / attention 1.6%） | — | ⚠️ **`pin=1425.0MHz` 钉频尺**（日志 `logs/p0/p05_buckets_ab_on.log` 首行写明）⇒ 所有百分比相对交付态**偏保守约 10%**，不得当 auto 分母 |

### 0.3 三个源各自给什么（这是"取长补短"的全部内容）

- **我们的（已在镜像里，不可回归）**：两路径可回放镜像配方 + 开箱自检 + 端到端多模态验收 +
  P0.5（M=1 稠密 fp16 GEMM 换出 Tensile 串行 K 循环，+62.6%）+ P2（8192/7168 两族 M=1 W4A16 GEMV 手写
  wave64 内核，+3.1%，env 可退）+ 156 个 gfx906 Tensile 补充文件 + 自编 RCCL 2.27.7 +
  一整套门/标尺纪律（G1–G8、L1/L2/L3 验收）。
- **fork（David-Lzy/vllm-gfx906，基线 0.27.2rc1.dev26 ⇒ 补丁不可直搬，只能搬机理）**，逐条核过：
  **能搬且零成本** = `_output_projection` 那 15 行纯 python（GDN 输出 norm 保持三维布局；我们侧
  `qwen_gdn_linear_attn.py:820/833` 结构逐位对齐、`RMSNormGated` 是 `CustomOp` ⇒ `forward_native` 现成）。
  **能搬但要实测才值得** = ~530 行纯 Triton SplitKV（门条件正好要 `head_size == 256`，就是我们）。
  **搬不动** = `LLMMB4` / `q_gemm.cu` gfx906 重调（都在 `csrc/`，且 CMake 要求单 arch 重编）。
  **没有的** = prefill、反量化→稠密 GEMM、启动工程、Tensile、RCCL（grep 确认为零）。
  **白拿的方法学** = mock-based 分派证明测试、router-lab A/B harness、"脏树拒绝构建"纪律。
- **Z100 三篇文章（只有方法、无源码）**：大 M "先反量化→稠密 GEMM"（作者 M=8192 3.22–3.58×、
  8K 模型级 +96.5%）、GDN 融合、FP8 KV、启动时间工程。**其中 prefill 那项是本项目里唯一可能比
  #20 全部阶段加起来还大的单项**，但我们必须先有同口径分母才立项。

### 0.4 已经判决关掉的（防止下一个人重做）

| 项 | 处置 | 一句话理由 |
|---|---|---|
| 手写 HIP attention（#21） | 🔴 关单 | 16K 实测只比 2K 降 14.4% ⇒ "长上下文必塌方"的前提对我们不成立；同头寸有"改一个 CLI 开关"的通道 |
| M-RoPE / 融合 QK-RoPE（#22） | ❌ 不做 | fork 的实现由 env 门控、只为他们 Qwen3.6 形状服务，我们无缺口证据 |
| KVarN | ❌ 不做 | digest 的"0.7×"是算术错（110/200 = **0.55×**），fork 40 篇文档零命中 |
| MTP 投机解码 | ❌ 不做 | 两轮实测负收益（作者也报 8K 27.3→6.9） |
| custom all-reduce | ❌ 不做 | PCIe P2P 上 0 收益 + 陈旧数据风险 |
| BF16/FP8 权重 | ❌ 不做 | gfx906 无矩阵指令；FP8 **权重** ≠ FP8 **KV**（后者可做，见 0.3） |
| 再从源码编 PyTorch/Triton | ❌ 废弃路线 | 已清退；现配方是"不重编 vLLM 扩展" |

### 0.5 未决（四条，全部是"先量再动"）

1. **#63**：`ROCM_ATTN` 冒烟（prefill **和** decode 都要覆盖）+ auto 档 16K 内核分桶 trace。
   已实测门谓词 `use_rocm_custom_paged_attention` 对我们 = `False`（Qwen3-0.6B 几何 = `True`）⇒ 不会碰 MFMA；
   残留风险是 KV stride 支路选择和 head 256 的 prefill 从未跑过。
   **16K 的 attention 头寸 = 15.9% 步时、只用到 9.0% HBM 峰值带宽 ⇒ 打满 = +16.8% 上限。**
2. **#62**：GDN readout 在 fp16 缓冲上的幅值溢出探针（症状"输出全是同一个 `!`"，上游 #54146）。
   必须覆盖 prefill/chunk 段；且与 `--mamba-ssm-cache-dtype fp16` 能回收的 **1.81 GB/卡** 一起读，不许单独采纳容量收益。
3. **1.34× 线索**：同为 TP2 / 27B AWQ，fork 报 44.74 tok/s、我们 33.3。按桶只能落在占 60.2% 的 W4A16，
   差别可能只是一个编译宏——**但混淆项有六个**（模型版本 3.6/3.8、torch 2.11+7.2.1 vs 我们更新的 2.13+7.2.4、
   4×32 GiB vs 2×16 GB、100K vs 16K、128 token vs 64 预算、**他们通篇不记录 GPU 时钟**）。
   便宜动作 = 在他们那台机之外、**我们这台机按他们的协议复测**，再决定要不要花一次重编。
4. **FP8 KV 的质量与速度**（容量已定价 +58.8%）。

### 0.6 操作路线（接手必读）

- GPU 主机：`ssh x@192.168.10.11`，模型在 `/home/x/models`（**绝对路径是容器相关的，换容器先 `ls`**）。
- **长 GPU 作业必须分离跑**：`docker exec -d … > /绝对路径.log 2>&1`——宿主有**静默掉电史**
  （双卡 init 期实测 102/106 °C · 255 W 后整机断电）。重启后 `docker start` 容器即回来。
- **`/sys` 在容器内只读 ⇒ 锁频只能在宿主做**；核时钟要读 hwmon，不是 `pp_dpm_sclk`。
- `docker commit` 必须带 `--no-pause`（默认 pause 会在 `/dev/kfd` 容器上和 containerd diff 死锁）。
- 往容器喂脚本一律 `docker exec -i`（少 `-i` 会吞 heredoc）；pkill 模式带括号防自匹配。
- 新增交付文件的登记**按 §3 P5 的两类走**（A 类正向安装 / B 类反向抽取；漏一处即构建失败）。
  改 `entrypoint/` 一次 = 一整轮 Path B→A + 一次 22 GB 重导 ⇒ **文档改动要攒批**。

### 0.7 阅读顺序（细节都在这三份里，别在本计划里找证据）

1. `docs/superpowers/plans/2026-09-10-gfx906-research-verification-and-replan.md` ——
   **本轮全部裁决与证据**（§1 逐条判决、§2 成本重估、§3 排序表、§4 FP8、§5 fork/上游核实、§6 长上下文实测）。
2. `docs/superpowers/plans/2026-09-10-gfx906-change-and-test-summary.md` —— 项目从头到尾的改动与测试总账（§5 是未来方向，已标注被本轮改判的条目）。
3. `docker/entrypoint/QUICKSTART.md` —— 交付用户看的那一份（§6 基线、§8 排障）。
4. fork 源码本地只读副本：`D:\gfx906-ref\vllm-gfx906-main`（+ `D:\gfx906-ref\gfx906docs`）。

## 1. Context（为什么做、要达成什么）

**现状**：交付镜像 `vllm-gfx906:latest 2fba84a62576` 已经可用（全新容器 selfcheck 23/0、开箱
`gfx906-serve` 端到端多模态验收过、P0.5 + P2 两个内核优化已进镜像、P2 有 env kill-switch 且实测能回退）。
基线：Qwen3.8-27B AWQ W4A16 / TP2 / 2×gfx906，2K decode **33.4 tok/s**，16K decode **28.56 tok/s**。

**本轮新增的事实（已核实，改变了可行性判断）**：
- fork 的 SplitKV 是**纯 Triton**、门条件要 `head_size == 256`（正是我们的几何），落在
  `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`（我们 0.28 = **507 行**，fork = 1053 行 ⇒ 差 ~546 ≈ 那份补丁）。
- fork 有一条**我们完全没有、且零重编可搬**的纯 python GDN 优化：`_output_projection` 保持
  `[tokens, heads, head_dim]` 三维布局过 `RMSNormGated.forward_native` 再 `flatten(-2)`（15 行、env 门控）。
  我们侧结构逐位对齐：`qwen_gdn_linear_attn.py:820 def _output_projection`、norm 调用在 **833**（fork 是 834），
  `RMSNormGated(CustomOp)` 在 `layernorm.py:172` ⇒ **`forward_native` 现成可用**，只有 `_output_projection` 一个改动点。
- fork 侧**不存在** prefill/反量化→稠密 GEMM、启动工程、Tensile 补充、RCCL 的任何内容（grep 已确认）
  ⇒ 这几项只能取 Z100 的**方法**，代码自建。
- 我们交付镜像里有一颗**真 bug**：`triton_decode_attention.py:509` 调用 `current_platform.is_gfx906()`，
  该谓词**不存在**（`__getattr__` 兜底 ⇒ 调用即 `TypeError`；行号随版本，fork 副本的兜底在 `platforms/interface.py:1131`）⇒ 不会返回可用的真值。
  我们跑 TRITON_ATTN 不碰它，但 **gfx906 上 MLA 模型只能选 TRITON_MLA** ⇒ 对交付用户是硬雷。
- ⚠️ **算术警报（写死进验收纪律）**：fork 自称 SplitKV 0.865 ms/层 × 16 层 = **13.8 ms/step**，
  而他们引擎级 44.74 tok/s = 22.3 ms/step、W4A16 又占 52.8% ⇒ **内核级与引擎级自相矛盾**。
  我们 16K 实测 attention 全部层合计仅 **5.55 ms/step（15.9%）**。
  ⇒ 任何"搬内核"的验收只能定在**我们自己引擎级 16K/32K 实测更快**，内核级数字不作数。

**目标**（用户裁定）：**攒成 v2 一次发**；范围 = **必修 + 便宜确定项**，在不大幅增加工作量的前提下做性价比选择；
新性能项**按风险分档**决定默认开/关。

## 2. 取长裁决表（进 v2 / 不进，附重启条件）

| 来源 | 物件 | 裁决 | 理由与成本 |
|---|---|---|---|
| 我们 | 两路径镜像配方 + selfcheck + QUICKSTART + P0.5 + P2 + 156 Tensile + RCCL 2.27.7 | ✅ 保留，不得回归 | 交付底座 |
| 我们 | §1.7 `is_gfx906` 地雷 | 🔴 **必修** | 20 min + **三处断言同步翻新（勘误 E2）** + 1 条 selfcheck 守卫；不加 = 交付用户踩雷 |
| fork | GDN `_output_projection` 布局保持（15 行 python） | ✅ **进 v2**（过 A/B 才默认开） | 零重编、改动点唯一、他们自带 179 行 bench 可借 |
| 我们 | #63 `--attention-backend ROCM_ATTN` 冒烟 + auto 档 16K 内核分桶 trace | ✅ **进 v2**（一次测量，双份产出） | 只改 CLI；顺带结掉"auto 时钟 P0 桶表"老遗留 |
| 上游 | #54146 一行补丁 + 50 行回归测试 | 🟡 **等 #62 出数再装** | 幅值有余量 ⇒ 记"不阻断"；逼近 65504 ⇒ 装 |
| fork | SplitKV ~530 行 Triton | ⏸ **不进 v2**，**重启条件**：#63 实测显示 ROCM_ATTN 支路在 16K 优于现 TRITON_ATTN | 内核级定价自相矛盾；先测后搬，别反过来 |
| Z100 | prefill 反量化→稠密 GEMM（作者 8K 模型级 +96.5%） | ❌ **二期** | 自建 1–2 天 + 必须先跑 fp16 `torch.mm` TFLOPS 探针（<15 ⇒ 先解决 rocBLAS） |
| fork | `q_gemm.cu` gfx906 重调（1.34× 线索） | ❌ **二期** | 在 `csrc/` ⇒ 实质等于再写一次 P2（2–3 天）；且 incumbent 变快会**吃掉 P2 的 930 µs/step**，不可叠加 |
| fork | `LLMMB4`（capture sizes 2/4/8） | ❌ **二期** | 同上通道成本；与 P2（只管 M=1）正可叠加，值得单独立项 |
| Z100 | 手写 HIP attention / custom-AR / MTP / FP8 权重 | ❌ 已关单 | 16K 无塌方 / PCIe 0 收益 / 两轮负收益 / 缺矩阵指令 |
| fork | KVarN / M-RoPE 融合 / Triton 3.6 `scf.if` 补丁 | ❌ 不做 | 0.55×（非 0.7×）/ 只为其 Qwen3.6 形状 / 我们不走那条路 |
| 我们 | FP8 KV（容量实测 **+58.8%**，非 2×） | 🟡 **只进文档不进默认**，质量+速度本轮定价 | 16 GB 卡的真实痛点是容量；但质量未验证，不能开箱替用户决定 |

## 3. 执行步骤

### P0 必修：拔掉 `is_gfx906` 地雷（0.5 h，无 GPU）

> **勘误 E1（2026-09-10 核验）**：原文档说「改 `container_src/` 里那份 `triton_decode_attention.py`」——
> **`container_src/` 内不存在该文件**。该文件属「反向抽取」型，与 `wna16_exllama.py` 那种
> 「`container_src/` 正向安装」型**不是同一条链**。真实链路如下：

```
活容器  $SP/vllm/v1/attention/ops/triton_decode_attention.py      ← 唯一真源
  │  docker/prepare_build_context.sh:46–47   docker cp（反向抽取）
  ▼
docker/build-context/patches/triton_decode_attention.py            ← 构建上下文（每次重生成）
  │  docker/Dockerfile.rocm_gfx906:111   Path B COPY
  ▼
vllm-gfx906:rebuilt
  │  Path A  FROM ${SNAPSHOT_IMAGE} AS final（L182）—— 继承，无需再 COPY
  ▼
vllm-gfx906:latest
```

- **正确动作**：进活容器改 `$SP/vllm/v1/attention/ops/triton_decode_attention.py` → **重跑
  `docker/prepare_build_context.sh` 反向生成** → 再走 P5 构建。
  改 `container_src/` 或只改 `docker/patches/` 都**不会进镜像**（后者会被 prepare 覆盖）。
- 内容改动：`if is_hip_ and not current_platform.is_gfx906():` →
  `from vllm.platforms.rocm import on_gfx906`（**已核验真实存在**：`vllm/platforms/rocm.py:323`），
  条件改 `if is_hip_ and not on_gfx906():`。
- ⚠️ **勘误 E2：三处既有断言把旧串当"成功条件"，必须同步翻新**（漏一处即 P5 构建 `rc≠0` / selfcheck FAIL）：

  | 位置 | 现内容 | 应改为 |
  |---|---|---|
  | `docker/entrypoint/selfcheck.sh:42` | `chk "…" grep -q is_gfx906 …` | `grep -q on_gfx906` |
  | `docker/Dockerfile.rocm_gfx906:152`（Path B 断言段） | `grep -q "is_gfx906" …` | `grep -q "on_gfx906"` |
  | `docker/Dockerfile.rocm_gfx906:220`（Path A 断言段） | `grep -q "is_gfx906" …` | `grep -q "on_gfx906"` |

  这三处断言的语义是「该补丁已生效」，改属性名后旧串消失 ⇒ 断言必然失败。**必须三点同改**。
- `docker/entrypoint/selfcheck.sh` **新增**一条通用守卫（`chk` 习语，`selfcheck.sh:15`）：
  对每个补丁文件里出现的 `current_platform.<attr>` 做 **`dir()`/`__dict__` 级存在性检查**，
  **不许用 `hasattr`**（兜底 `__getattr__` 下 `hasattr` 恒真）。
- 验证：进容器 `python -c "…import triton_decode_attention"` + 触发那行的最小调用；并跑一遍
  `TRITON_MLA` 的 import-only 冒烟（不需要权重）。
- 顺带（**本轮裁定：暂时保留不动**）：`docker/fix_gfx906_block.py` 是**孤儿文件**（全仓零引用），
  它第 23 行写入的正是本节的坏串。本轮**不删、不加注释、不改内容**，只在文档里登记，
  防止接手者误把它当成生效路径（P0 的生效路径见上方的反向抽取链）。

### P1 #63：后端冒烟 + auto 档 16K 归因（半天 GPU，分离跑）

- 全新容器、`docker exec -d` + 日志（宿主有静默掉电史）。两臂：现默认 TRITON_ATTN vs `--attention-backend ROCM_ATTN`。
- **冒烟必须同时覆盖 prefill 与 decode**：残留风险已换成 (i) `has_native_kv_cache_layout`
  （`chunked_prefill_paged_decode.py:24`，按 stride 判 packed）在我们 hybrid stride-padded 布局上会不会选错支路；
  (ii) `context_attention_fwd` × head 256 从未跑过。已实测 `use_rocm_custom_paged_attention` 对我们 = `False` ⇒ 不会碰 MFMA。
- 同一次跑里出 **auto 时钟 16K 内核分桶 trace** ⇒ 结掉姊妹文档 §4 第 5 条（现有桶表全是 `pin=1425` 尺）。
- **产出即决策**：赢 ⇒ 触发 SplitKV 重启条件（但留二期）；不赢 ⇒ 后端默认不动，把"gfx906 交付默认 TRITON_ATTN"
  这条从经验升级为实测。

### P2 #62：GDN readout 幅值探针（1 h，与 P1 同容器排队）

- 16K 满 ctx 贪心，采 cast 前 `b_o` 与 `b_h` 的 `max|·|`，**必须覆盖 prefill/chunk 段**（不只是 decode）。
- 与 `--mamba-ssm-cache-dtype fp16` 能回收的 **1.81 GB/卡**一起读 ⇒ **不许单独采纳容量收益**。
- 出数后决定 P3' 是否装 #54146。

### P3 搬：fork 的 GDN 布局保持（半天，零重编）

- 落点唯一：`$SP/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 的 `_output_projection`（我们 820/833）。
- **必须 env 门控 + 默认关**起步，沿用我们 P2 的写法（`container_src/*.py` → `p05_install.sh` → 容器 → 反向生成进 build context）。
  门：`current_platform.is_rocm() and envs.<我们的变量名>`，命名跟我们自己的约定而不是照抄他们的。
- 借 fork 的 `tools/gfx906/benchmark_qwen_gdn_output_norm.py`（179 行）改造成我们的 A/B；
  **验收 = 引擎级 tok/s + 逐 token 贪心一致性**，不接受"离线张量对拍全快"（作者踩过 packed-view stride 的坑）。
- GDN 桶现在只占 2.6%（pin=1425 尺）⇒ 期望值不高，**这是低成本顺带项，不是主攻**。

### P4 FP8 KV 定价（半天，质量 + 速度）

- 容量已实测：fp16 67,356 @ block **784** vs fp8 106,977 @ block **1568** ⇒ **1.588×**（hybrid 页对齐吃掉一半）。
- 补：贪心 64/256 token 双臂**输出一致性**（teacher-forced margin + top-N 包容，沿用 [[kernel-swap-acceptance-criteria]] 的 L2/L3 写法）+ 16K decode tok/s。
- 过关 ⇒ 进 QUICKSTART 作为**显式推荐的可选项**（"要更长 ctx / 更多并发就加这个 flag"），**不改交付默认**。
- ⚠️ 改 block_size 到 1568 会改变 attention 几何 ⇒ 需重跑 P1 的门谓词一次（防"换配置把别的支路打开"）。

### P5 一次批量进镜像（用户裁定：v2 只发一次）

- **纪律**：`entrypoint/`（QUICKSTART.md / serve.sh / selfcheck.sh）每改一次 = 一整轮 Path B→A + 一次 22 GB 重导
  ⇒ **所有文档 + 所有补丁攒成一次构建**，禁止中途试跑滚镜像 ID。
- **勘误 E3/E4：登记链分两类，且 Path A 不需要为新补丁加 COPY。**
  - **A 类·正向安装**（源在 `container_src/`，如 `wna16_exllama.py`）：
    `container_src/` 源 → `./p05_install.sh <repo-path> <容器绝对路径>` 装进容器
    （**脚本在仓库根目录**，非 `docker/`；L22–27 是"配方未 COPY 即拒绝安装"的校验门）
    → `docker/prepare_build_context.sh:41–58` 反向抽取 → `Dockerfile` Path B `COPY`（L109–116）
    → 断言 + `selfcheck.sh` 一条 `chk`。
  - **B 类·反向抽取**（源在活容器，如 `triton_decode_attention.py`）：
    直接改活容器 `$SP` 内文件 → 重跑 `docker/prepare_build_context.sh:46–47` → 同上 COPY / 断言 / `chk`。
  - ⚠ **更正（原文档"Path A COPY（L197，A 不继承新文件）"与实况相反）**：
    Path A 是 `FROM ${SNAPSHOT_IMAGE} AS final`（L182），**继承** `:rebuilt`（即 Path B 产物），
    `Dockerfile:194–196` 注释明说 base 的其它 python 文件来自 build-context。
    只有 `layers_utils.py` 因"repo 权威、不依赖上次 prepare 的新鲜度"被**额外** re-COPY（L197）。
    给 Path A 加 COPY 是**可选的加强写法**，不是必需项。
  - **断言位置**：Path B **L146 段** / Path A **L214 段**（两处都要覆盖新补丁的存在性；行号已核验）。
- 构建后复验（沿用 G-P5 的写法）：全新容器 `selfcheck` 全绿、md5 逐字对齐、**纯净容器 A/B 证明每个 env 开关真能回退**。

### P6 交付收尾

- **#18 分发**：~22 GB 传输 / 对端 ~70 GB（containerd 下 `.Size` = 压缩态 = `docker save` 流字节）。
  registry 推送 vs `docker save` 归档取决于对方网络。
- **#24 干净机验收**：在**对方那台机**跑开箱（不是我们这台），跑 selfcheck + 一段多模态 + 一次 16K 长 ctx。
- **commit**：`docker/patches/p2_w4a16_gemv.hip` 目前仍未跟踪（已核验 `??`）⇒ 它是 Path B 编 `.so` 的唯一源，
  **漏提交 = 配方不可回放**。
  ⏸ **需你明确授权**（一次推送授权不覆盖下一次）；提交用行内 `git -c user.name=362132718 -c user.email=362132718@users.noreply.github.com`，
  永不改 git config；推前扫密钥（文件名 + 内容）；只走 SSH remote。

## 3.1 步骤 ↔ 任务对照

| 步骤 | 任务 | 前置 |
|---|---|---|
| P0 地雷 | **#64** | 无（可立刻做，不占 GPU） |
| P1 #63 冒烟 + auto 16K trace | **#63** | 无（GPU，分离跑） |
| P2 #62 GDN 幅值探针 | **#62** | 与 P1 同容器排队 |
| P3 GDN output-norm 搬 + A/B | **#65** | P1/P2 出数后排 |
| P4 FP8 KV 定价 | **#66** | 独立 |
| P5 批量进镜像 + 出厂复验 | **#67** | #64/#63/#62/#65/#66 全部出数 |
| P6 分发 / 干净机验收 | 复用 **#18**、**#24** | #67 |

## 4. 默认值分档（v2 开箱拿到什么）

| 项 | 默认 | 依据 |
|---|---|---|
| §1.7 地雷修复 | **开**（纯 bug 修复） | 只可能变好 |
| P0.5 / P2 既有内核 | **开**（已验收，各自 env 可退） | 维持现状 |
| attention 后端 | **不动 TRITON_ATTN**，除非 P1 实测赢 | 交付默认必须有实测分母 |
| GDN 布局保持 | **关**，P3 A/B 赢 ⇒ 改开 + 留 env | python 层但改了数值路径 |
| #54146 | **看 P2 出数** | 幅值有余量就不动 |
| FP8 KV / `mamba-ssm-cache-dtype fp16` | **文档化的可选项**，默认不开 | 质量/溢出未定价 |

## 5. 关键文件

- `container_src/wna16_exllama.py`（P2 shim + custom op + `_P2_SHAPES` 白名单，作为 **A 类**新补丁的写法范本）
- `docker/patches/layers_utils.py`、`docker/patches/p2_w4a16_gemv.hip`（`docker/patches/` 目前只有这两个真文件）
- `docker/prepare_build_context.sh`（**L41–58 补丁登记（[3/7] 段）**、L19–21 活容器检查、`CTX="$HERE/build-context"` 在 L12）
- `p05_install.sh` —— **勘误 E3：在仓库根目录，不在 `docker/`**（L22–27 是唯一的注册门）
- `docker/Dockerfile.rocm_gfx906`（Path B rebuild stage L81–176 / Path A final stage **L182**–238；
  hipcc L129–133；**Path B 断言 L146 段 / Path A 断言 L214 段**；Path B COPY 段 L109–116；Path A 的 L197 只 re-COPY `layers_utils.py`）
- `docker/entrypoint/selfcheck.sh`（`chk` 习语在 L15；**L42 是需翻新的 `is_gfx906` 断言**）、
  `docker/entrypoint/QUICKSTART.md`、`docker/entrypoint/serve.sh`
- 目标文件（容器内 `$SP=/usr/local/lib/python3.12/dist-packages`，**就地覆盖**）：
  `$SP/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`、`$SP/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`、
  `$SP/vllm/v1/attention/ops/triton_decode_attention.py`
- 历史遗留（**保留不动**，勿当生效路径）：`docker/fix_gfx906_block.py`（孤儿文件，全仓零引用）
- 只读参照：`D:\gfx906-ref\vllm-gfx906-main\`（fork 源码）+ `D:\gfx906-ref\gfx906docs\`
- 文档：`docs/superpowers/plans/2026-09-10-gfx906-research-verification-and-replan.md`（本轮裁决全在此）、
  `docs/superpowers/plans/2026-09-10-gfx906-change-and-test-summary.md`

## 6. 验收纪律（三条硬规则，防重演本轮犯过的错）

1. **内核级数字不作验收**——只认我们自己的引擎级 tok/s（fork 的 0.865 ms/层与其 44.74 tok/s 自相矛盾，见 §1）。
2. **时钟口径先读日志首行**——`logs/p0/p05_buckets_ab_on.log` 是 `pin=1425.0MHz` 尺，
   交付态 auto 步 ≈30.0 ms ⇒ 那张表所有百分比偏保守约 10%，不得当交付分母（本轮已错一次）。
3. **任何内核/布局级改动必须过真实非连续视图 + changing-input 图回放**（作者栽过的 KV packed-view stride；
   我们的对应物是 hybrid stride-padded 页 + `FULL_DECODE_ONLY` 图回放）。
   另：SplitKV 那类每调用 `torch.empty` 中间缓冲的写法，在图捕获下是**已知失败模式**（我们 B3 吃过 `new_zeros` 的亏）。

## 7. 工作量与时间线

| 阶段 | 墙钟 | GPU 时间 | 风险 |
|---|---|---|---|
| P0 地雷 | 0.5 h | 0 | 无 |
| P1 #63 | 半天 | ~3 h（双臂冷启动 720 s + trace） | 低（只改 CLI，可回退） |
| P2 #62 | 1 h | ~1 h | 无 |
| P3 GDN 搬 | 半天 | ~2 h | 低（env 门控、默认关） |
| P4 FP8 定价 | 半天 | ~3 h | 低 |
| P5 批量进镜像 | 1 天（含构建等待） | — | 中（登记漏项即构建失败，已有清单） |
| P6 分发 + 干净机验收 | 取决于对方 | — | 中（22 GB / 对方磁盘 70 GB） |
| **合计** | **约 3–4 个工作日**（P6 另计） | **约 9–10 h GPU** | — |

## 8. 端到端验证（v2 出厂判据）

1. 全新容器 `gfx906-selfcheck` 全绿（现 23 项 + 新增守卫），且**无 `/root/.cache` 泄漏**（构建期断言已守）。
2. 纯净容器 A/B：每个新 env 开关 off 臂 **launch 计数 = 0**、on 臂计数符合预期，且**贪心输出逐位一致**（P2 已建立此判据）。
3. 开箱 `gfx906-serve`：2K decode ≥ 33.4 tok/s（**不得回归**）、8 并发 ≥ 66 tok/s、一次带图问答、一次 16K 长 ctx 完成。
4. P4 若判 FP8 可用 ⇒ QUICKSTART 那条推荐配置要在**新容器**上照抄可跑通。
5. 对方干净机上一次成功拉起 + selfcheck ⇒ #24 结单。
