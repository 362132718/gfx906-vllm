# #20 gfx906 手写 wave64 W4A16 decode GEMV 内核 — 设计

日期：2026-09-03（v2，已并入外部核验 `z100-vllm-gfx906-w4a16-gemv-spec-review.md` 的采纳项）
状态：设计逐节确认 + 外部核验逐条处置完毕，待用户复核 → writing-plans
交付归属：所有成果必须可进交付镜像（见记忆 `final-goal-shippable-image`）

## 0. 目标与非目标

**目标**：为 gfx906（wave64、60 CU、每 CU 64KB LDS、无 MFMA）手写一个 W4A16 **M=1** decode GEMV 内核，替代当前 ExLlama fused 路径在 batch1 decode 中的位置，把 batch1 decode 从实测 22.0 tok/s（冷卡）往 Z100 作者的 35.6 tok/s 方向推。**35.6 既不当验收线也不当天花板**，只当方向。

**非目标（本轮明确不做）**：
- M=2..8 的小批量矩阵路径（作者用 BATCH_ROWS=2 / ROWS_PER_WAVE=2 覆盖，留作 #20 之后的独立任务）
- M>50 路径（现在走 reconstruct→fp16 dense GEMM，本来就不该由 fused 内核做）
- attention 内核（#21）、M-RoPE（#22）、FP8 KV（#23）
- 追求浮点逐位相同 —— 允许 ULP 级差异。**"与现路径逐 token 等值"不再是验收判据**：换量化 GEMM 内核这一类改动，上游 vLLM 自己用的是 top-N 包容而非等值（已在本仓库 0.28 源码逐字核对，见 §3 L3 与 §8）。精度把关搬到 L2，对**无自由度的理想参考**做 ULP 计数。

**成功判据**：不承诺绝对 tok/s（用户 2026-09-03 决策"拿数据说话"），但有一道**数值化的开工门**（§1 P0）。这道门的作用已按外部核验修正：它实际上**几乎必过**（见 §1 的敏感性表），所以真正的决策点是 §1 的 **GPU busy ratio 路由表**，不是 1.15× 本身。

## 1. 阶段划分与集成方式

### P0 → P1 的顺序是硬性的，且 P0 时间盒 = 1 天

batch1 decode 22.0 tok/s = **45.5 ms/token**。按 checkpoint 的 400 个量化张量精确算（`Σ K·N/2` + 3.91% scales/zp）：**12.64 GB 量化权重 → TP2 每卡 6.32 GB**；实测 `BW_eff` = **821 GB/s**（保守值；手写 Triton 纯读实测上限 928，证据见 `p0_bweff.log`）⇒ **W4A16 流式地板 = 7.7 ms/token**，加 lm_head 1.6 ms ⇒ 全权重地板 ≈ **9.2 ms**，与实测 45.5 ms 差 **4.9×**。也就是说我们离带宽地板很远，**而这未必是 GEMV 的锅**。

**一个新发现的独立嫌疑（P0 必须单列一桶）**：`lm_head` 未量化（在 `ignore` 里），是 **248320×5120 bf16 = 2.54 GB**，TP2 后每卡每 token 仍要读 **1.27 GB**（按 821 GB/s 算 ≈1.6ms，占 3.5%）—— 而 **gfx906 没有 BF16 矩阵通路**（QUICKSTART 里明写"不要用 BF16 权重"）。它每 token 只算一行，却既吃带宽又可能走慢路径。这与 #20 无关，但很可能是同一笔 45.5ms 里的一块，**P0 分解里要单独看它的时间**（也是 #22/#24 的候选目标）。

P0 产出四件事，缺一不算完成：

- **(a) `BW_eff` —— 已实测（2026-09-04）**：远超 L2（4MB）的纯流式读探针（grid-stride，结果归约到不可优化掉）。结果 **928 GB/s**（手写 Triton 纯读，两卡、1 与 4 GiB 收敛）/ **821 GB/s**（`torch.sum` 独立实现），copy 读+写的 时间比 2.09 与之一致。**采用 821 作地板**（不取更美化 #20 的那个数）。外部先验 720–780 GB/s **被证伪**（低 7–21%），已弃用；标称 1024 仍只作对照。证据 `p0_bweff.log`。
- **(b) decode step 分解（五个桶）**：
  1. `sum(所有 kernel 时长)` = GPU busy
  2. 其中 **RCCL/通信 kernel 单独一项** —— TP2 下行并行需 allreduce 的是所有 row-parallel 投影：attn `o_proj`(16) + mlp `down_proj`(64) + GDN `out_proj`(48) = **128 次/token**。消息 = hidden 5120 × 2B = 10KB，PCIe 传输 ~0.9µs、延迟主导 ~5µs ⇒ ~0.64ms = **1.4%**，未被图捕获时约 **3%**。它本就会被算进 (1) 所以不会丢，但**不单独识别就没有归属**。
  3. `wall − GPU busy` = 气泡 / launch gap
  4. **CPU 侧 step 时间**（scheduler + IPC，需在 worker 侧取）
  5. wall

  (1) 还要按**组件**拆开报告，不能是一个黑数：`W4A16 GEMV 合计` / `attention` / `GDN 内核` / **`lm_head bf16 matvec`** / `RCCL` / 其余 elementwise-norm。P0 的产出不是一道门，而是这张组件表。
- **(c) GPU busy ratio = (1)/(5)** 路由表（真正的决策点）：
  | busy ratio | 结论 | 下一步 |
  |---|---|---|
  | < 0.6 | 瓶颈是 CPU/图覆盖/调度，**写任何 GEMV 内核都换不来 tok/s** | 不开 #20 内核；转配置与图覆盖 |
  | 0.6 – 0.8 | 混合成因 | 按 (b) 的占比决定，且先吃掉非内核的那块 |
  | ≥ 0.8 | GEMV / GDN / attention 是主要嫌疑 | 按 (b) 的组件占比在 #20 / #21 / #21b(GDN) 之间路由 |
  | 附加行 | 若 (2) 通信占比异常高（远超上面的 1–2%） | 查 RCCL 是否被图捕获、卡间走 xGMI 还是 PCIe（本机实测 `--showtopo`：link type = **PCIE、2 hops，无桥接器**） |
- **(d) 内核收到的真实 M —— 方法已因实测纠正**。原计划用 Python 钩子在 `apply_weights` 里记 M，但 **FULL 图捕获后 decode 不再执行 Python**，钩子只能看到 prefill/编译/捕获这些 eager 通道（实测：一次真实 serve 采到的 5 个形状全是 `m=14` 的 prefill，**没有任何 m=1 行**——这不能证明 M≠1，只能证明该仪器看不见）。所以真实 M 只能靠两条路之一证明：
  1. **eager 普查**（`EAGER=1`）：每步都走 Python，能拿到 batch1 decode 的 M 与完整形状普查；
  2. **C++ 侧计数**：在 `gptq_gemm_out` 里按 (M,K,N) 去重打一行日志 —— 这是唯一能覆盖**图回放**的手段，顺带验证 #20 需要的重建回路。
  **已实测关闭（2026-09-05，`EAGER=1` 普查 `/work/p0/wna16_shapes.eager.csv.*`）**：batch1 时内核**确实收到 M=1**，且每形状的调用次数与"7 次 decode step × 256 次投影"精确吻合（m=1 行为 112 / 448 / 448 / 448 / 336，对应 16·7 / 64·7 / 64·7 / 64·7 / 48·7）。同一次请求里还出现 **m=14（prefill）、m=48（显存探针批）、m=2048（prefill 分块）** 三种批 ⇒
- **dispatch 必须严格 `M == 1`**，任何 `M <= k` 的宽松写法都会把 prefill 打进为 M=1 优化的内核；
- 图模式不需要额外的 M 证明：1 就在 capture sizes 里且 size=1 图捕获成功（8/8），故 batch1 不会被 pad 到更大尺寸。

**1.15× 开工门保留但降级**：外部核验的敏感性表（GEMV 占比 40/60/70% → e2e 增益 1.22/1.60/1.94×）显示三种情形全过这道门，所以它否不掉什么。保留它是为了留档，**决策权重放在 (c)**。

**测量手段（已实测确认，不靠猜）**：`rocprofv3` 在交付镜像里**跑不起来**（`python: error while loading shared libraries: libdw.so.1`），因此 P0 用 **`torch.profiler`（kineto，CPU+CUDA activity）** —— 已在本机容器实测通过，能给出逐内核设备时间与内核名（含 Tensile 的 `Cijk_Ailk_Bljk_HHS_...`）。不往交付镜像里装 profiling 依赖。

### 与 ExLlama 的关系：按形状并存，不是全局替换

dispatch 在 `WNA16ExllamaKernel.apply_weights` 一侧决定：`M == 1` 且该 (K,N,group) 形状在白名单内 → 走新内核；否则保持现路径。白名单由 P0/L1 数据填充，不靠猜。回退是配置级而非代码级。

**硬约束（2026-09-04 实测，见记忆 `gfx906-env-pitfalls` 坑 5）**：`apply_weights` 位于 **dynamo 追踪区域内**。dispatch 必须是纯 Python 对 host 整数的判断 —— 不调用任何返回非 Tensor 的 torch API、不做 I/O，否则引擎初始化期就失败（我们已踩一次：`torch.* op returned non-Tensor`，栈却指向 GDN 模块）。因此**白名单命中在 `process_weights_after_loading` 里算一次、存成 per-layer 布尔**，热路径只做 `if self._use_w64 and M == 1:`，不在每次调用里查 dict / 比元组。

### 新算子（签名已按真实桥接布局修正）

```
gfx906_w4a16_gemv_out(
    Tensor!   c_out,     # [1, N] fp16，调用方预分配（沿用 _cbuf 槽位）
    Tensor    x,         # [1, K] fp16
    Tensor    qweight,   # [K/8, N] int32，ExLlama shuffle 后的 nibble 序
    Tensor    scales,    # [K/128, N] fp16
    Tensor    qzeros,    # [K/128, N/8] int32 —— 沿 N 打包，LSB-first，GPTQv2 编码
    int       group      # 128
)
```

- **qzeros 永不为空**：本 checkpoint 实测 `symmetric: false`、`zp_dtype: torch.int8`（非对称，带真 zero point）；而桥接层对**对称** checkpoint 会合成一份常数 zp（`wna16_exllama.py:163-174`）。两条来源在加载后归一成同一份 `[K/G, N//8]` 张量 → **内核不需要"无 zp"分支**，白名单 key 也不需要 `has_zp`（外部核验建议的这一条不必采纳）。
- **`g_idx` 传空**：桥接的 `can_implement` 已拒绝 actorder（`wna16_exllama.py:108`），内核无需处理置换。
- **zp 沿 N 打包 = 一个访问模式约束**：lane `l` 要的是字 `(n0+l)//8` 的第 `(n0+l)%8` 个 nibble → 64 个连续输出跨 8 个 int32 字，与 qweight（沿 K 打包、每 lane 同字）完全不同。**scales 与 qzeros 都是每 N 一列**，每进一个 group 读一次 64-lane 向量。
- 复用现有 `csrc/libtorch_stable` + `torch_bindings.cpp` 注册管线，与 `gptq_gemm_out` 同一扩展、同一"调用方拥有输出缓冲"约定（现路径 `ops.gptq_gemm_out(c_out, ...)` 已经是原地写并返回同一 slab，见 `wna16_exllama.py:254-258`）。

### 开关与回退

单一环境变量 `VLLM_WNA16_EXL_W64`（默认值按 §4 规则）。op 缺失或符号不匹配时 dispatch 静默退回 ExLlama 并打 warning，由 selfcheck 显式报告"缺失"，而不是让用户从 tok/s 反推。

### 构建成本

不新增构建 stage、不新增镜像层：新 `.cu/.hip` 只是现有 C 扩展里的一个翻译单元，在 dev 容器里 `setup.py build_ext --inplace` 编（实测改动→`.so` 落地是分钟级）。
**但必须写清：Path B（`--target rebuild`）根本不编译 C++**，它只解 `dist-packages.tar.gz` 再叠 Python 补丁 ⇒ 新 `.so` 进交付有两条且只有两条路：**Path A 的 `docker commit --no-pause`**（迭代期用这条），或**重打 tar**（`FORCE=1`，见 §4 与计划 Handoff 的护栏 1）。原写"秒级增量、rebuild 自动带上"是错的。

### 白拿的增益（两条）

1. 新内核不用 atomicAdd（每 lane 一个 fp32 累加器 → 直接 store）→ 消掉 ExLlama 在 gfx906 上的 1-ULP 跨运行非确定性。
2. **命中白名单的层可以省掉 `c_out.zero_()`**（现路径每次投影都先清 slab 才被 atomicAdd 累加）。真实分母（运行时实测）：400 个量化张量经 vLLM 融合后 = **256 次 GEMV/token**（16 qkv + 64 o/out 共用形状但分开调用 + 64 gate_up + 64 down + 48 qkvz），所以省掉的是 256 次 launch，按 1–2µs/launch 约 **0.3–0.6%**。**降级为记录项**：真实存在但不是收益来源。

### 真实形状集（**运行时实测**，2026-09-04；取代此前两版推导）

架构实测：hidden 5120、intermediate 17408、64 层（`full_attention_interval=4` ⇒ 16 层 full-attn + 48 层 GDN）、24 Q head × head_dim 256 且 **`attn_output_gate=true`**（q_proj 输出翻倍 = 12288）、4 KV head（k/v 各 1024）、GDN 48 value head ×128 + 16 key head ×128、vocab 248320。400 个量化张量 = 48(qkv)+16(o)+128(gate,up)+64(down)+144(linear_attn)。

**两处我推导错、实测纠正的地方**：
1. GDN 的 `in_proj_qkv` 与 `in_proj_z` **运行时融合成一个 `in_proj_qkvz`**（10240+6144 = 16384，TP2 → **N=8192**），不是两个形状；
2. attn `o_proj` 与 GDN `out_proj` **共用 (3072, 5120)**，白名单/自整表按形状去重 ⇒ **只有 5 个形状**。

| 算子 | K | N | waves | waves/CU | splitK（`240/waves` snap 到 1/2/4/8/16） | 次数/token |
|---|---|---|---|---|---|---|
| attn `qkv_proj` | 5120 | 7168 | 112 | 1.87 | **2** | 16 |
| attn `o_proj` + GDN `out_proj` | 3072 | 5120 | 80 | 1.33 | **4** | 64 |
| mlp `gate_up_proj` | 5120 | 17408 | 272 | 4.53 | 1 | 64 |
| mlp `down_proj` | 8704 | 5120 | 80 | 1.33 | **4** | 64 |
| GDN `in_proj_qkvz` | 5120 | 8192 | 128 | 2.13 | **2** | 48 |

**每 token = 256 次 GEMV**（此前写的 304 / 160 都不对），其中除 `gate_up`(64) 外 **192 次需要 Split-K** ⇒ 两阶段归约 = 每 token 多 ~192 次 reduce launch，总计 **~448 次 launch**。所以 §被否清单里"persistent + last-block 归约"不能默认推到 P2：**L1 必须先量 graph 内每次 reduce launch 的代价**，显著就直上 persistent。

**测量口径与残留未知**：上表来自一次真实 serve 的 **prefill 通道（m=14）**，(K,N) 可信；但**图回放里的 M 无法用 Python 钩子证明**（FULL 图捕获后 decode 不再执行 Python），所以"batch1 时内核真收到 M=1"仍待独立证据 —— 见 §1(d) 的改法。证据文件 `/work/p0/wna16_shapes.csv.*`。

**所有 K 都是 128 的整数倍**（3072/24、5120/40、8704/68）⇒ **内核不需要尾部 group 分支**，L0 也少一整类测试用例。

### 映射

前提：ExLlama 桥接已把权重存成 `[K/8, N]`，每个 int32 内部是 even/odd nibble 交错的 `shuffle_4bit_8` 序（`dst_bits = (idx//2)*bits + (idx%2)*16`）。**不重新排权重。**

- **并行轴 = 输出维 N**：1 个 wavefront（64 lane）负责 64 个**连续**输出，lane `l` ↔ 输出 `n0+l`。
- **流水轴 = K**：所有 lane 同步沿 K 前进，每步读一个 int32 行 → N 连续 ⇒ **256B 完全合并**（64 lane × 4B）。
- **激活按 K 段载入 LDS**：不整段放（K=17408 时 34KB 占掉 LDS 一半以上，且限制 CTA/CU）。每段 128 或 256 个 K，与 group=128 边界对齐；lane 用 **`ds_read_b32`**（或 `ds_read2_b32` 一次两个）广播读。**不要让每个 lane 各自去 global 读 x** —— L1 能兜住但会吃掉 L1 带宽。
- **zp 沿 K 恒定 ⇒ 整个 zp 项必须提升出 K 循环（本轮新增，P0 级）**。qzeros 只按 `(group, n)` 索引（实测：`weight_scale` 是 `[N, K/128]`，桥接归一后 `[K/G, N//8]`，**都不含 k**），故对固定 (n,g)：
  `Σ_{k∈g} (q−zp)·x = dot_q_g − zp_{g,n}·S_g`，`S_g = Σ_{k∈g} x_k`（与 n 无关 → **波内算一次**）。
  这是下面 4.36 预算下唯一能把 op/weight 从 4.25 压到 3.85 的手段；数值代价 ~2⁻²⁴·√128 ≈ 7×10⁻⁷ 相对，比 fp32 累加自身噪声还低两个量级。
  **写死形式 A**：`out += scale_g · (dot_q_g − zp_{g,n}·S_g)`（不采用形式 B `scale·dot_q − (scale·zp)·S_g`；两者代数相等但舍入不同，见 §3 L2 的 epilogue 规定）。
- **反量化与累加（指令级约束）**：
  - 只用 **`v_dot2_f32_f16`**（两对 fp16 乘积 → fp32 累加，packed 2× 速率）；**禁止 `v_pk_fma_f16`**（fp16 累加）。fp16 累加的实测 RMS_ULP 是 **50–513**，而门是 0.5。
  - 依据（clpeak on MI50）：half2 24971 GFLOPS vs 标量 half 6523 → 标量 fp16 毫无加速。
  - 整数乘法走 **`v_mul_i32_i24`**（满速）；32 位 int mul 只有 ~1/3 速率。优先移位/掩码/`v_lop3`。
  - 编译期常量写死 `constexpr int WAVE = 64;`，**不要**依赖 `__AMDGCN_WAVEFRONT_SIZE__` / `warpSize`（ROCm 7.x 已非 constexpr）。
- **写回**：每 lane 恰好一个输出 → 直接 store，无跨 lane 归约、无 atomic。

### VALU 预算：地板与天花板几乎重合（2026-09-04 用实测带宽重算）

预算（用**实测** `BW_eff` 推，不再用假设值）：60 CU × 1.725 GHz × 64 lane = **6.62×10¹² lane-op/s**。
- 对保守地板 **821 GB/s**：821/0.5 = 1.64×10¹² weight/s ⇒ **预算 4.03 op/weight**，而提升 zp 后要 3.85 ⇒ **余量仅 ~4.7%**（外部核验按 760 GB/s 算出的 13% 偏高，因为我们的实测带宽更高）。
- 对纯读上限 **928 GB/s**：预算降到 **3.57** ⇒ **3.85 > 3.57，内核会被 VALU 绑死**，拿不到读带宽。

按每 int32 字（= 某输出 n 的 8 个 K 位置）逐条数：

| 步骤 | 逐元素减 zp | 提升 zp（采用） |
|---|---|---|
| `v_bfe_u32` 提 nibble ×8 | 8 | 8 |
| 打包 + fp16 magic + `pk_sub` | 16 | 16 |
| 减 zp | 4 | 0 |
| `v_dot2_f32_f16` ×4 | 4 | 4 |
| `S_g` 波内归约摊到 16 字 | 0 | 0.8 |
| 循环/寻址（摊） | 2 | 2 |
| **合计 / op-per-weight** | **34 / 4.25** | **30.8 / 3.85** |
| **VALU 天花板等效带宽** | **780 GB/s** | **860 GB/s** |

⇒ **把 VALU 天花板那一行与实测带宽直接对比，结论是硬的**：
- **不提升 zp（4.25 op/w）天花板 780 GB/s < 实测保守地板 821** ⇒ 这一版**结构上出局**：还没到带宽地板就先撞 VALU，任何优化都救不回来。提升 zp 不是"更快一点"，是**唯一能过线的写法**。
- 提升后（3.85 op/w）天花板 **860 GB/s**：高于保守地板 821，低于纯读上限 928 ⇒ **内核注定是 VALU-bound，约差 7% 拿满读带宽**。因此每一处多余指令都 1:1 变成带宽损失 —— 循环/寻址那摊的 2 条/字（占预算 52%）和 nibble 提取方案的选择，比点积本身重要得多。
- **含义修正**：#20 的胜负不在"离带宽地板还有多远"，而在**能不能把 op/weight 压到 3.85 以下**。所以 §3 L1 的"实测 op/weight"不是记录性指标，而是主 KPI。
**⇒ 主战场是 nibble 提取 + 转换（24 条/字），不是 dot2（4 条/字）—— 差 6 倍。** 原设计把优化注意力全放在 packed 点积上是放错了地方。候选：16 项 fp16 LDS LUT（1 op/nibble ⇒ 8 条/字替掉 24 条，但要验 bank conflict —— 64 lane 取 16 个地址会落在 8 个 bank 上、2-way 冲突）→ **由 L1 决定，不预设**。
L1 因此必须**同时报"距 `BW_eff` 还差多少"和"实测 op/weight"**两个数，否则无法区分"带宽受限"与"VALU 受限"，扫出来的最优变体也解释不了。

### 其他
- **延迟隐藏预算**：`BW × ~600ns` ⇒ 在飞 ~450KB/卡 ⇒ 每 CU 需 8–16 条独立 load 在飞 ⇒ unroll 8–16、每 CU 常驻 ≥4 wave；VGPR 预算先算（256KB/CU ÷ 64 lane ⇒ 8 wave 时每 lane 64 VGPR 上限）。
- **元数据读放大**：scales `[K/128, N]` fp16 = qweight 的 **3.125%**；qzeros `[K/128, N/8]` int32 = **0.78%**（原 spec 漏算，外部核验给的 1.56% 是除法错了一位）⇒ 合计 **3.9%**。Split-K 会把这两份重复读 splitK 次 → 地板分母按真实字节算（§3 L1）。

### 被否清单
- 重新排权重成 K-major tile 序：牵动加载路径与 `cbuf` 复用，收益未证 → 否
- 内核内分配 workspace：破坏 graph 捕获 → 否
- atomicAdd 归约（ExLlama 现做法）：非确定性 → 否
- M≤8 通用 tile：LDS/寄存器压力完全不同，会拖死 M=1 调优 → 拆后续任务
- 逐元素减 zp（不提升出 K 循环）：4.25 vs 预算 4.36 = 零余量 → **否，强制走提升形式**
- ~~persistent + last-block 归约"暂缓到 P2"~~：**改为 L1 实测后决定**（Split-K 是默认路径 ⇒ 每 token ~240 次 reduce launch 是真实分母）

## 3. 验证与度量

### L0 内核前：把 nibble 映射钉死（新增，采纳自外部核验）

在写任何内核之前，先做**构造已知权重 → 单元素比对**的单元测试：手工构造 `[N,K]` 已知 codes → 过桥接的 `process_weights_after_loading` → 内核逐 (k,n) 解出来的值必须与解析值全等。以 fork 里**实际打过 gfx906 补丁的** `csrc/quantization/gptq/q_gemm.cu`（本地镜像 `container_src/q_gemm.cu`）为唯一事实来源，把位级映射画进实现文档。
理由：映射写错的表现是"误差看着不大但 argmax 乱跳"，是 L2 里最难查的一类失败。

### L1 单形状微基准（决定性）

**形状集 = §2 表里那 7 个 per-rank 组合**。不再写"枚举范围"—— checkpoint 原始张量既未分片也未融合，拿它当校验范围等于没有校验。两条前置：
1. **读 `quantization_config.ignore`**（1 分钟，决定枚举范围）。本 ckpt 实测：303 条，覆盖 `lm_head`、visual、每层 `linear_attn`（norm）与 `linear_attn.in_proj_a/in_proj_b`；但 **`in_proj_qkv` / `in_proj_z` / `out_proj` 是量化的**（各有 48 个 `weight_packed`）⇒ GDN 的 144 个投影**在** WNA16 工作集内。
2. **以运行时实测为权威**：用已有 `VLLM_WNA16_EXL_DEBUG` 通道在内核入口打 `(K,N,M)`，跑一次真实 serve 去重，与 §2 推导表互校 —— 不一致即说明我们对融合/分片的理解有错。这一步顺带交付 §1(d) 要求的真实 M。

测法：`torch.cuda.Event`，16 warmup + 64 次中位数；**eager 与 CUDA graph 内各测一遍**（作者数据：QKV 66.9→55.4µs，图内外差 ~17%）。

**锁时钟（本轮新增，且已实测可行）**：`echo manual > .../power_dpm_force_performance_level` + `echo 8 > .../pp_dpm_sclk` → 1700MHz，采完立即恢复 `auto`（已在本机验证读回与恢复）。注意 **`pp_od_clk_voltage` 在这两张刷了 VII BIOS 的卡上不存在**，所以只能锁离散 DPM 档，不能锁任意频率；每次采样同时记录只读时钟。L1 也锁，不只 L3。

四个对照：ExLlama fused 现路径、reconstruct→fp16 dense GEMM、`BW_eff` 地板（分母 = 内核真正从 HBM 拉的字节，含 scales/zp 与 Split-K 重复读）、以及 **Split-K 各档变体**。

产物：每形状"距 `BW_eff` 还差百分之几"+"graph 内哪个变体赢"。**选型只看 graph 内数字**；eager 记录但不作依据（交付默认 graph，eager 只用于排错）。

### L2 数值一致性（改成对"无自由度理想参考"做 ULP 计数）

**为什么不能只跟 ExLlama 比**：任何"两条实现互比 + 容差"的判据对 epilogue 类缺陷结构性失明 —— scale 在 fp16 里乘、中间多一次 fp16 cast、scale 折进归约，这些都只把输出挪动 ≤1 个 fp16 间距，容差再紧也看不见。所以精度把关换仪器。

**理想参考可逐位确定（W4A16 的幸运，INT8 那边做不到）**：x 是 fp16（11 bit 有效）、反量化权重 |w|≤15（4 bit 整数）⇒ `w·x` ≤15 bit ⇒ 单个乘积在 fp32 里精确；K=17408 的求和放 **fp64**（53 bit）全程无舍入；per-group scale 相乘一次；**唯一合法舍入是末次 cast 到 fp16 的 RNE**。
⇒ 参考 = fp64 累加 + 单次 RNE cast，零自由度。
**两点限定（不可含糊）**：
- "fp64 全程无舍入"是**数据依赖**而非定理：17408 项精确求和所需位宽 ≈ 指数跨度 + 15 + log₂K，x 动态范围正常时刚好落在 53 bit 内，极端离群值下 fp64 也会舍入。⇒ **harness 必须用 `math.fsum`（或 Kahan）并与朴素 fp64 累加交叉验证**，不通过则该形状的 L2 结果作废。
- **参考放 CPU + numpy float64、按 N 分块**：`K×N` 的 fp64 权重在 17408² 时是 2.4 GB，放 GPU 会挤占本就紧张的显存；放 CPU 顺带让"零自由度"更无可争议（完全脱离 HIP/ROCm 栈）。约 0.3 s/形状。

**判据（AND 关系）**：
- **L2-c** `RMS_ULP ≤ 0.5`（0.3–0.5 告警带）。**引用实测值**：fp32 累加 + 末次 RNE = **RMS_ULP 0.083 / MEAN_ULP −0.0005**，与理想参考不同的输出仅 0.317%；被禁的 fp16 累加 = **50–513（重尾）** ⇒ 分离度 **≥600×**，判定无歧义。
  （删除我之前写的 0.13 / ≤1.5 / ≈11.5 / "100×"：`√δ` 近似只在 δ<1 时成立，我对 δ=132 又开了一次根号 —— 那个 11.5 就是 `√√17408` 的误用，外部核验指出了这点，是对的。）
  **诊断带**：若 `RMS_ULP` 落在 0.5–2，**先别判死**，再对**同一实现的 fp32 顺序累加参考**比一次；若对 fp32 顺序参考 ≈0 而对 fp64 理想参考落在 0.5–2，那是合法的累加序噪声（真实数据不会全体同向舍入），不是缺陷。
- **L2-d** `|MEAN_ULP| ≤ 0.02`（带符号均值）。抓"截断取代 RNE"：截断会让均值单向偏移 ~−0.5 ULP。
- **L2-e 形状覆盖**：每个白名单形状都跑 L2-c/d，**含 M=1**（照抄上游 `test_machete_mm.py` 的 `MNK_SHAPES` 习惯，已核对该表首项就是 `(1,128,128)`）。
- 保留：同输入两次调用 `torch.equal` 全等（确定性来自无 atomicAdd）。
- 保留前置检查：先证明 reconstruct→dense GEMM 这条对照自身是 fp32/fp64 累加、且 hipBLASLt 在 gfx906 上有该形状后端（ROCm 曾修过 m=1/n=1 只给 16-bit 精度的 bug）。

**epilogue 规定（写死，靠构造与 review 保证，不靠数值测试）**：
1. per-group scale 在 fp32 累加**之外**应用 —— 先出该 group 的 fp32 点积，乘 scale（fp32）后累进 fp32 输出累加器；
2. **zp 采用形式 A**：`out += scale_g · (dot_q_g − zp_{g,n}·S_g)`（不采用 B `scale·dot_q − (scale·zp)·S_g`；两者代数相等、舍入不同，而**这种差异任何数值判据都看不见**）；
3. 全流程只有末次 cast 到 fp16 用 RNE。
**且这不是风格约定，是定量强制**：把 scale 折进激活（`x'_k = fp16(scale_g·x_k)`）每元素引入 2⁻¹¹ 舍入，随机游走后相对误差 `2⁻¹¹·√K/√2 = 4.55×10⁻²` = **46.6 ULP**，与被禁的 fp16 累加同量级、**超 RMS_ULP 门 93 倍**。⇒ 列为 L0/L2 的**显式 code review 检查点**，不依赖测试发现。

**结论：Split-K 在数值上是免费的**（部分和长度 K/S + S 路树形归约，S=8 相对 S=1 差 ~0.02 ULP，远低于 epilogue 预算）⇒ **splitK 纯按性能扫**；只加一个 10 分钟检查：L2 harness 里跑 S∈{1,2,4,8}，确认 RMS_ULP 不随 S 漂移。
注意上游注记"有 zeropoints 时放宽 atol，因为 machete 在 scale 之后应用 zp 会在 0 附近产生噪声" —— 我们的 ckpt 正是非对称有 zp。若 RMS_ULP 在 0 附近明显偏高，**先查 zp 应用顺序，不要放宽阈值**。

### L3 端到端（放弃"自由运行等值"，改为可归因的统计判据）

**先测噪声地板（新增强制步骤）**：同一内核、同一权重，prefill 与 decode 两条 regime 互比。文献报告这一项自身就有百分之几的翻转率且未能归因 ⇒ **不先测地板，就会把引擎自身的 regime 差异算到新内核头上**。地板同时用来标定下面所有阈值。

- **L3-a′ teacher-forced logits 比对（主判据）**：拿现路径产出的续写序列，**单趟 prefill** 喂两条路径，dump 每个位置的 top-N logits + 索引。
  - 机制：`/v1/completions` 的 `prompt_logprobs`；**vLLM 默认 `max_logprobs=20`**（已核对 `vllm/config/model.py:245`），要 top-32 必须显式 `--max-logprobs 32`，否则静默拿不到。
  - **前置：必须 pin 住分块 prefill**（`max_num_batched_tokens` 足够大 + `long_prefill_token_threshold` / 直接关 chunked prefill），保证每个序列**单块** prefill。分块会改变 attention 的归约顺序，引入与内核无关的位置间差异，**直接污染 A2 这条核心判据**（一个分块造成的假分歧会被当成真 bug）。
  - 规模：64 prompt × 256 位置 × 32 × 2 路径 ≈ 2MB；**不要 dump 全 vocab**（4.9GB×2）。
  - **A1** 翻转率与由 `σ_logit` + 实测 margin 分布预测的值同量级（不设绝对阈值）。
  - **A2（核心，等价于"不可解释分歧数 == 0"的正确形式化）**：存在 `M₀` 使 `margin > M₀` 的位置上翻转数 **为 0**，且 `M₀/σ_logit` 落在实测合理倍数。**一次发生在 margin=50σ 处的分歧 = 真 bug，无论总翻转率多少。**
  - **A3** 翻转位置不聚集、无周期 ⇒ 排除状态类 bug（KV/group 索引/越界）。
  - **A4** 无非有限值；翻转目标必须互在对方 top-N 内。
  - 已知局限（照录以免误用）：teacher-forced 首个分歧位置 ≠ 自由运行首个分叉位置 ⇒ 只做**分布级**判定，不许用它断言"第 N 个 token 会分叉"。
- **L3-b′ 自由运行只做质量判定，用 top-N 包容，不用等值、也不用百分比一致率**：`argmax 相同 → pass；否则双方 token 互落对方 top-8 → pass；否则 FAIL`。上游 `check_logprobs_close` 默认 `warn_on_mismatch=True`，**我们改成 fail**（没有"换张 GPU 就 skip"的退路）。**明确不设"top-1 一致率 ≥99.5%"这类阈值** —— 它标定在一个不存在的翻转率上。
- **L3-c′ 误差不随步数增长**：**只在 teacher-forced 固定上下文下测**，逐层 `e_layer[t]`，要求对 t 无系统性增长（回归斜率在噪声内）、末期 ≤ 初期 3×。自由运行下测这个没意义（一次翻转就让上下文分叉）。teacher-forced 下任何随 t 增长都是纯状态 bug。
- **L3-d′ 质量不退化（辅助）**：固定文本 perplexity `|ΔPPL|/PPL < 0.1%`。
- **CUDA graph 捕获测试（新增）**：照上游 `test_machete_mm.py::test_machete_cuda_graph` 写一个专用例 —— 这是"P0 结论里 capture size 决定内核真实 M"与内核设计之间唯一的连接点。
- **混合路径整体重验（保留）**：以上全部以打开白名单的真实配置跑。
- **性能与回归**：batch1 decode A/B —— 同一容器只切 `VLLM_WNA16_EXL_W64=0/1`，各两轮取优，**冷卡起测（结温 <50°C）+ 锁离散 DPM 档**；热卡数据作废（22.0 冷卡 vs 15.9 热卡前例）。同时确认 `Graph capturing finished`、KV 容量、`[serve]` 行不因新内核变化。
- 回退也测：`=0` 时落回 ExLlama 且 tok/s 回基线。
- 缓存串台：`c1_cachekey_test.sh` **覆盖热缓存下 0→1 与 1→0 双向**。规则不变：只有改变权重内存表示才分 cache 档；预期不分档，用双向实测验证而非推理。

**阈值一律自标定**：文献里的 4.7%/token、ROC-AUC 0.942、~9σ 等数字是**方法与量级的参照，不是我们的门限**（我们 vocab ~150K、margin 更小 ⇒ 翻转率只会更高）。所有 L3 阈值由本地噪声地板 + margin 分布定；P0/P1 报告里必须同时给出这两个分布。

### 测试载体
L0/L1/L2 用一次性容器 `vllm-gfx906-rc`（当前 Exited，用时启动）；L3 用 `gfx906-serve` 正式入口。模型在主机 `/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16`（注意 dev 容器挂载点是 `/home/x/models`，不是 `/models`）。长任务一律 `docker exec -d` + 日志，黑匣子保持运行（现确认仍在跑，pid 2472）。

## 4. 构建、交付与风险

### 构建与交付
- 新内核放 vLLM 源树现有扩展里：`csrc/libtorch_stable/quantization/gptq/` 新增一个翻译单元 + `CMakeLists.txt:439` 的源列表 + `ops.h`/`torch_bindings.cpp` 各一处登记（`rebuild` 目标**不会**编译它，理由见 §1「构建成本」与本节下一条）。Python 侧改动在 `wna16_exllama.py`（dispatch + 自整表）。
- 两条构建路径都能带上新 `.so`，但机制不同：**A** 由 `docker commit --no-pause` 把 dev 容器里编好的库带进快照；**B** 只认 `dist-packages.tar.gz`，故必须 `FORCE=1` 重打 tar（否则 [7/7] 静默跳过，见计划 Handoff 护栏 1）。库一律由构建阶段产出，禁止手工 scp 覆盖，沿用 md5 核对。
- `gfx906-selfcheck` 新增：`ops.gfx906_w4a16_gemv_out` 存在 + 对 M=1 单形状出有限值。
- `serve.sh` 的 `[serve]` 行加 `w4a16=exllama-w64`。
- QUICKSTART：更新基线表；排错表加"decode 仍在 20 tok/s 量级 → 看 `w4a16=` 是否为 exllama-w64"；明写 eager/graph 变体选型可能不同；**明写 ROCm 7.2.4 对 gfx906 官方不支持**这一事实（外部核验：ROCm 系统需求表 gfx906 = 不支持，预编译库会触发运行时错误 —— 我们靠自补 Tensile/rccl 才能跑，这点必须让用户知道）。

### 风险与处置

| 风险 | 处置 |
|---|---|
| 手写 HIP 几轮追不上 ExLlama | kill 判据：P2 结束时 L1 无任何形状 >1.1× → 停投 |
| **P0 归因指向 GDN/Triton 或 CPU/气泡而非 GEMV** | 用 §1(c) 路由表事先决定去处：GDN 主导 → 新开 #21b；CPU/气泡主导 → 配置与图覆盖，不开新内核任务。#21 只是第三顺位 |
| 数值偏差在长 prompt 累积 | L3 分层判据 + 分歧点归因（top-2 gap vs 扰动量），不靠"看着通顺" |
| 参考实现自身不准导致误判 | L2 前先做 fp64 对照确认参考是 fp32 累加 |
| 新 op 与 torch/ROCm ABI 不匹配 | op 缺失自动退回 ExLlama（不抛异常）；selfcheck 报告 |
| 长压测打死主机 | 冷卡起测、分形状小批、黑匣子在挂。**不用"压 `power1_cap` 到 150W"这个手段了**：这两张卡实测不遵守 190W 帽（曾读到 202/224W），压帽既不可靠又可能连 mclk 一起拉低，使 `BW_eff` 地板漂移；要限功率就锁离散 sclk 档 |
| 热降频污染 A/B（L1 与 L3 都会） | 锁 DPM 档 + 同热状态成对采样 + 记录只读时钟；任一轮结温 >85°C 作废重取 |
| 镜像 `.so` 与源树不一致 | md5 核对 + 构建阶段产出 |
| Split-K 多一次 launch → eager 下小 N 变慢 | 接受不对称，不优化 eager；QUICKSTART 明写 |
| batch1 被 pad 到更大 capture size 使 M=1 前提失效 | P0 (d) 实测内核收到的真实 M，不成立则整个 #20 的形态要重议 |

### 默认值决策规则（按形状分档）
- **进白名单**：该形状 graph 内 >1.1× ExLlama（各变体取优）**且** L2 数值与确定性全过。
- **总闸默认 1**：白名单非空，**且**混合路径下 L3 分层判据全过 + 冷卡 tok/s 不降 + P0 预测增益被证实。
- **总闸默认 0（实验开关）**：有形状进白名单、数值也过，但 e2e 增益 <1.15× → 说明瓶颈不在 GEMV。交付默认仍 ExLlama，P0 分解表转交下一个任务。
- **不进 dispatch**：任一形状数值门未过 → 剔除；白名单为空则 `.hip` 留源树但不注册进默认路径。

## 5. 实施阶段

- **P0（时间盒 1 天）**：`BW_eff` 探针 + 五桶分解与组件表（含 `lm_head` bf16 matvec 与 RCCL）+ busy ratio 路由 + 真实 M/真实 (K,N) 确认 + 逐形状基线（含实测 op/weight）→ 出口是 §1(c) 路由结论，不是 1.15×
- **P1**：L0 映射单元测试 → **L2 harness（fp64 零自由度理想参考 + RMS/MEAN_ULP 仪表）** → wave64 GEMV（先不 Split-K，大 N 跑通）+ L1/L2
- **P2**：Split-K 扫档（纯按性能，含 S∈{1,2,4,8} 的 RMS_ULP 不漂移抽查）+ 自整表 + dispatch 白名单 + `VLLM_WNA16_EXL_W64` + selfcheck + CUDA graph 捕获专用例
- **P3**：**先测同路径噪声地板与 margin 分布并据此定阈** → teacher-forced harness（`--max-logprobs 32`）→ 混合路径整体重验（L3-a′/b′/c′/d′）→ 冷卡锁档 tok/s A/B → 缓存双向验证 → 进 Dockerfile 配方 → 重建交付镜像 + md5 核对

## 6. 本轮实测/核验新增的既有事实
- **checkpoint 是非对称量化**：`Qwen3.8-27B-OBLITERATED-AWQ-W4A16` 的 `quantization_config`：`strategy=group, group_size=128, num_bits=4, symmetric=false, zp_dtype=torch.int8, format=pack-quantized, actorder=null`；ViT 的 `model.visual.*` 全在 `ignore` 里（视觉塔 fp16 未量化）。
- **桥接后的 qzeros 是 `[K/G, N//8]` int32、沿 N 打包、GPTQv2（stored nibble == true zero point）**，且对称 checkpoint 会被合成常数 zp → 内核无"无 zp"分支。
- **锁频可行但只能锁离散档**：`power_dpm_force_performance_level` 与 `pp_dpm_sclk` 存在（root:root，sudo 可用），manual+level 8 → 1700MHz 读回成功；`pp_od_clk_voltage` 不存在。此前记忆里"pp_dpm_sclk 写被拒"的结论是错的，已更正。
- **`rocprofv3` 在交付镜像里因缺 `libdw.so.1` 直接起不来**；`torch.profiler` 可用且能给逐内核设备时间。
- 主机 dev 容器模型挂载点是 `/home/x/models`（→ 容器内同名路径），不是 `/models`。
- 当前两卡 VRAM 占用 61MB / 21MB（空闲），`vllm-gfx906-rc` 处于 Exited(137)。

**2026-09-04 追加（读 checkpoint + config + 主机拓扑实测）**
- 架构（`config.json` 的 **`text_config`**，注意字段是嵌套的，顶层没有 hidden_size 等）：hidden 5120、intermediate 17408、64 层、24 Q head × head_dim 256、4 KV head、**`attn_output_gate=true`**、`full_attention_interval=4`（⇒16 full-attn + 48 GDN）、GDN 48 value head ×128 / 16 key head ×128、**vocab_size = 248320**。
- **`ignore` 的真实语义**：303 条里含 `lm_head`、`model.visual.*`、以及每层的 `linear_attn`（norm）与 `linear_attn.in_proj_a/in_proj_b` —— 但 `linear_attn.in_proj_qkv / in_proj_z / out_proj` **是量化的**（各有 48 个 `weight_packed`）。⇒ GDN 的 144 个投影在 WNA16 工作集内；**`lm_head` 不在 WNA16 路径上，不进 L1 枚举**（外部三份模型卡一致把它列为保留高精度，与我们的 dump 相符）。
- **张量数 vs launch 数**：`weight_packed` 共 **400** 个 = 48 attn qkv + 16 o + 128 gate/up + 64 down + 144 GDN；经 vLLM 融合后每 token **304** 次 GEMV。外部核验给的"160 次"漏了 GDN 那 144 个。
- **卡间链路**：`rocm-smi --showtopo` ⇒ GPU0↔GPU1 = **2 hops、Link Type = PCIE**，即**没装 Infinity Fabric 桥接器**，TP2 的 allreduce 走 PCIe（与既有"custom all-reduce 无收益"的结论一致）。QUICKSTART 应按 PCIe 写，不要暗示 xGMI。
- 所有 per-rank K（3072 / 5120 / 8704）都被 128 整除 ⇒ 无尾部 group。

## 7. 判据变更（原"待确认"，已由用户转来的 v2 判据文件回答）

原 #20 约束"逐 token ID 与现路径一致"**正式废止**。理由与替代：
- 上游对同类改动（换量化 GEMM 内核）不用等值判据：`tests/kernels/quantization/test_marlin_gemm.py` 头部明写 Marlin 用锁导致轻微非确定 + `@pytest.mark.flaky(reruns=3)`；`tests/models/quantization/test_gptq_marlin.py` 走 `check_logprobs_close`（top-N 包容）；`tests/models/utils.py:254` "Break out since sequences will now diverge."；`test_fp8.py` 因数值发散只测 4 token。**全部已在本仓库 0.28 源码逐字核对。**
- 自由运行下任何"连续 N>5 个 token 全等"的门都会以过半概率判死正确实现（首个分歧期望位置在几十 token 量级）—— 这条同时否掉了核验文档自己的 v1 L3-a 和我上一版的"32-token 硬门"。
- 替代 = **L2 对零自由度理想参考做 ULP 计数（精度）+ L3 teacher-forced margin 归因（端到端）+ 自由运行 top-8 包容（质量）**，全部阈值由本地噪声地板标定。

## 8. 附录：外部核验逐条处置（含否掉的理由）

核验方式：读 `container_src/wna16_exllama.py` 与 `q_gemm.cu`（我们的真实桥接）、ssh 到主机读 sysfs 与 checkpoint `config.json`、在容器里实跑 rocprofv3 与 torch.profiler。

| 核验条目 | 处置 | 依据 |
|---|---|---|
| §2.1 P0 缺 CPU/气泡桶 + busy ratio 门 | **采纳**（最高价值一条） | §1(b)(c) 已补四数分解与路由表 |
| §2.1 必须报告生效 cudagraph_mode / capture size / 真实 M | **采纳** | §1(d)；不成立则 #20 形态重议 |
| §2.2 "算子签名缺 qzeros" | **否（事实错误）** | spec v1 第 46 行就有 qzeros；核验者大概读的是中途版本 |
| §2.2 建议"assert zero_point==False / 白名单加 has_zp" | **否（对本项目多余）** | checkpoint 实测 `symmetric:false`；桥接对对称 ckpt 会合成常数 zp（`wna16_exllama.py:163-174`）→ 归一后永远非空，内核不需要分支 |
| §2.2 "zeros 在这条路径上是活的" | **采纳其结论**，并暴露了 spec 的真错：qzeros 形状标错 | 真实是 `[K/G, N//8]` int32 沿 **N** 打包、GPTQv2 编码（`wna16_exllama.py:21`），已改进签名注释并推出一条 lane 访问约束（§1） |
| §2.3 nibble 映射必须先有构造测试 | **采纳** | 新增 §3 L0 |
| §2.4 "Split-K 用 atomicAdd 与 torch.equal 自相矛盾" | **否（针对的写法我们没提）** | v1 §4 就写了"部分和 slab + 第二个 reduce 内核，零 atomicAdd"；确定性来自无 atomic，不靠运气 |
| §2.4 推荐 persistent 内核 + last-block 归约 | **暂缓**，列 P2 可选 | 理由成立（省一次 launch），但捕获安全需单调计数器技巧；先量 L1 再决定值不值 |
| §3.1 L3 绝对全等在数学上不可达 → 分层判据 | **采纳，用户已用后续 v2 判据文件确认**（§7）；且 v2 进一步把它升级为"L2 ULP 计数 + L3 teacher-forced 归因" | 60+ 层 ULP 扰动 vs 贪心 top-2 gap |
| §3.2 参考实现要先证明是 fp32 累加 | **采纳** | L2 前置检查 + hipBLASLt 后端可用性一并验 |
| §3.3 L1 也要锁时钟 | **采纳并已在本机验证可行** | manual + `pp_dpm_sclk=8` → 1700MHz 读回，恢复 auto 成功 |
| §3.3 用 `pp_od_clk_voltage` 锁任意频率 | **不可用** | 这两张卡该节点 ABSENT，只能锁离散档 |
| §3.3 降 sclk 优于压 TDP | **采纳方向** | 我们实测 190W 帽不被遵守（读到 202/224W）→ 压帽本就不可靠，已改风险表 |
| §3.4 profiler 可能用不了 | **采纳（结论对、原因不同）** | 实测 rocprofv3 在交付镜像里因缺 `libdw.so.1` 起不来；P0 用 torch.profiler（已实测出逐内核时间）。不是"ROCm 7.0 删了 MI50 数据"那个原因 |
| §5 指令级约束（dot2_f32_f16 / 禁 v_pk_fma_f16 / i24 乘法 / WAVE=64 写死 / 延迟隐藏预算 / x 分段进 LDS / splitK 扫档） | **全部采纳** | 与 v1 §2 同向并更具体，已并入 |
| §6 "新算子原地写 vs ExLlama 返回新张量 → FX 图不同" | **否（对代码理解有误）** | `ops.gptq_gemm_out(c_out, ...)` 已是原地写并返回同一 slab（`wna16_exllama.py:254-258`），两条路径调用约定相同 |
| §6 缓存测试要覆盖 0→1 与 1→0 双向 | **采纳** | 不拿推理替代测量 |
| §6 ROCm 7.2.4 对 gfx906 官方 unsupported 要写进文档 | **采纳** | QUICKSTART 增加明写 |
| §4 "1.15× 门否不掉什么，P0 时间盒一天" | **采纳** | 敏感性表三种情形全过；决策权重移到 busy ratio |
| clpeak 先验（760GB/s、half2 24971、launch 11.44µs） | **采纳为先验，不采纳为结论** | `BW_eff` 一律自测，先验仅用于校验探针是否错 |
| "arXiv 2603.22774 的 CPU 侧 46–73ms" | **无法核验**（该文号未查到；且是别的负载） | 不进入设计；由 P0 (b) 的实测 CPU 侧 step 时间取代 |
| §6 文字口径："§1 四个对照只列三个"、"§0 成功判据重复两次" | **否（当前文件不存在）** | `grep` 确认 `成功判据` 仅 1 处（第 17 行）、四个对照列满 4 项 —— 核验对象是编辑中途的版本 |

### 8.2 第二份核验（验收判据 v2）的处置

全部上游引用在**我们自己的 0.28 源码树**里逐条查过（`docker exec` + grep）：

| 引用 | 核验 |
|---|---|
| `check_logprobs_close` 的 top-N 包容 + `warn_on_mismatch: bool = True` | **逐字命中**（`tests/models/utils.py:91/102/233/246`） |
| "Break out since sequences will now diverge." | **逐字命中**（`utils.py:254`） |
| FP8 "只测 4 个 token" | **逐字命中**（`tests/models/quantization/test_fp8.py:44-45/129-130`） |
| Marlin 用锁→轻微非确定 + 重跑 | **命中但数字有偏差**：实际是 `@pytest.mark.flaky(reruns=3)`（核验文档写 reruns=2）；文件在 `tests/kernels/quantization/test_marlin_gemm.py` |
| Machete `MNK_SHAPES` 含 M=1 / `atol = 1 if zero_points else min(5e-2·√k,1)` / zp 注记 / `test_machete_cuda_graph` | **全部命中**（文件实名 `test_machete_mm.py`，`test_machete_cuda_graph` 在 411 行） |
| "GPTQ 与 Marlin 无 bitwise 正确性…互落 top-3" 这句原文 | **未在 0.28 树中找到**（该版本改用 `check_logprobs_close` 的 top-N）。结论方向不变，措辞不引用 |
| `top_logprobs` 上限 | **新增实测事实**：`max_logprobs` 默认 **20**（`vllm/config/model.py:245`）⇒ teacher-forced dump top-32 必须显式 `--max-logprobs 32`，核验文档未提 |
| RMS_ULP 预算表 fp16 累加那行 | **算术不一致**（该项已被 §8.3 再更正：我当时写的 ≈11.5 是 `√√K` 误用，正确是实测 **50–513**）："隔 3 个数量级"亦偏保守。阈值 0.5 依然把两者分得很开 ⇒ 采纳判据、引用数字以 §3 L2-c 为准 |
| "fp16 累加 6.4e-2 与你实测吻合，独立印证" | **否（循环引用）**：那个 6e-2 是它自己上一份文档里的闭式估算，不是我们的测量。不作独立证据 |
| arXiv 2608.13756 / 2609.00363 的定量数字（4.7%/token、0/64 全等、ROC-AUC 0.942、epilogue 失明、prefill-vs-decode 8.6%/2.9%） | **本地无法核验**（无外网核对全文）。处置：**采纳其方法与判据形状，不采纳其常数值** —— 所有阈值改为 §3 "阈值一律自标定"，先测我们自己的噪声地板与 margin 分布。这些数字若日后证实不同，只需换标定的数，不需换判据 |

### 8.3 第三份核验（spec v2 逐条评审，2026-09-04）的处置

**直接采纳（我原设计确实错/漏）**：VALU 预算重算与结论反向（§2 新增预算表）、zp 沿 K 恒定 ⇒ 提升出 K 循环（P0 级代数变形，全文漏写）、epilogue 补 zp 形式并把"折 scale"从约定升级为定量禁止（46.6 ULP / 超门 93×）、`ds_bread`→`ds_read_b32`、我那个 11.5 是 `√√17408` 的误用（改正为实测 0.083 与 50–513）、Split-K 是默认路径 ⇒ persistent 归约必须重估、RCCL 单列一个桶、L1 同时报实测 op/weight、主战场是 nibble 提取而非 dot2、K%128==0 省尾组、fp32 顺序参考做诊断带、fp64 参考放 CPU 分块、pin 分块 prefill、fp64 无舍入是数据依赖（加 `math.fsum` 互验）。

**方向采纳、数字用我们的实测替换**：
- 它给的"真实形状集 = (5120,4096)/(3072,5120)/(5120,17408)/(8704,5120)"**与本 checkpoint 不符**：`q_proj` 实测 N=**12288**（它按 24×256=6144 算，漏了 `attn_output_gate=true` 会让 q 翻倍）、KV head 是 **4** 不是 8（k/v 各 1024 而非 2048）、且它假设 gate/up 在 ckpt 里已融合（实际是**两个独立张量**，vLLM 加载时才合并）。per-rank 真值见 §2 表（qkv 7168 / o 5120 / gate_up 17408 / down 5120 / GDN 5120·3072·5120）。
- 它说的"N=1024 根本不存在"对**内核输入**成立、对**checkpoint** 不成立：`k_proj`/`v_proj` 存的是 N=1024，融合 + TP2 后并入 7168。它由此推出的"3/4 形状要 Split-K"结论**在我们的真值下更强**（7 个里 6 个要），所以结论采纳、推导表替换。
- 它把每 token 投影数算成 **160**：漏了 GDN 的 144 个量化投影（`in_proj_qkv/z`、`out_proj` 各 48）。实测分母是 **304**（400 个张量融合后）。我原写的 450 也是错的（那是未去重的张量级猜测）。
- 它建议"`has_zp` 进白名单 key / 或 assert 无 zp"：**对本项目不需要** —— 见 §8.2 与本文件 §1 的 qzeros 归一事实。

**核验者自己的算术错**：qzeros 占比 —— `[K/128, N/8]` int32 = `K·N/256` 字节，对 `qweight = K·N/2` 的比是 **0.78%**，它写 1.56%（除错一位），于是总开销是 **3.9% 而不是 4.7%**。已按 3.9% 写入 §2。

**它无法定、我们已实测掉的两个"阻塞项"**：`linear_attn` 是否量化（已查：三大投影量化、norm 与 in_proj_a/b 不量化）；卡间是否 xGMI（已查：`--showtopo` = PCIE、2 hops，无桥接器）。

## P1 实测结论（2026-09-09）：预测 vs 实测

原型微基准（三个变体 × splitK × unroll，171 组合，1700/1000 锁频，graph 内计时，
与现路径同尺）的结果见 `docs/superpowers/plans/2026-09-09-p1-report.md`。逐条对照：

| 设计条目 | 预测 | 实测 | 结论 |
|---|---|---|---|
| §映射 位规则 `(i//2)*4+(i%2)*16` | — | L0a 逐字全等 + L0b one-hot 直读 max_rel=4.88e-4（=1 fp16 输出量子） | **证实**（nibble-map.md 钉死） |
| §VALU 预算：提升 zp 3.85 op/weight ⇒ V1 快于 V3 | V1 > V3 | **V3 全线快于 V1（每形状 10-40%）** | **证伪**：butterfly 归约 + 去偏 + 双 barrier 的真实代价未被预算表覆盖 |
| §映射 "数值代价 ~7e-7" | 抵消噪声可忽略 | magic 偏移使 fp32 累加在 ~4096 尺度舍入 ⇒ V1 噪声 9.1e-4（1024 组）~1.6e-2（真实形状），无偏提取（LUT/cvt）才为 0 | **证伪**；无偏提取的代价 = LUT 28 ops/word 或 cvt 路径 ⇒ **新增设计自由度** |
| §:110 reduce launch 代价未量 | "L1 必须先量" | in-graph 边际 **1.887 µs/次** ⇒ 192 次/token = 362 µs/step | 可承受，persistent 非必需 |
| §其他 "Split-K 重复读元数据 splitK 次" | 地板 ×splitK | split 沿 K 不相交 ⇒ 元数据只读 **1 次** | **证伪**（地板按 1 份算） |
| §:157 LUT "1 op/nibble ⇒ 8 条/字" | 可能优于提取 | 本格式一对的两 nibble 相距 16 位 ⇒ 字节索引 dword LUT 不可行，实为 2 LDS/对 ⇒ **退化** | 证伪（V2 且首版还漏了 zp 减法，已修） |
| §:100 vs :181 形状数 | 5 或 7 | **5**（两 rank 各 16896 行，extra/missing=∅，m=1 直接观测） | 5 |
| §:181 5 vs 7 | — | 同上 | 文本待改（本节即改） |

**总体判定**：171 组合中仅 (5120,8192) ×1.19 与 (5120,7168) ×1.14 两族（均 V3/精确数值）胜过
ExLlama；加权整体 ×0.83，按形状分派 +3.6% decode。判据未干净命中 ⇒ P2 缩窄与否交用户
（报告"给 P2 的取舍"A/B/C）。

**给 P2 的三条硬输入**：
1. 用 V3 结构（逐元素 `v_pk_sub_f16` 减带偏 zp 或 cvt 无偏码）——精确且实测最快；
   magic+提升形式（V1）的噪声与开销在真实形状不划算。
2. splitK=8 是 (5120,7168/8192) 的最优档；unroll 4-8 优于 16（寄存器压力旁证，VGPR 未测得）。
3. op/weight 与 VGPR 的 ISA 通路未打通（`.so` bundler 列表为空、`.cuda.o` 反汇编不出）——
   P2 开工前先补这条工具链，否则主 KPI 缺失。
