---
title: 外部调研核实 + 项目重排（本地实证部分）
date: 2026-09-10
input: `C:\Users\Admin\WorkBuddy\2026-09-10-12-23-06\z100-vllm-gfx906-调研汇总与行动手册.md`
method: 每一条判定都落在**容器/宿主实测输出**或**upstream tag 原文**上；不采信任何未经回读的转述
companion: `2026-09-10-gfx906-change-and-test-summary.md`（本文对其 §5 排序做修正）
status: §1–§6 全部落地（§4 FP8 冒烟、§5 fork/上游核实、§6 长上下文实测均已出数）；§3 排序表已按 §6 重排
---

# 调研核实与项目重排

> **结论先说**（这条 digest 整体质量不低，但它的九条行动项里有四类问题）：
> 1. 🔁 **4 条我们已经做过**：分段数旋钮、`on_gfx906()` 架构谓词、`BLOCK=32`、按 33 ms 重算的桶表。
>    它是在**上游 tag** 上检索后推断"你照抄了名字但没接上"，没有读我们的树。
> 2. ❌ **1 条前提被当场证伪**：C3 的 `sramecc-`（我们两张卡都是 `sramecc+`）。
> 2b. ✅ **只有 C1（GDN readout fp16 溢出）整条成立、且我们确实没做** —— 它是 digest 最有价值的一项，
>    补丁条件在我们的配置下**恰好为真**（`mamba_ssm_cache_dtype` 由 checkpoint 解析成 **fp32**，不是 fp16）。
>    代价是我在这一节**连着判错两次方向相反**（§1.3 有全文），所以它的每一步都留了实测出处。
> 3. 🚩 **1 条归属错误**：说我们的 custom all-reduce 是"社区完全没有的原创资产"——
>    我们**从未写过实现**，只压测过内置的那一个，B4 的结论就是"与 RCCL 持平 ⇒ 交付禁用"。
> 4. ⚠️ **2 条成本被低估一个量级**：`LLMMB4` 与 `q_gemm` 调参都在 `csrc/` 里，
>    而我们**两条交付构建路径都不重编 vLLM 扩展**。"同版本同文件 ⇒ 代码可直接搬"在 python 层成立，
>    在 C++/HIP 层不成立。
>
> **核实过程的副产品**：在交付镜像里挖到一个真 bug（§1.7），以及一条把 `elementwise`
> 那 12.4% 直接结案的离线归因（§1.9，零 GPU 时间）。

---

## 1. 逐条裁决（本地证据）

### 1.1 C2 `VLLM_TRITON_ATTN_NUM_SEGMENTS=64` 是 no-op —— ❌ 对我们不成立（已做）

digest 说它在五个版本的 `envs.py` 里检索这个变量都是 0 命中，并给出 upstream `v0.28.0` 的
`NUM_PAR_SOFTMAX_SEGMENTS = 16` 硬编码 ⇒ 推论"你一直在 16 段上跑"。

**上游那半是对的，推论那半是错的。** 联网回读 `v0.28.0` tag 原文：

```python
# upstream v0.28.0  vllm/v1/attention/backends/triton_attn.py L62
NUM_PAR_SOFTMAX_SEGMENTS = 16      # Number of parallel tiled softmax segments
```

**我们的树里这一行早就被改成 env 可读**（`triton_attn.py` L55-64，注释里写着 gfx906/60 CU 与
"64 segments is optimal (+26% 8K decode)"，默认仍保留上游的 16）：

```python
NUM_PAR_SOFTMAX_SEGMENTS = int(
    _os.environ.get("VLLM_TRITON_ATTN_NUM_SEGMENTS", "16")
)
```

实测三重证据（交付镜像内的容器，2026-09-10）：

| 检查 | 读数 |
|---|---|
| `import NUM_PAR_SOFTMAX_SEGMENTS; print(...)` | **`segments = 64`** |
| `printenv VLLM_TRITON_ATTN_NUM_SEGMENTS` | `64` |
| `docker/entrypoint/serve.sh:21` | `export VLLM_TRITON_ATTN_NUM_SEGMENTS="${...:-64}"` |

⇒ 该项**已在交付镜像里生效**，digest 排的"第 1 项（半天）"是重复劳动。
它顺带那条坑（**ENV 类守卫必须验"运行时是否消费"**）仍然有效并且正中我们：
`selfcheck.sh:56` 现在只是**比 env 字符串**，应改成断言**导入后的常量 == 64**。这是 5 行的改动。

### 1.2 C3 `sramecc-` 自检 —— ❌ 前提不存在

```
rocminfo:  L162  Name: amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-     （card0）
           L275  Name: amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-     （card1）
```

两张 Radeon Pro VII 都报 **`sramecc+`**，与"近期 ROCm 预编译库只带 sramecc+"的说法**方向相反**。
⇒ "拷 156 个 Tensile 文件 + 重编 RCCL + `.pth` 绕 GEMV 段错误是同一根因的三个症状"这个假设对我们不适用。
我们那三件事的根因仍是：**宿主 ROCm 7.2.4 干脆不带 gfx906 的 kernel**（与 sramecc 标志无关）。
这条 30 秒自检值得做，做完就可以从行动手册里划掉。

### 1.3 C1 GDN readout fp16 溢出 —— ✅ 成立、且适用于我们（本节我连着判错两次）

**机理在我们树里确实存在**（这是 digest 最有价值的一条）：

**内核事实**（`fused_sigmoid_gating.py`，容器内实读，行号对得上 digest）：

```
L102  b_h = tl.zeros([BV, BK], dtype=tl.float32)          # 状态累加器 fp32
L136  b_h *= tl.exp(b_g)                                   # 衰减 ∈ (0,1]
L144  b_h += b_v[:, None] * b_k[None, :]                   # 累加
L146  b_o = tl.sum(b_h * b_q[None, :], 1)                  # readout：无 softmax、未归一化
L147  tl.store(p_o, b_o.to(p_o.dtype.element_ty), ...)     # ← 按 o 的 dtype 落盘
L225  o = q.new_empty(NK, *v.shape)                        # ← 继承 q.dtype = fp16
L166  tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), ...)   # 状态按 cache dtype 存
```

**dtype 事实**（实测 `EngineArgs(...).create_engine_config()`，不是读通用链推的）：

```
CKPT_TEXT.mamba_ssm_dtype      = float32     # checkpoint text_config 自己声明
RESOLVED mamba_ssm_cache_dtype = float32     # Qwen3_5ForConditionalGenerationConfig.verify_and_update_config 覆盖了 auto
RESOLVED mamba_cache_dtype     = auto        # conv state 跟随模型 dtype = fp16
```

⇒ 状态存盘是 fp32，**不会**在状态块里留 `inf`；但 readout 落进的是 **fp16** 缓冲。
L134 只把 `b_q` 做 L2 归一化 + scale，所以 `b_o` 与**未归一化**的 `b_h` 同量级；
`b_h` 的稳态上界 ≈ `|b_v|·|b_k| / (1 − exp(b_g))`，衰减越接近 1 越大 ⇒ digest 说的
"长上下文后超过 65504" 在数学上成立，**触发点就是 L147 那一次 cast**，随后 `RMSNormGated` 出 `inf/inf = NaN`。

⚠️ **本节我连着判错两次，两个方向都错，记在这里防止再错**：
1. **初稿**只读 `mamba_utils.py:103-106` 的通用链（`auto → conv_state_dtype` → fp16）就断定
   "递归状态存 fp16 ⇒ 暴露面比 digest 说的更大"。错：Qwen3.5 的配置类在通用链之前就把 `auto`
   改写成了 checkpoint 的 `float32`，通用链轮不到执行。
2. **二稿**据"状态是 fp32"判这条"没有触发面"，理由写成"`o` 是 `v` 的凸组合，`|o| ≤ max|v|`"。
   **那个理由是我编的，且是错的**：L146 的 readout 里没有任何 softmax/归一化。二稿还据此下了
   "不打这行补丁"的结论 —— 照它执行就是**把一条真实的稳定性项当噪音关掉**，是三次判断里后果最重的一次。

⇒ **裁决**：digest 这条 **成立、且适用于我们**。它给的补丁条件
`initial_state.dtype == torch.float32` 在我们配置下**恰好为真** ⇒ 装上去会真的生效（不是 no-op）。
唯一还没证的是"会不会真的烧到 65504"。

⇒ **行动 = 幅值探针**（这个判断我初稿就给对了，错的是理由）：16K 满 ctx greedy 长输出，
采 `b_o`（cast 之前）与 `b_h` 的 `max|·|` 随 token 曲线。离 65504 有 ≥4× 余量 ⇒ 记"不阻断、按限制存档"；
逼近 ⇒ 装 #54146 那一行，成本是 `o` 由 fp16 变 fp32（`(NK, T, HV, V)` 小张量翻倍）。
digest 还指出 chunk/prefill 侧 `chunk_o` / `chunk_delta_h` / `wy_fast` 是同模式、作者明说留作 follow-up
⇒ **探针必须一并覆盖 prefill 段**，只测 decode 会漏掉 16K 一次性累加的那条路径。

**顺带一条从 dtype 事实里掉出来的杠杆**：fp32 状态是交付默认**已经在花的钱** ——
`(HV/tp, 128, 128) × 48 层` ⇒ 每序列每卡 **fp32 75.5 MB / fp16 37.7 MB**，`max_num_seqs=48` 下
**fp32 共 3.63 GB/卡**。所以 `--mamba-ssm-cache-dtype fp16` 是一个可**回收 1.81 GB/卡** 进 KV 池的容量杠杆
（原稿把它写成"待付出的成本"，方向反了）。但它同时会把溢出从 L147 提前到 L166，即真的引入 digest 担心的
状态污染面 ⇒ **这两个探针（幅值 / fp16 状态容量）必须一起读，不能单独采纳容量收益**。

顺带核掉一个可能的"现成轮子"：上游确实有 `--enable-mamba-cache-stochastic-rounding`，
但它只挂在 Mamba2 的 SSU 分发（`ops/ssu_dispatch.py`、`ops/mamba_ssm.py`），
**GDN 侧 grep `stochastic|RS_ROUND` = 0 命中** ⇒ 不是可用补救（且它治的是舍入偏差，不是本条的溢出）。

**A1（44/44 内核 vs naive fp32 参考）覆盖不了它**：A1 是短 horizon，且参考本身是 fp32 ⇒
量不到"fp32 值超出 fp16 上限"这一类失败。

### 1.4 §5.1 "custom all-reduce 是你唯一的原创资产" —— 🚩 归属错误

全仓检索 `custom_all_reduce`：命中的只有 `b4_probe.py` / `b4_probe2.py` / `b4_stale.py`
（**探针**，直接 import 内置 `CustomAllreduce` 类做压测）与各测试脚本里的 `disable_custom_all_reduce=True`。
`container_src/_custom_ops.py` 是 upstream 副本。**我们没有任何自研 all-reduce 实现。**
B4 的原始结论是："**内置** custom AR 在本硬件可用且 52 轮变尺寸压测 CLEAN，但性能与 RCCL 持平
⇒ 交付保持 `disable_custom_all_reduce=True`"。"Bounded Staging Ring + 256 KB 阈值"是 **Z100 作者**的设计。

⇒ 它建议的"把 RCCL 单独立项"这条**有效**（见 §1.9）；"整理成 patch/博客反哺社区"这条**必须删除或改写成
"B4 的负结论 + 变尺寸压测方法学"**。否则这份手册会把一个不存在资产写进后续计划。

### 1.5 §3.2 "`on_gfx906()` 是 fork 新加的" —— ❌ 我们已有（且是我们自己的补丁）

联网回读 `v0.28.0` tag 的 `vllm/platforms/rocm.py`：**整个文件不含 "gfx906"**。
而我们镜像里同一文件有：

```
L177/L181  "gfx9006" -> "gfx906" 的 amdsmi 报错名归一
L232       _ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a","gfx942","gfx950"])
L236-239   # gfx906 (MI50 / Radeon VII / Vega20, GCN 5.1). Deliberately NOT part of _ON_GFX9 ...
           _ON_GFX906 = "gfx906" in _GCN_ARCH
L367       def on_gfx906() -> bool: return _ON_GFX906
L371-381   def _set_gfx906_nccl_workarounds() ... ; _set_gfx906_nccl_workarounds()
L457-458   # gfx906 has no native FA; the Triton-AMD path is the viable one
```

⇒ **后果**：搬 fork 的 `LLMMB4` hunk 时**不要**再新增这些符号，直接用我们这份。
两边是**在同一支 stock 0.28.0 上各自独立发明了同名谓词**（他们用**同样被排除 gfx906 的** `_ON_GFX9` 列表
⇒ 机理判断也一致），所以这是"撞车"而不是"落后"，但一旦按 diff 打补丁就会**符号重定义冲突**。

⚠️ **订正本节初稿的一处推理错误**：我原先写"`ahead 99 / behind 0` 与'它新增 `on_gfx906()`'互相矛盾，
说明它的基线也不是 stock 0.28.0"。**这条是我错了**——stock 0.28.0 里确实没有 gfx906，
"新增这些符号"**正是** "ahead 99 / behind 0" 的含义。联网按 SHA 实测（§5.1）：
`v0.28.0 = 2cf0a691…` 是 `main` 的真祖先，`compare` 给出 **ahead_by 99 / behind_by 0 / merge_base = 2cf0a69**，
数字与它声称的**逐字一致** ⇒ "同版本、可干净 cherry-pick"这个前提是**成立的**（细节见 §5.1）。

### 1.6 §3.5 "prefill 上游已实现，只是不在你那条路径" —— ⚠️ 半对，且参照物要换

- ✅ `auto_awq.py:927` 确实有：`FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256` → `awq_dequantize` + `torch.matmul`。
- ✅ `auto_gptq.py` 里 grep `HEURISTIC|dequantize|torch.matmul` → **0 命中**（它没说错）。
- ❌ 但**我们两条都不走**：模型 `config.json` 的 `quantization_config.quant_method = **compressed-tensors**`
  （架构 `Qwen3_5ForConditionalGeneration`，`text.dtype = bfloat16` 而我们跑 fp16）。
  `compressed_tensors/schemes/*.py` 里同样**没有任何** heuristic/dequant→matmul 分支。
- 🔎 **同树里更该参照的轮子**（digest 没找到）：
  `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` ——
  一个**已经存在的、按 M 分派的 W4A16 混合内核**：

  ```
  M <= MAX_SKINNY_BATCH_SIZE(=5):  HIP skinny GEMM (wvSplitK_int4_g)
  M >  MAX_SKINNY_BATCH_SIZE:      Triton W4A16 fused dequant GEMM
  Stores the weights ONCE as int8 [N, K//2] (ExLlama shuffle packed).
  Both paths read this single buffer ... No dual weight storage.
  ```

  它的"大 M 走 fused dequant"那一半在 gfx906 上我们不划算（我们的 Triton W4A16 比 ExLlama 慢 ~8×），
  但**骨架**（单份权重 + 按 M 分派 + 白名单）正是 §5.2 要的形状，而且它证明
  "同一份 packed 权重被两个内核读"是可以做到的（我们 §2.2 曾为此吃过两种重排的亏）。
  ⇒ 该项判定：**仍是自建**，但从"照抄上游一个分支"降级为"参照现成骨架 + 自己写 dequant-once→稠密"，
  且 digest 那条"必须吃原始 qweight/scales/qzeros 而非 repack 后的 ExLlama 布局"提醒**有效**。

### 1.7 🔴 新发现：交付镜像里有一个不存在的平台谓词（真 bug）

```python
# vllm/v1/attention/ops/triton_decode_attention.py:509（我们的补丁）
BLOCK = 32
if is_hip_ and not current_platform.is_gfx906():
    BLOCK = 16
```

`is_gfx906` **在平台类上不存在**：

| 探针 | 结果 |
|---|---|
| `n in dir(RocmPlatform.__class__)` | `real-attr = False`（`is_gfx906` / `is_gfx9` 都是） |
| `hasattr(current_platform, "is_gfx906")` | **True** —— 因为 `interface.py:1145` 的 `__getattr__` 兜底返回 `None` 并 warn |
| `current_platform.is_gfx906()` | **`TypeError: 'NoneType' object is not callable`**（实测） |
| `is_hip_`（同文件 L40 模块级） | `True`（`type(current_platform)=RocmPlatform`，`torch.version.hip=7.2.53211`） |

⇒ 在 ROCm 上**任何执行到这行的路径都会抛**。当前不可达：该文件只被
`v1/attention/backends/mla/triton_mla.py` 与 `triton_turboquant_decode*.py` import，
我们跑 TRITON_ATTN 标准注意力 ⇒ 没碰到。但 **gfx906 上 MLA 架构模型只能选 TRITON_MLA**
⇒ 对交付用户是一颗硬雷，且报错信息（`'NoneType' object is not callable`）完全不指向根因。

**修法**：`from vllm.platforms.rocm import on_gfx906`（已核实 `on_gfx906()` 返回 True）。
**并给 selfcheck 加一条通用守卫**：对每个补丁文件里出现的 `current_platform.<attr>` 做
"`__dict__`/`dir` 级存在性检查"，而不是 `hasattr`（`hasattr` 在这个兜底 `__getattr__` 下**永远为真**）。

顺带的诚实结论：这一行也从没生效过 ⇒ 我们在 MLA 路径上既没有 BLOCK=32 也没有实测。
全仓 sweep 这种"发明谓词"的模式，**只有这一处**。

### 1.8 §3.4 FP8 KV 门禁 —— ✅ 成立

容器内实测：`capability = DeviceCapability(major=9, minor=0)`、
`has_device_capability(89) = True`、`is_rocm() = True`；
`platforms/rocm.py:1007 is_fp8_fnuz()` 判的是 `"gfx94" in _GCN_ARCH` ⇒ 我们 **False** ⇒ dtype 走
`torch.float8_e4m3fn`（两个 fp8 类型在本 torch 里都可用）。
⇒ digest 关于"能力位撞车反而放行"的判断，本地实证成立。选项 A 的冒烟见 §4。

### 1.9 §2 第 3 项"按 33 ms 重算桶表 / elementwise 从未归因" —— 🔁 一半已做，另一半刚做完（0 GPU）

**(a) 桶表早就按 33 ms 算过**：`logs/p0/p05_buckets_ab_on.log`（P0.5 Task 5 产物）就是
分母 33.01 ms 的表，`%wall` 列：`w4a16_gemv 60.2 / rccl_wait 23.0 / elementwise 12.4 /
dense_gemv 5.2 / gdn 2.6 / attention 1.6 / memcpy 0.4`。digest 的"≈18%"就是我们表里
5.9 ms 折价 ÷ 33 ms，**同一件事**。

⚠️ 但它在写法上把桶当成可加的了。用 P0 原始 trace（`/home/x/vllm-work/p0/p0_kernels.json`）实测重叠：

```
Σ per-step kernel time = 73 349.8 µs/step
union（去重叠后占用）    = 73 197.0 µs/step
墙钟 per-step           = 53 678.0 µs/step      ⇒ sum/wall = 1.37，union/wall = 1.36
```

⇒ 各桶**不是互斥槽位**（`rccl_wait` 大量与计算重叠）。所以"RCCL = 18% 头寸"只能作为
**待定价区间**记账，不能当成"减掉它就有 18%"。它建议的"给 RCCL 单独立项"这个**动作**是对的，
**算术口径**要按 overlap 写。

**(b) elementwise 拆开归因已经做完（纯离线，未用 GPU）**：679 launches/step、4 086.5 µs/step，
前六名全是 **inductor 已经把 W4A16 的输出后处理融进同一个 kernel**：

| kernel（截断） | launches/step | µs/step | 单次 |
|---|---:|---:|---:|
| `triton_poi_fused_gptq_gemm_out_mul_silu_slice_zero_5` | 48 | 392.9 | 8.19 µs |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_gptq_gemm_out_zero_4` | 48 | 320.9 | 6.69 µs |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_6` | 32 | 260.3 | 8.13 µs |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_4` | 16 | 133.6 | 8.35 µs |
| `triton_poi_fused_gptq_gemm_out_mul_silu_slice_zero_3` | 16 | 133.4 | 8.34 µs |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_gptq_gemm_out_zero_6` | 16 | 133.1 | 8.32 µs |

⇒ **digest 那句"零新内核的融合机会"要降级**：能融的已经融了（名字里同时带 `gptq_gemm_out` 与
`rms_norm`/`silu`）。剩下的形态是"很多个 ~7–8 µs 的小 reduce"（vs graph per-node 地板 2.209 µs），
要再省只能**减少 kernel 数量**本身，两个具体靶子：
① `rms_norm` 的 `red`（两趟归约）；② 名字里的 `zero` —— 那正是我们为 ExLlama atomicAdd
加的 `zero_()`（P2 内核是覆写语义、不需要它 ⇒ **被 P2 替换掉的那两族理论上还能再省一次 memset**，
这条 digest 没提，是我们自己的数据里能直接读出来的）。

### 1.10 其它核对到的模型侧事实（用于校 digest 的前提）

`config.json`（宿主 `/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16`）：
`num_hidden_layers=64`，`layer_types` = **48 × linear_attention + 16 × full_attention**（`full_attention_interval=4`），
`num_key_value_heads=4` ⇒ TP2 每卡 2 个 KV head（digest 对，`seq_threshold_3D = 128//2 = 64` 的推导成立），
`head_dim=256`，`hidden_size=5120`，`intermediate_size=17408`，GDN：`linear_num_key_heads=16 /
linear_num_value_heads=48 / 两维 head_dim 都是 128`。
另外 serve 日志显示 hybrid 对齐机制**已在运行**（下面这个 1568 是 **fp8 KV 臂**的值；fp16 交付默认是 **784**）：
`Setting attention block size to 1568 tokens to ensure that attention page size is >= mamba page size`
+ `Padding mamba page size by 0.13%` ⇒ 它担心的"KV 减半会把 block_size 顶大"是真效应，
但我们树里已有这套对齐逻辑，不是新增风险面（具体值见 §4 的受控对照）。

### 1.11 🔴 #21 的前提被两头削弱：那个轮子我们可能已经在转，且它离我们不是一次"搬内核"

fork 侧读起来像"gfx906 长上下文 decode 必须有自己的 split-KV 内核"：

- **phase-141**（同一 TP4 rank、`head_dim=256`、2 KV head、784-token hybrid page、32,780 cached tokens、
  batch-1、纯内核不出服务器）：**SplitKV paged decode 0.865 ms/层 vs pure-Triton unified attention
  13.961 ms/层 ⇒ 慢 16.1 倍**（`max_abs_error=1.53e-05`，判据是"直接内核 ≥+20%"）。
- **experiment-summary**：未修复的 generic 路径在长解码复现器上 **0.238 tok/s**，32K 前缀命中 **1.353 tok/s**。

两头证据都指向同一个结论的反面。

**(a) 我们装出来的 `TRITON_ATTN` 本来就跑在分段（split-KV 同构）路径上：**

| 证据（容器内 `dist-packages/vllm/`） | 值 |
|---|---|
| `v1/attention/backends/triton_attn.py:43` | `from ...triton_unified_attention import unified_attention` |
| `triton_attn.py:56-64` | `NUM_PAR_SOFTMAX_SEGMENTS = int(os.environ.get("VLLM_TRITON_ATTN_NUM_SEGMENTS","16"))`，**我们镜像里是 64**，注释写明理由（+26% 8K decode，128 无进一步收益） |
| `triton_attn.py:55,149` | `MIN_LAUNCH_GRID_SIZE_2D=128`；`seq_threshold_3D = 128 // num_heads_kv = 128//2 = 64` |
| `triton_attn.py:152-160` | `FULL_DECODE_ONLY` 下把该阈值**吸附到最接近的 capture size** ⇒ `min([1,2,4,8,16,24,32,48] 距 64)` = **48** |
| `triton_unified_attention.py:323, 703-760` | `IS_3D` 时 `tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS_PER_SEQ*TILE_SIZE)`，并配分段 max/expsum reduce 内核 |

⇒ **batch ≤ 47 的 decode 已经在 64 段 split 上**，与 fork 的 SplitKV-29 是同一个结构思路（他们 29 路、我们 64 段）。
所以 phase-141 那 16.1× **不能直接搬给我们**：它的对照 arm harness 在未公开的
`perf/gfx906-triton-unified-qwen27` 分支上，**文档没有披露跑的是默认 16 段还是关掉分段**；
而 16 段 vs 64 段本身就是 192 vs 768 个 program，在 60 CU 的卡上足以解释掉大部分差距。

真正可能存在的缺口是另一件事，而且比我原来写的 #21 小得多：阈值被吸附到 48 ⇒ **恰好 batch=48 那一档
掉回不分段的 2D 核**，而 48 序列 × 2 KV head 只有 96 个 program，低于它想要的 128。
这是"高并发 + 长上下文"的一档，不是交互档。

**(b) 更要紧：fork 的 SplitKV 不在 TRITON_ATTN 里，而在 `ROCM_ATTN` 的内核文件里 —— 而 `ROCM_ATTN`
在 v0.28 已经不依赖 MFMA。**

| 证据 | 值 |
|---|---|
| `grep -rln chunked_prefill_paged_decode vllm/v1/` | 只有 `ops/chunked_prefill_paged_decode.py` 与 **`backends/rocm_attn.py`** 两个命中 ⇒ 该内核是 ROCM_ATTN 专属 |
| `rocm_attn.py:446` | decode 直接调 `chunked_prefill_paged_decode(...)`（纯 Triton） |
| `rocm_attn.py:349` | prefill 用 `ops.triton_prefill_attention.context_attention_fwd`（纯 Triton） |
| `rocm_attn.py:35` | cache 写入用 `triton_reshape_and_cache_flash`（纯 Triton） |
| `rocm_attn.py:193-194` | `get_supported_head_sizes() = [32,64,...,224,256]` ⇒ **含我们的 256** |
| ~~`grep asm_pa\|mfma\|ops.rocm_ rocm_attn.py` = **0 命中**~~ | ❌ **方法不足，见下** |

⚠️ **上一条"0 命中"是我 grep 范围与模式的双重疏漏**（2026-09-10 复核时被抓出来）：
MFMA 那条路**不在 `rocm_attn.py` 里，在它 import 的文件里**——
`ops/chunked_prefill_paged_decode.py:411` 调 `ops.paged_attention_rocm(...)`（HIP/`_rocm_C` 内核，
就是 8-30 那次崩的 `paged_attention_ll4mi_QKV_mfma4_kernel`），而且就算在正确的文件上，
我写的模式 `ops.rocm_` 也匹配不到 `ops.paged_attention_rocm` 这个名字。
⇒ **"0 命中"当时既没证明有、也没证明没有。** 正确的问法是"那个门在**我们的几何**上开不开"：

**运行级实测（容器内直接调那个谓词）** —— 门在 `platforms/rocm.py:416 use_rocm_custom_paged_attention`：

| 检查 | 结果 |
|---|---|
| `on_gfx906()` / `on_cdna()` | `True` / **`True`** ⇒ gfx906 走 CDNA 那一支门 |
| `hasattr(torch.ops._rocm_C, "paged_attention")` | **True** ⇒ MFMA 内核**确实在我们镜像里**（`nm -D _rocm_C.abi3.so` 有 paged 符号） |
| CDNA 支门的条件 | `head_size ∈ {64,128}` **且** `block_size ∈ {16,32}` **且** `gqa_ratio ∈ [1,16]` **且** 无 sliding window / 无 sinks |
| **我们的几何**（head 256、block 784、ratio 5） | **`False`** ⇒ ROCM_ATTN 在本模型**只会走 Triton 支路** |
| **Qwen3-0.6B**（head 128、block 16、ratio 2） | **`True`** ⇒ 正是 8-30 那次 device-side assert 的路径 |

⇒ 三条结论：
1. **(b) 的结论站住，但理由换成了实测的门求值**：ROCM_ATTN 在**我们这个 27B 混合模型**上不会碰 MFMA。
2. **项目记忆 `gfx906-attention-backend`（"gfx906 必须用 TRITON_ATTN"）是**几何相关**而非架构定律**：
   它是 head 128 + block 16 那一档的经验，不能外推到 head 256。已同步修正该记忆。
3. **`#63` 的主要风险被排除，残留风险换了位置**：不再是"MFMA 崩溃"，而是
   (i) `has_native_kv_cache_layout`（`chunked_prefill_paged_decode.py:24`，**按 stride 判 KV 是否 packed**，
   这正是 fork 那条"KV packed-view stride"踩坑的上游版本）在我们 hybrid  stride-padded 布局上会不会选错支路；
   (ii) prefill 侧 `context_attention_fwd` 与 head 256 的组合从未跑过。⇒ 冒烟必须**同时覆盖 prefill 与 decode**。

若属实，#21 的成本从"2–3 天写 HIP 内核"变成：先做一次 `--attention-backend ROCM_ATTN` 冒烟（**改配置，不改代码**），
再决定要不要把 fork 那约 530 行纯 Triton + 一个 `on_gfx906()` 门（`VLLM_ROCM_ENABLE_GFX906_SPLITKV`，默认关，
code-miner 复核：**不需要重编任何扩展**）作为补丁装进来。

**证据强度必须标清**：(a) 是**源码级已证**；(b) 现在是**门级已证、端到端未证**——
`use_rocm_custom_paged_attention` 在我们的几何上返回 `False` 是实测的，
但 `ROCM_ATTN` 整条后端（prefill 的 `context_attention_fwd` × head 256、hybrid 布局下的
`has_native_kv_cache_layout` 支路选择）在我们的 hybrid GDN 模型上**从没起来跑过**。

**§6 实测后的结算（写在这里，因为本节是当时的预判）**：16K = **28.56 tok/s**，落在本节自己划的
"25–32 ⇒ (a) 生效"区间 ⇒ **没有塌方，"手写 HIP attention"这一支按预判关单**。
但实测同时给出了本节没有预料到的**第二个数**：16K 档 attention 占 **15.9% 步时、
只用到 ≈9.0% HBM 峰值带宽**（§6.1(2)）——所以 (b) **不是因为塌方才顶上，是因为便宜通道上有 16% 的头寸**。
一支关、一支开：#21 关单，#63（`--attention-backend ROCM_ATTN` 冒烟 + 16K 内核分桶 trace）立项。

---

## 2. 成本重估：两条"搬代码"项其实不是搬代码

`LLMMB4` 在 `csrc/rocm/skinny_gemms.cu`；`q_gemm` 调参在 `csrc/libtorch_stable/quantization/gptq/q_gemm.cu`。
两者都是 **vLLM C++/HIP 扩展的翻译单元**，而我们**两条交付路径都不重编 vLLM 扩展**
（Path B 只用 hipcc 编 `docker/patches/p2_w4a16_gemv.hip`，其余是 python 覆盖 + 随 dist-packages 带的预编 `.so`）。

已核实的事实：`skinny_gemms` 在我们当年那次源码构建里**确实编进去了**
（`build/temp.linux-x86_64-cpython-312/CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny_gemms.hip.o`
与 `skinny_gemms_int4.hip.o`；`CMakeLists.txt:1496-1515` 只把 **gfx1250** 排除，gfx906 在 `VLLM_SKINNY_ARCHES` 内）
⇒ `ops.LLMM1` 符号是现成的，这也是 P0.5 能直接换调度的原因。

三条落地路线（成本差一个量级）：

| 路线 | 内容 | 评估 |
|---|---|---|
| **(a) P2 模式** | 新内核做成独立 TU，hipcc 编 + ctypes shim，不碰 `_C.so` | ✅ **已验证可行的通道**（P2 就是这么进镜像的）。`LLMMB4` 自包含 ⇒ 首选 |
| **(b) 重建 `_rocm_C_skinny`** | 只编那 2 个 TU 的小 target，替换对应 `.so` | 中等：干净，但要保证与已装的 torch_bindings 签名一致 |
| **(c) 全量重编 vLLM** | `/home/x/vllm-work/vllm-official` 还在 | ❌ 会把我们刚用两道断言清掉的 `/root/.cache` 泄漏问题再引一遍 |

⇒ **`q_gemm` 调参**改的是 `libtorch_stable` 主 `_C` ⇒ 只能走 (c)，或把内核**复制**成独立 TU 走 (a)——
后者实质等于**再写一次 P2**，不是"1 天搬代码"。
⚠️ 并且 digest 自己点出的张力要提前定价：**incumbent 变快会吃掉 P2 的 930 µs/step 净收益**，
两者不可叠加 ⇒ 若做，必须先测调参对那 192 次/step 的总收益，再重算 P2 净值。

`LLMMB4` 走 (a) 的额外好处：它作用于 **capture sizes 2/4/8**，与我们 §4 限制第 7 条
（"M>1 未覆盖"）正面对应，且**不与 P2 抢同一批调用**（P2 只管 M=1）⇒ 收益可叠加。

---

## 3. 重排后的执行顺序（"搬" / "建" / "测" 分开标）

排序依据 = **单位成本的期望收益**，不是 digest 的"改动行数"。`搬` = 现成物可直接取；
`建` = 自己写；`测` = 只花测量时间、结果决定后面要不要花建造时间。

| 序 | 项 | 判定 | 具体动作 | 类型 | 成本 |
|---|---|---|---|---|---|
| **0** | `is_gfx906` 地雷 + selfcheck 谓词守卫（§1.7） | 🔴 交付镜像里的真 bug | 改用 `rocm.on_gfx906()`；selfcheck 加"`dir()` 级存在性检查，不用 `hasattr`" | 建 | 20 min |
| **1** | 提交 + 把 5 个反向生成补丁纳入 git | ✅ **比外部任何一项都急**：`docker/patches/p2_w4a16_gemv.hip` 仍未跟踪 ⇒ **配方不可回放** | 需你授权；推前扫密钥 | 建 | 半天 |
| **2** | 长 ctx decode 归因 | ✅ **已完成（§6）** | 2K 33.36 / 8K 31.95 / 16K 28.56 tok/s，无塌方；副产品是首份 prefill 数 | 测 | 已花 |
| **3** | **#63**：16K attention 归因 trace + `--attention-backend ROCM_ATTN` 冒烟 | 🟢 **新立项**（§6.1(2) 给出 15.9% 分母、§6.1(3) 一个未归因的 2.2× 斜率跳变） | 一次 **auto 档 16K** trace 按内核分桶 ⇒ **顺带结掉姊妹文档 §4 第 5 条**（现有桶表全是 `pin=1425` 尺，auto 档内核级切分从未取过）；冒烟只改 CLI，**不动代码**，顺带否证/坐实"0.28 上 gfx906 必须 TRITON_ATTN"这条交付默认 | 测 | 20 min + 半天 |
| **4** | **#62**：GDN readout 幅值探针（C1 的正解） | ⚠️ 改判（§1.3） | 16K 满 ctx greedy 采 cast 前 `b_o` 与 `b_h` 的 `max\|·\|`，**必须覆盖 prefill/chunk 段**；与 `--mamba-ssm-cache-dtype fp16` 那 1.81 GB/卡**一起读**，不单独采纳容量收益 | 测 | 1 h |
| **5** | §5.5 那条 **1.34×**（他们 44.74 vs 我们 33.3） | 🟡 **最高价值未解线索**，且指向 60.2% 的桶 | 先按**他们的 C1 协议**（固定 128 token + C8 并发）在我们这台机复测，再决定要不要花一次重编去 A/B 那个编译宏 | 测 | 半天 GPU |
| **6** | #54146 补丁 + 它的 50 行回归测试 | ✅ 唯一"现成轮子 + 现成验收门" | **等序 4 出数再装**（幅值有余量 ⇒ 记"不阻断"；逼近 65504 ⇒ 一行改动装进去） | 搬 | 20 min（含在序 4 内） |
| **7** | `elementwise` 归因 | ✅ 已完成（§1.9b） | 结论已给；要动手则靶子改成"减 kernel 数量 + P2 侧省掉 `zero_()`" | 测 | 0（离线） |
| **8** | FP8 KV（#23） | 🔄 冒烟过，容量 **+58.8%**（§4），**质量与速度均未定价** | 只按**容量/并发/长 ctx 可交付性**判定；若要上生产必须先做质量档 | 测 | 待定 |
| **9** | `LLMMB4`（capture sizes 2/4/8） | ⚠️ 从"搬代码"改判"新增 TU"（§2） | 走 §2(a) P2 通道；门沿用 G1–G8，G4 扩到 n=2..8 全格；**不要重复定义 `on_gfx906`** | 建 | 2–3 天 |
| **10** | §1.6 prefill dequant-once→稠密 GEMM | ⚠️ 自建，参照物换成 `rdna_hybrid_w4a16.py`；现在有 §6.3 的 250–435 tok/s 做分母 | 先跑探针：`torch.mm` fp16 在 (8192,8192,5120) 的 TFLOPS；<15 ⇒ 先解决 rocBLAS 再立项 | 建 | 1–2 天 + 探针 |
| **11** | RCCL 单独立项（**定价，不解**） | ⚠️ 动作对、算术要改 | 按 overlap/union 语义给区间，不写"18% 可减" | 测 | 半天 |

**明确关单 / 不做（附理由，别只留个 ❌）**

| 项 | 处置 | 理由 |
|---|---|---|
| **#21 手写 HIP attention** | 🔴 **关单**（§6.2） | 16K 无塌方 ⇒ 恐慌前提不成立；同头寸（15.9%）有"改一个 CLI 开关"的通道（序 3），2–3 天手写是重复造轮子 |
| **`q_gemm.cu` gfx906 重调** | ⏸ **暂不做**，条件挂起 | 在 `csrc/` 且 `CMakeLists.txt:1252` 要求单 arch 重编 ⇒ 不是"1 天搬"。**只有序 5 复测证明差距真在 stock 算子里才重启** |
| **#22 M-RoPE / 融合 QK-RoPE** | ❌ 维持不做（§5.7） | 他们的实现由 `VLLM_ROCM_ENABLE_GFX906_QWEN36_FUSED_QK_ROPE_GATE` 门控、默认关，且**只为 Qwen3.6 那套形状服务**；我们没证据说 Qwen3.8 的 M-RoPE 路径有性能缺口 |
| **KVarN** | ❌ 不做 | digest 的 0.7× 是算术错（110/200 = **0.55×**），且 fork 40 篇文档零命中、无旁证 |
| GDN 融合 / B4 custom-AR / MTP / FP8 权重 | ❌ 与 digest 一致 | 维持不做 |

**与 digest 排序的主要分歧**：它把 C2 排在第 1（已完成）、把"重算桶表"排在第 3（已完成）、
把 `LLMMB4`/`q_gemm` 标成"半天/1 天可搬"（需要 C++ 通道：2–3 天 / 条件挂起），
并且完全没覆盖 §1.7 这个真 bug、§1.9b 的 `zero_()` 头寸、以及 §5.5 那条 1.34× 线索。
**digest 唯一排在我们前面、而我认为它排错了的，是 C1**：它把"幅值会不会溢出"当成已确认事实直接装一行补丁，
我们改成"先量再装"（序 4）。

---

## 4. FP8 KV（#23）—— 冒烟已过；容量收益实测 **+58.8%**，不是 +100%；质量未定价

受控对照（同容器、同配置、同 `Available KV cache memory: 2.66 GiB`，auto 时钟）：

| 臂 | attention block size | mamba page padding | GPU KV cache size | 16K 并发 |
|---|---|---|---|---|
| fp16（交付默认） | **784** tokens | 0.13% | **67,356** tokens | 4.11× |
| `fp8_e4m3` | **1568** tokens | 0.13% | **106,977** tokens | 6.53× |

⇒ **1.588×**（并发 4.11→6.53 同比例），**不是 fp8 名义上的 2×**。
原因可指到具体一行：`Setting attention block size to 1568 tokens to ensure that attention page size
is >= mamba page size` —— fp8 只压 half 的 attention 页，而 hybrid 对齐要求 attention 页 ≥ mamba 页，
于是块内 token 数被迫 **翻倍**（784→1568）来重新对齐；两臂的 mamba padding 都是 0.13%、可用 KV 都是
2.66 GiB ⇒ **GDN 状态页才是共同约束**，fp8 买到的是"被对齐折扣后的那半"。
⇒ 容量规划请用 **+59%**，不要用 2×。

**没有做的（必须标出来，别让人以为已过质量门）**：
- **质量未定价**。短上下文探针量不了 fp8 KV 的精度代价；要判必须走 L3 teacher-forced margin
  （见 [[kernel-swap-acceptance-criteria]]），不能看逐 token 等值。
- **速度未测**。fork 侧同样没测：`compatibility-matrix.md` 对 FP8 E4M3 KV 只写
  "Software path; MI50 has no native FP8 acceleration"，对 TurboQuant 只写
  "Capacity increased; latency did not clear default gate"，**整个 40 篇 phase 文档里没有任何一个
  KV 压缩的 tok/s 或 ms 数字**。⇒ "无原生 fp8 加速 ⇒ 可能反而更慢"在我们和他们身上都还是假设。
- 结论：#23 只能按**容量/并发/长 ctx 可交付性**立项，不能按性能立项。

## 5. fork 与上游核实（两个深挖代理 + 我自己读原文 + GitHub API 实查）

### 5.1 digest 引用的外部对象**全部为真**（API 实查，非推断）

| 对象 | 实查结果 | digest 说法 |
|---|---|---|
| `David-Lzy/vllm-gfx906` | ★1，last push 2026-09-07 | 存在 ✅（但**极小众**，价值全在那 40 篇 phase 文档） |
| `mixa3607/ML-gfx906` | ★337，2026-08-30 | 337★ ✅ |
| `huawei-csl/KVarN` | ★487，2026-06-22 | 487★ ✅ |
| `Igneous/vllm-gfx906-mobydick-kvarn` | ★5，2026-09-05 | 存在 ✅ |
| `NnnHU/sglang-mi50` | ★2，2026-08-29 | 存在 ✅（★ 数是 2 不是权威信号，但"17 个数值脚本可抄"这条待验） |
| `Intermountainh8ter/rocm-gfx906-sramecc-fix` | ★3，2026-07-19 | 存在 ✅（但 §1.2 已证我们报 `sramecc+`，前提不对） |
| `ttdxq/gfx906-vllm` | ★34，2026-08-25 | fork 的 phase-159 正是评审它并 **source-level no-go** |
| `ai-infos/triton-gfx906` | ★6，2026-02-03 | 存在 ✅，fork 从它构建 Triton 3.6 wheel |
| `vllm#54146` | **PR，open**，+64/−1，1 commit | ✅ 见 §5.2 |
| `vllm#54308` / `#55291` | issue，open / issue，open | ✅ 均真实 |
| `vllm#33811` | PR closed，"[Hardware][AMD] Add comments explaining gfx906 (MI50/MI60) is not supported" | ✅ |

### 5.2 🔴 #54146 独立证实了 §1.3 的**第三版**判断（也否掉了我的第二版）

PR 标题：`GDN readout: honor fp32 SSM state precision to avoid fp16 overflow → NaN`。正文逐字：

> "decode degenerates to a single infinitely-repeated token **when the SSM state cache is fp32
> (`--mamba-ssm-cache-dtype float32`)**. Once the recurrent state grows large over long context…
> greedy/low-temperature decoding emits one token (e.g. `!!!!…`) forever."

⇒ **触发条件正是我们的交付配置**（§1.3 实测 `mamba_ssm_cache_dtype = float32`，由 checkpoint 自动带出，
不是用户显式开的）。作者改的正是 `fused_sigmoid_gating.py:225` 那一行（+14/−1），**并附 50 行回归测试**
`test_fused_sigmoid_gating_readout_finite_under_large_fp32_state`：构造 `state × 1e6` 后断言 readout 有限。
⇒ 这既是"现成的轮"（补丁 + 验收测试都不用我们自己写），也解释了一个此前被我归错的症状：
QUICKSTART §8 把"输出全是 `!`"只归到 W4A16 桥接，而 **#55291 的症状也是 repeated `!`** ⇒ 排查表要加第二个病因。

### 5.3 digest 的三处实质错误

| # | digest 说法 | 实据 | 判定 |
|---|---|---|---|
| 1 | `VLLM_GFX906_LEGACY_QGEMM_ROWS_PER_BLOCK` 像运行期可调（列成"可配，默认 8"） | 它是 **CMake 构建选项**（`CMakeLists.txt:1202-1208`），且 `:1252-1255` 有 `FATAL_ERROR`：**只有 arch 恰好等于 `gfx906` 单目标**才允许开。fork 的 7 个**运行期** env 是另一套（`envs.py:128-134`），不含 ROWS_PER_BLOCK | ❌ 机制错：不是"改环境变量"，是"重编 `_C_stable_libtorch`" |
| 2 | fork 的 W4A16 相对我们的差异之一是"累加 half2 → **fp32**" | phase-123 + 代码：fp32 group scales/partial sums 的门是 `bit == 8`，**"W4 path untouched"**；修的是 packed-INT8 embedding/lm_head | ❌ 归错路径 |
| 3 | KVarN "吞吐 ≈**0.7×**（110 vs 200 tok/s）" | 110/200 = **0.55×** | ❌ 算术错（且 fork 40 篇文档里 **KVarN 零命中**，无从旁证） |

### 5.4 fork 语料能用什么、不能用什么（**口径警告**）

- 实际是 **40 篇 phase 文档**，不是 51；被引用为**承重证据**的 Phase 01/09/36/44/56/75/82/102/110/114/140/143/153/167
  **本身没有公开**（§5.5 里那句"GPTQ 52.8% / RCCL 24.1% / GDN 3.6%"就是二手转引未公开的 Phase 102）。
- **全部硬件结果在 4×32 GiB MI50 上**；我们是 2×16 GB RP VII ⇒ 他们每个 GiB 与 "KV tokens" 数字都要 **÷2 再 ÷2（卡数）** 才是我们的量级。
- **整个语料没有任何一处记录 GPU 时钟/功耗/频率** ⇒ 他们的 tok/s 与我们的"auto 交付时钟"口径不可直接换算。
- **他们从没测过单个 TP2 引擎的产能**：TP2 只作为 (i) TP4 内核 screen 的一个 rank、(ii) 2 卡 dev harness、
  (iii) `TP2x2 = 两个 TP2 引擎` 出现。也**没有任何 8K/16K/100K 的 decode tok/s**，只有"短固定 128"和"32K 前缀命中"两档。
- **没有任何 prefill tok/s**。⇒ digest 若暗示"他们有我们没有的长上下文性能数据"，不成立；§6 这次实测**是我们先测的**。
- 唯一可无损移植的是**方法学与纯 Python**：见 §5.6。

### 5.5 🟡 最高价值的未解线索：同为 TP2 / 27B，他们 C1 **44.74** tok/s，我们 **33.3**

`phase-115`（Qwen3.6 27B AWQ，**TP2，2×MI50**，100K，FP16 KV，v0.27→v0.28）：C1 `43.97 → 44.74 tok/s`，
32K `11.84 → 12.27`。我们（Qwen3.8 27B AWQ，TP2，2×RP VII，16K，fp16 KV，auto 时钟，开箱 serve）：
**33.3 tok/s**（§6 这次实测的 2K/8K 档同为 31.7–33.4 ⇒ 短 ctx 下 33.3 可复现）。差 **1.34×**。

我们的 P0 桶表（`logs/p0/p05_buckets_ab_on.log`，首行 `pin=1425.0MHz` ⇒ **钉频尺**，不是交付步；
auto 交付步 ≈30.0 ms ⇒ 下表每个百分比都**偏保守约 10%**）与 fork 的 phase-102 归因**结构几乎逐项对齐**：

| 桶 | 我们（µs/step 与 %wall，**pin=1425 尺**、33.01 ms 步） | fork（rank-0 GPU 时间，**时钟未记录**） |
|---|---|---|
| **W4A16 / GPTQ 线性** | **19 866.6 µs = 60.2%** | **52.8%**（C1） |
| RCCL | 7 604.4 µs = 23.0% | 24.1% |
| elementwise | 4 086.7 µs = 12.4% | 未列 |
| dense fp16 M=1 GEMV（P0.5 已换出） | 1 707.5 µs = 5.2% | — |
| GDN | 847.3 µs = 2.6% | 3.6% |
| **attention** | **513.6 µs = 1.6%** | — |

⇒ 两条直接推论：
1. ~~**#21 的头寸只有 1.6%**~~ —— **这条已被 §6.1 实测推翻，留着说明错在哪。**
   1.6% 是**短 ctx** 的数；attention 的 KV 读取随 ctx 线性涨，16K 实测头寸是 **15.9%**（+16.8% 上限）。
   被推翻的是"所以 #21 没有分母"这半句；**站住的**是 fork 那句判据的用法本身（"即使零成本也过不了门"
   在短 ctx 对我们成立）以及"先量分母再动手"的顺序。长 ctx 那一半见 §6.1(2)、处置见 §6.2。
2. **1.34× 的差距按桶只能落在 W4A16 线性里**（60.2% 的桶，要省 10.7 ms 才能把 30.9 ms 步压到 22.4 ms
   = 该桶的 54%）。fork 那个桶的实现是 `q_gemm.cu` 的 gfx906 重调（`BLOCK_KN_SIZE 256`、
   `dot22_8_f` 走 `__ockl_fdot2`、`v_bfi_b32` 解包、per-block C 零化），**编译进 `_C_stable_libtorch`**。
   注意：**我们和它调的是同一个 stock 算子 `ops.gptq_gemm(use_exllama=True)`**，差别只在有没有那个编译宏。

⇒ 但这条线索的**混淆项必须一并列出，不能当结论**：不同模型版本（3.6 vs 3.8）、不同栈
（他们 torch 2.11 + ROCm 7.2.1，我们 **更新的** 2.13 + 7.2.4 ⇒ 也可能是新版本回退）、
不同卡（32 vs 16 GB）、不同 ctx（100K vs 16K）、且他们的 44.74 出自 **128-token** 固定预算而我们主档是 64 token。
**便宜的甄别动作**：先按他们协议（C1 固定 128 token + C8 并发）在**我们这台机**复测一遍，
再决定要不要花一次重编去 A/B 那个编译宏 —— 别反过来。

### 5.6 值得直接拿的三样（都不需要重编）

| 物 | 位置 | 为什么是"搬"而不是"建" |
|---|---|---|
| **#54146 补丁 + 它的 50 行回归测试** | 上游 PR | 一行改动 + 现成验收门；§1.3 |
| **mock-based 分派证明测试** | `tests/model_executor/layers/test_rocm_unquantized_gemm.py` | 无 GPU、无 `_rocm_C` 链接即可证"该形状走了哪个内核" ⇒ 正是我们 G-P2 路由门的写法 |
| **Router-lab A/B + replay + summarize harness** | `tools/gfx906-router-lab/` | 与架构无关，含亚 ms→10 ms 延迟桶和 2 ms 开销门 |

另有两条**方向性提醒**（不是代码）：`AGENTS.md:94-95`"一次性内核基准不放 `tests/`"；
以及他们把 **v0.28 在真实负载上比 v0.23 慢 78.1%（4.31×）** 这个负面结果**公开发布**并据此回滚晋升 ——
对应的教训是 phase-168 那句："Backend changes should not be promoted from short decode micro-workloads alone."

### 5.7 明确**不是**轮子、不要搬的

- `LLMMB4` / legacy QGEMM 重调：都在 `csrc/`，且 `CMakeLists.txt:1252` 要求单 arch 重编 ⇒ §2 路线 (a)/(c) 的成本。
- Triton `scf.if` fat-pointer `intersect` 补丁：**只为他们 Qwen3.6 融合 QK-RoPE 服务**（`#22`，已降级）。
  我们跑 Triton 3.8.0 且不走那条融合路 ⇒ 与我们的决策无关，"3.8 是否已含"这个 digest 问题可以**不答**。
- 他们的 `docker/Dockerfile.gfx906-v028-release`：**单阶段、把整个 Triton clone 和 vLLM 源码留在镜像里**，
  且必须重编 `_C_stable_libtorch` + Rust frontend ⇒ 与我们"不重编 + 两路径瘦身"的配方根本冲突，只可借其
  **构建期自检断言**与"脏树拒绝构建"两条纪律。

---

## 6. 长上下文 decode 实测（#21 的分母；交付镜像 + 全新容器 + 开箱 `gfx906-serve`，auto 时钟）

冷启动 `READY_AFTER=720s`；`Setting attention block size to 784 tokens`（fp16 臂）；
`resolved_model_id=Qwen3.8-27B-OBLITERATED-AWQ-W4A16`；64 token / `temperature=0` / `ignore_eos` / 流式，每档 3 次。

| ctx（prompt_tokens） | 档 | decode tok/s | ms/step | prefill tok/s |
|---|---|---|---|---|
| ~2 050–2 067 | 2K | 32.24 / **33.36 / 33.36** | 31.0 / 30.0 / 30.0 | 331.0 / **435.2 / 434.4** |
| ~8 164–8 242 | 8K | **31.65 / 31.97 / 31.94** | 31.6 / 31.3 / 31.3 | 347.9 / 342.5 / 333.7 |
| ~16 188–16 217 | 16K | **28.57 / 28.44** / 28.67† | 35.0 / 35.2 / 34.9† | 257.7 / **249.2 / 250.7** |

`A2B SUMMARY`（warm 均值）+ 收尾扫描：`2K 33.36 / 8K 31.95 / 16K 28.56`，
`fatal_signature_hits=0`、`stat_lines=16`、`A2B_VERDICT=OK rc=0`。
† 16K 第三次的原始 prompt 超了 `--max-model-len 16384`（HTTP 400），客户端按设计缩 10% 重投 ⇒
该行是 **14 701 token** 而非 16K；SUMMARY 的 28.56 是 16 188 与 14 701 两次的均值（差 0.06 ms，可忽略）。

### 6.1 判决：没有塌方，但**我上一版"头寸只有 1.6%"的推论被这次实测推翻了**

**(1) digest / fork 那个"长上下文必塌"的恐慌，对我们不成立。**
2K → 16K 只掉 **−14.4%**（33.36 → 28.56），8K 档更是只有 −4.3%。
他们未修复路径在 32K 是 **0.238 / 1.353 tok/s**（−99.3% 量级）。
⇒ §1.11(a) 成立：我们本来就在 64 段 split-3D 路径上，`VLLM_ROCM_ENABLE_GFX906_SPLITKV` 那个开关**不是我们的缺口**。

**(2) 但 16K 档暴露了真实的 attention 头寸。** 逐步算：

| 量 | 值 | 算法 |
|---|---|---|
| 每 token 的 full-attn KV | 64 KiB（TP2 ⇒ **32 KiB/卡**） | 4 kv head × 256 head_dim × (K,V) × 2 B × **16** 层（`layer_types` 只有 1/4 是 full） |
| 2K→16K 增量步时 | **+5.04 ms** | 29.98 → 35.02 |
| 同增量的 HBM 地板 | **0.453 ms**（= 442 MiB） | 14 149 tok × 32 768 B ÷ 1024 GB/s |
| ⇒ attention 实际带宽利用率 | **≈9.0% peak** | 0.453 / 5.04 |
| 交叉验证（P0 桶表，短 ctx，**pin=1425 尺**） | **≈12.8% peak** | 地板 65.6 µs（2 050 tok × 32 KiB）÷ 实测 `attention` 513.6 µs |

⚠️ **这把尺是混的，必须标出来**：+5.04 ms 是 **auto 交付时钟**下测的，而 513.6 µs 来自
`pin=1425` 那张表。短 ctx 的 attention 是延迟受限、随时钟走的 ⇒ 换算到 auto 约
`513.6 × 30.0/33.0 ≈ 467 µs`，代进去 attn@16K = 5.51 ms、份额 **15.7% vs 15.9%**。
**结论不受影响（差 0.2 pp），但正式数只能由 #63 那次 auto 档 16K 分桶 trace 给。**

带宽口径 = **十进制 1024 GB/s**（MI50/RP VII 规格 1.024 TB/s），脚本 `logs/a2b/calc_attn_headroom.js`。

**两个独立口径给出同一个数量级的利用率 ⇒ 这条 attention 路径确实远未贴着 HBM 跑。**
把它记进交付步就是：16K 时 attention ≈ **5.5 ms / 35.0 ms = 15.9% 的步时**，
理论上限（全压到带宽地板）是 **33.4 tok/s，+16.8%**；而 2K 时整个桶只有 1.6%。
**§5.5 推论 1 里"#21 的头寸只有 1.6%"只在短 ctx 成立，长 ctx 要乘 10 —— 那句作废，以本节为准。**

**(3) 新发现的异常，比绝对值更值得追：斜率超线性。**
2→8K 是 **0.215 ms / 1k token**，8→16K 是 **0.465 ms / 1k**（**2.2×**）。
decode 期唯一随 ctx 增长的工作就是 full attention（GDN 状态、W4A16、RCCL、lm_head 都是 O(1)），
所以 KV 读取本应严格线性 ⇒ 这个 2.2× 跳变**尚未归因**。同档三次离散只有 0.23 tok/s，不是噪声。
候选：KV 足迹变大后的 HBM 页/TLB 行为、每段走更长连续块时段内 reduce 效率、L2 抖动。
⇒ **判据（一次 16K 的 auto 档 trace，20 min GPU）**：按内核分桶读 attention 的 16K 实际 µs，
若 ≈5 ms 就是它，若明显小于 5 ms 则这 5.04 ms 落在别处、上面整段推论都要重做。

### 6.2 #21 的处置（这次实测唯一要回答的问题）

- **手写 HIP attention：仍然关单**，但**理由换了**。旧理由是"头寸 1.6%"（错，已作废）；
  新理由是"**同样的头寸有便宜十倍的通道**"—— 16K 的 +16.8% 天花板要 2–3 天手写内核（我们在 0.28
  上还没有一次成功的自写 attention 内核记录），而 §1.11(b) 说 0.28 的 ROCM_ATTN 已是纯 Triton
  且支持 head 256 ⇒ 换后端是**一个 CLI 开关**，且能直接吃到 fork 那份 ~530 行 SplitKV。
- 顺序因此固定：**#63（ROCM_ATTN 可行性冒烟 + 16K 内核分桶 trace）先跑**，它同时回答
  (3) 的归因和"这个 15.9% 能不能用换后端拿到"。拿到 ⇒ #21 永久关单；拿不到 ⇒ 再谈手写。
- **不要做的事**：因为"16K 有 15.9%"就去改 `--max-model-len`、加段数或重编 `_C`。
  并发 1 的 16K 单请求延迟，优先级低于 §5.5 那条 1.34× 的**整体**差距（那才是 60.2% 的桶）。

### 6.3 附带产出的两组新基线数字（此前从未测过）

- **prefill tok/s：434.8 / 338.1 / 249.9**（2K/8K/16K warm）。这是交付路径第一份 prefill 数，
  也是 §1.6 那句"prefill 上游已实现"第一次有了自己的分母；fork 语料里**一个 prefill tok/s 都没有**。
- **TTFT：4.7 s / 23.5–24.7 s / 58.7–64.9 s**。16K 首 token 一分钟 —— 若要做长上下文的可交付性叙事，
  这是比 tok/s 更先撞墙的指标（且它同样压在 #23/FP8 的**容量**收益上，不压在 §6.1 的延迟收益上）。
- 冷启动 `READY_AFTER=720s`、块大小 784（fp16 臂）与 §4 一致；日志首行 `A2B_ABORT docker run failed`
  是修挂载前那一次失败尝试的残留，**本次运行的记录从第二行 `=== A2B LONG-CONTEXT DECODE PROBE` 起**。
