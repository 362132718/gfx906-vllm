# MoE 性能归因 —— 同口径三点矩阵（2026-09-11 夜）

## 结论一句话

`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` 在 2×MI50 单流 decode：**27.08 → 46.03 tok/s（+70.0%）**，
全部来自两处**零重编**改动。对照 llama.cpp `llama-bench` 的 58.28 tok/s，**仍落后 1.27×**。

## 矩阵

全部同一口径：`MM=1`（开视觉）· `MAX_LEN=16384` · `MAX_NUM_SEQS=32` · `PREFIX_CACHE=1` ·
`GMU=0.92` · TP2 · 图模式 · ctx=1600 · 生成 64 token · **warmup + REPS=3 取中位**。

| 臂 | m==1 补丁 | MoE 调优配置 | 步长 | **单流 tok/s** | 8 路聚合 | reps (ms) |
|---|---|---|---|---|---|---|
| `mx_base` | OFF | OFF | 36.93 ms | **27.08** | 70.4 | 36.10 / 36.93 / 38.18 |
| `mx_m1` | **ON** | OFF | 26.38 ms | **37.91** | **37.8** ⚠ | 25.44 / 26.38 / 27.48 |
| `mx_both` | **ON** | **ON** | 21.72 ms | **46.03** | 81.1 | 21.39 / 21.72 / 22.18 |

⚠ `mx_m1` 的 conc8 是非单调异常值（70.4 → 37.8 → 81.1），未定位；单流三点单调且离散度 ≤3%，
故单流归因可信，conc8 只用 `mx_base` 与 `mx_both` 两个端点。

## 归因

| 改动 | 步长变化 | 单流提升 |
|---|---|---|
| **m==1 快速路径**（`docker/patches/layers_utils.py`） | 36.93 → 26.38 ms（**−28.6%**） | 27.08 → 37.91（**+40.0%**） |
| **MoE kernel config**（`moe_tuning/configs/*.json`） | 26.38 → 21.72 ms（**−17.7%**） | 37.91 → 46.03（**+21.4%**） |
| **合计** | **−41.2%** | **+70.0%** |

## 两处改动是什么

### 1. m==1 快速路径
`shared_expert_gate = ReplicatedLinear(hidden, 1)`（`qwen3_next.py:152`）每层一个，
被 P0.5 的 `m % 4 == 0` 挡在门外 ⇒ Tensile 用 `MT64x64x16` 平铺 GEMM 做**标量点积**（151 µs/次 × 40 层）。
改法：**跳过 BLAS，mul + fp32 归约**。数值 rel err `3.489e-04`，单次 43.6 µs vs 270.3 µs。

**⚠ 不要对 `weight` 调任何张量方法**（如 `weight.reshape(-1)`）—— 它是 `BasevLLMParameter`，
经 `parameter.py __torch_function__` 派发；在 AOT/dynamo 追踪下会变成
`'NoneType' object has no attribute 'size'`，**直接炸掉 `MM=1` 引擎初始化**。
必须写成广播式 `(x_view * weight).sum(-1, dtype=torch.float32)`。

### 2. MoE kernel config
调优器产出 `E=256,N=256,device_name=AMD_Radeon_Pro_VII,dtype=int4_w4a16.json`：
```json
{ "triton_version": "3.8.0",
  "1": { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2,
         "waves_per_eu": 0, "SPLIT_K": 1 } }
```
调优器选**极小 tile + `num_warps=1`**，与 M=1×8 专家的形状吻合，而默认回退配置显然不是。

## 副作用（正向）

**两处修复顺带把 Mamba cache block 数从 33 提到 ≥48**（`MM=1` + 前缀缓存开 + gmu 0.92）。
原因：去掉 rocBLAS GEMM 的工作区分配 + 更小的 tile ⇒ 峰值显存下降 ⇒ 多出的显存变成 KV/Mamba 容量。
**⇒ `MAX_NUM_SEQS=48` 只有在两处修复都生效时才起得来。**

## 测量方法（今天踩了三个坑，都已修）

1. **前缀缓存**：同一 prompt 重复测取 `min()` ⇒ 命中缓存那次，prefill 虚高 6–32×。
   修法：不同非ce 的同长 prompt + `enable_prefix_caching` 可控。
2. **缺 warmup**：`tA` 是加载后第一个请求，吸收首次开销（Triton JIT、分配器增长），
   而 `step=(tB−tA)/(N−1)` ⇒ **step 被系统性低估**（曾报出 8.89 ms/step = 112 tok/s，
   而分桶实测单卡 device-busy 就有 ~18 ms —— 物理不可能）。
   修法：**先用第三个 prompt warmup**，再 **REPS=3 取中位**。
3. **trace 步长 ≠ 真值**：torch profiler 的 CUPTI 记录会串行化执行，trace 给 72.6 ms 而真值 23.76 ms。

**纪律**：报任何 prefill/decode 数字前，先确认 ①无缓存命中 ②有 warmup ③时钟口径取自 hwmon。

## 仍未做

- **conc8 异常值定位**（`mx_m1` = 37.8，非单调；仅归因用中间点，不影响交付）
- batch 8 单独调优（约 2 h，当前 batch-1 配置已让 conc8 +15%）
- P7 批量进镜像

---

# 附：GDN readout 风险核查 —— **结论：在本镜像上不成立**

上游 vLLM #55291 / #54146 描述：SSM state cache 为 fp32 时，GDN 线性注意力的 readout
累加值超出 fp16 上限 65504 ⇒ 写入 fp16 缓冲变 inf ⇒ `inf/inf = NaN` ⇒
**贪心解码退化成单 token 无限重复**，且**只在长上下文出现**。
QUICKSTART 此前把它列为「本镜像正是该配置，但尚未复现」。我们的目标 MoE 模型同属该混合家族。

## 三层证据

### ① 内核级：`gdn_verify.py` → **44/44 PASS**
```
TOTAL 44/44 PASS
ALL GDN KERNELS MATCH NAIVE REFERENCE     0 FAIL, 0 nan
```
覆盖 `causal_conv1d_fn/update`、`fused_post_conv_prep`、`chunk_gated_delta_rule`、
`fused_recurrent_gated_delta_rule_packed_decode`，均与 fp32 naive 参考对齐。
另核实 `container_src/fla_ops/fused_sigmoid_gating.py:102` 的 h 状态**本来就是 fp32**
（`b_h = tl.zeros([BV, BK], dtype=tl.float32)`）。

### ② 长上下文无 NaN 指纹：`gdn_longctx_probe.py`
14,000 token 上下文、短/长对照、换 nonce 复测 → `longest_run=1`，无 collapse。
（此探针的填充词是周期重复的，输出会复读填充词 ⇒ 只能证明"没崩"，证明不了"连贯"。）

### ③ 长上下文 + 短问题（决定性）：`gdn_shortq_probe.py`
3 个问题 × 短/长上下文、同一引擎、贪心：
```
short q0/q1/q2   ctx≈275   matched=3/3
long  q0  ctx=14022  len=9    matched=True   '<think></think>391<|endoftext|>'
long  q1  ctx=14019  len=7    matched=True   '<think></think>Blue<|endoftext|>'
long  q2  ctx=14017  len=246  matched=True
SUMMARY short=3/3 long=3/3 collapsed=none
```
1. **14K 上下文下全部答对**（17×23→`391`，晴天天色→`Blue`）
2. **`len=9` / `len=7`** ⇒ 答完**自然触发 EOS**，非被 cap 截断 ⇒ 模型健康
3. **`collapsed=none`** ⇒ 无无限重复

## 测试设计上的两个坑（都踩过并已修）

- `N_GEN=96` 太小：`think` 前言吃光配额 ⇒ 短上下文"全失败"是**假象**，不是模型问题。改 256。
- `ignore_eos=True`：模型答完后无限喷 `<|im_start|>`（run=39/42），**看着像 collapse 实为采样参数**。
  改 `ignore_eos=False` 才能观察自然终止。

## 交付影响

QUICKSTART 里该条目应从「已知上游病因，尚未复现」**升级为「已验证不触发」**（三层证据），
且**不需要**改 `fused_sigmoid_gating.py` 的 dtype ⇒ **不需要额外改动进镜像**。

---

# P7 交付构建 —— 已完成

## 产物

```
vllm-gfx906:latest    c31a8b65273c   69.8 GB   29 层
vllm-gfx906:rebuilt   5894eb537a67   69.8 GB   （路径 B 中间产物）
```

## 交付镜像的口径数字（全新容器，非复用）

```
step 21.61 ms  →  46.28 tok/s   prefill 1129 tok/s
  reps=3 [21.24, 21.61, 21.99]        ±1.8%
conc8 agg 84.2 tok/s (per-stream 10.52)
GPU KV cache size: 77,451 tokens（16K → 4.73× 并发）
```
`gfx906-selfcheck` → **SELFCHECK PASSED**（全绿，含新增的两条 MoE 检查）。

## 三个会静默产生错镜像的坑

1. **抽取源不是默认的 `vllm-gfx906`，而是 `a2b-t1`。**
   默认容器陈旧：其 `triton_decode_attention.py` 仍是**未拔雷**版、`qwen_gdn_linear_attn.py`
   是 76603 字节的旧版。**P0 与 P3 两个补丁只装在 `a2b-t1`。**
   第一次 `prepare` 用默认容器 ⇒ tar 带旧文件 ⇒ 路径 B 在第 18/19 步断言失败。
   ⇒ **prepare 前必须跑逐条断言体检**：`moe_tuning/assert_probe.sh`，21/21 全过才动。

2. **宿主 `docker/` 与仓库 `docker/` 是两份，且宿主陈旧。**
   宿主 `Dockerfile.rocm_gfx906` 还在断言 `grep -q "is_gfx906"`（已被 P0 拔掉的谓词）；
   宿主 `entrypoint/selfcheck.sh` 整段缺失 `hasattr` 回归守卫。
   方向是**仓库 → 宿主**，同步后需 md5 三方一致。

3. **两条构建路径的 `entrypoint`/`patches` 来源不同。**
   路径 B 从**容器**抽（`prepare [3/7][4/7]`）；路径 A 从**仓库的宿主副本** COPY。
   **两边都要喂**，只喂一边必有一条路径是旧的。

## 新增断言（纪律：自检必须覆盖新改动）

Dockerfile 两条路径各加 2 条；`selfcheck.sh` 加 2 条 `chk`：
- `MoE m==1 scalar-gate fast path`
- `MoE tuned kernel config for this device`

## 构建时延（供排期）

| 步骤 | 耗时 |
|---|---|
| `prepare`（`FORCE=1`，含重打 7.3 GB tar） | **~5 min** |
| 路径 B（19 步，含解 tar + 断言） | **~9 min**（`exporting layers` 占 325 s） |
| 路径 A（基于 `:rebuilt`） | **~20 s** |
| 全新容器 `gfx906-selfcheck` | 全绿 |

---

# P8 交付收尾 —— 已完成

## 交付物

```
/home/x/vllm-gfx906-c31a8b65.tar   22.07 GB
  vllm-gfx906:latest = c31a8b65273c（69.8 GB / 29 层）
```

**可恢复性已验证**：先 `docker rmi vllm-gfx906:latest`，再 `docker load -i <tar>` ⇒
`Loaded image`、**ID 仍是 `c31a8b65273c`**。不是"文件存在"而已。

## 干净容器验收（镜像自带 `gfx906-serve`，默认参数）

```
serve RESULT=STARTED    GPU KV cache size: 57,344 tokens（16K → 3.50× 并发）

C1 文本  TEXT_HAS_391: True          7.1s   ✓
C2 图片  KEYWORD_SCORE: 6/6          2.6s   ✓ 与脚本画的三个对象完全吻合
C3 视频  VIDEO_KEYWORD_SCORE: 3/3    3.1s   ✓
```

**⇒ `gfx906-serve` 的默认 `MAX_NUM_SEQS=48` / `GPU_MEM_UTIL=0.90` 对 MoE 模型可用。**
原先担心的「默认会因 Mamba block 不足起不来」没有发生 —— 因为两处修复把 block 数从 33 提到 ≥48。
QUICKSTART 那条注记应作为**条件性警告**保留，而非"默认不可用"。

## 验收脚本的两个坑（都是 served-model-name）

`serve.sh` 的 `--served-model-name` 取自**模型目录 basename**：
1. 探针硬写 `model:"moe"` ⇒ 全线 404
2. serve 成 `moe`，但 **`a3_client.py` 把 model 硬编码为 `qwen38`** ⇒ 图片探针 404
3. serve 成 `qwen38`（并把 `qwen38` 传给视频客户端）⇒ 三条全过

⇒ **探针的 model 名必须与 `--served-model-name` 对齐**。另：文本探针 `max_tokens=64` 会被
`think` 前言吃光，改 256 才看得到答案。

## 未做（需要用户）

- **#24 的「对方那台机」** —— 需目标机访问方式。本地可做的（开箱 selfcheck + 三模态 + MoE + 长 ctx）已全部完成。
- **commit** —— `docker/patches/p2_w4a16_gemv.hip` 仍未跟踪，是 Path B 编 `.so` 的唯一源。

---

# 两个模型的回归验收 —— 都在这版镜像上通过

## 为什么 27B 必须重测

上一版 `:latest` 是 **39 小时前**从**陈旧的 `vllm-gfx906`** 构建的 ⇒
**它不含 P0 拔雷、也不含正确的 GDN 文件**。27B 的 33.4 tok/s / 图像 6/6 出自 9-02 的 C2 验收，
**不覆盖这版镜像**。今天这版 `c31a8b65` 是第一次把 P0 + P3 + P0.5 + m==1 + MoE 调优凑齐。

## 结果（均用镜像自带 `gfx906-serve`，默认参数）

| 模型 | 档位 | decode | KV tokens | 文本 | 图片 | 视频 |
|---|---|---|---|---|---|---|
| **Qwen3.8-27B dense** | T1 | **34.18 tok/s**（基线 33.4，+2.3%） | **67,356** | ✓ | **6/6** | **3/3** |
| **Qwen3.6-35B-A3B MoE** | T2 | **46.28 tok/s** | 57,344 | ✓ | **6/6** | **3/3** |

- 27B 日志：`Using WNA16ExllamaKernel for CompressedTensorsWNA16` ✓
- **27B 的 67,356 tokens 与 handover 基线逐位一致** —— 干净的回归信号
  （此前用陈旧容器对比跑测得 59,467，差 −11.7%，当时归因为容器差异；现在确认新镜像就是 67,356）
- 两个模型都**不需要任何额外调参**，`MAX_NUM_SEQS=48` / `GPU_MEM_UTIL=0.90` 直接可用

## 验收脚本

`p8_accept_27b.sh`（27B）/ `p8_accept_c.sh`（35B）—— 都走**镜像自带 `gfx906-serve` + 默认参数**，
不是我们的测试框架。踩过的两个坑记录在上一节（served-model-name 对齐、文本探针 token 预算）。





## 复现

```bash
# 矩阵（约 35 min，需独占 GPU）
bash moe_tuning/ab_matrix.sh          # 读 moe_tuning/abmatrix.log

# 单点测量
bash p5_decode.sh "arm:EAGER=0;MAX_LEN=16384;MAX_NUM_SEQS=32;MM=1;GMU=0.92;N_GEN=64;CTX=1600;PREFIX_CACHE=1;REPS=3"
```
