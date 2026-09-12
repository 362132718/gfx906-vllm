---
title: "gfx906 vLLM 汇总报告：改了什么、为什么改、测试数据、下一步（截至 2026-09-10）"
date: 2026-09-10
scope: 全项目（环境层 → 功能层 → 性能层 #20-P0/P0.5/P1/P2 → 交付层）
status: 交付镜像可用（`:latest 2fba84a62576`，23 ok / 0 FAIL）；#20-P2 已进镜像并默认开启；58 项改动未 commit；#18 分发未落地
---

# gfx906 vLLM 汇总报告（2026-09-10）

> 这份文档是**索引 + 结论**，不是原始证据。原始证据在四份分报告里，每份都带复现命令与日志路径：
> `2026-09-04-gfx906-w4a16-p0-report.md`（归因）、`2026-09-08-gfx906-p05-report.md`（skinny GEMV）、
> `2026-09-09-p1-report.md`（三原型 + 标尺重标定）、`2026-09-09-p2-report.md`（两族落地 + 七道门）。
> 计划本体在同目录，spec 在 `../specs/2026-09-03-gfx906-w4a16-wave64-kernel-design.md`。

## 0. 一页总览

**目标**：给同型号显卡（gfx906 / GCN 5.1 / wave64：Radeon Pro VII、MI50、MI60、S7150、Z100）用户
一个**开箱即用的完整 Docker 镜像**，跑 Qwen3.8-27B AWQ-W4A16，TP2 / 2×16 GB。这条目标决定了本报告
里每一个改动的完成标准都是"能进镜像"，不是"我们机器上跑通了"。

**当前交付态（真实开箱，非钉频、非离线 harness）**：

| 项 | 值 |
|---|---|
| 镜像 | `vllm-gfx906:latest` = `2fba84a62576`（Path B `:rebuilt 3db9747fd931` → Path A `final`） |
| 体积 | 归档 21.95 GB（save 流 22.07 GB，pigz 只省 0.6%），**解包需 ~70 GB 磁盘** |
| 开箱自检 | 全新容器 `gfx906-selfcheck` → **23 [ok] / 0 FAIL** |
| 起服务 | 一条 `gfx906-serve <模型目录>`；冷启动 710 s（编译 384 s）/ 同容器二次 420 s（编译 64–67 s） |
| decode batch1 | **33.3 tok/s**（P2 开，默认）/ 32.3（`VLLM_WNA16_P2_GEMV=0`）/ 32.4（P2 之前） |
| decode 8 并发 | 66.3 tok/s 合计（每流 8.3） |
| prefill | 2K 442.8 / 8K 346.6 / 16K 235.0 tok/s（eager 口径） |
| KV 容量 | 64.3k–67.4k tokens @16K（冷高 warm 低 3.6%，未定位）/ 纯文本 18K×5 = 92 842 |
| 多模态 | 图像与视频端到端可用（含 CUDA graph），6/6 与 3/3 关键词判定 |

**decode 累计 14.6 → 33.3 tok/s = ×2.28**，四段各自独立可归因、且都是**同一探针**读数：

| 阶段 | tok/s | 手段 | 报告 |
|---|---|---|---|
| eager | 14.6 | 起点（Triton W4A16 曾测到 3.9，属配置陷阱） | — |
| +CUDA graph | 22.0 | capture-safe `gptq_gemm_out`（预分配 + 调用前 `zero_()`）+ `FULL_DECODE_ONLY` | B3 |
| +skinny M=1 GEMV | 32.4 | 10 行门谓词，稠密 fp16 M=1 GEMM 从 Tensile 串行 K 循环换到 `ops.LLMM1` | #20-P0.5 |
| +手写 wave64 GEMV | 33.3 | 两族（8192/7168）M=1 GEMV 换自研内核 + 运行期分派 + kill-switch | #20-P2 |

**必须先读的一条口径警告**：本项目混用三把尺——① 1425/1000 MHz 钉频 + 离线 harness（P0/P0.5），
② 1700/1000 钉频 + CUDA graph 单内核微基准（P1/P2 的 G-P0…G-P5），③ **auto 时钟 + serve 路径**
（交付态，唯一可对外引用的数字）。**跨档外推已被实测证伪一次**（按 sclk 比例外推 30.3→36 tok/s，
实测只有 32.4/32.6，偏乐观 ~10%：decode 里访存受限的部分不随 sclk 缩放，且 serve 多一层
API/调度/分词）。本报告每个数字都标了口径；**不要把它们互相换算**。

---

## 1. 我们做了什么（四层）

### 1.1 环境层：让 0.28 栈在 gfx906 上不崩

官方 ROCm 对 gfx906 的支持停在 6.4.x，我们跑的是 **ROCm 7.2.4 + torch 2.13.0+rocm7.2 +
triton 3.8.0 + vLLM 0.28.0（源码构建）**。这一层的五件事全部是"缺东西/走错路"的修复，没有优化：

1. **RCCL 换 gfx906 构建** — 发行版 `librccl` 里没有 gfx906 kernel，TP2 初始化直接 segfault。
2. **Tensile gfx906 补充文件（156 个 `.hsaco/.co/.dat`）** — 缺失时普通 matmul 段错误。
3. **Triton BLOCK=32 / `VLLM_TRITON_ATTN_NUM_SEGMENTS=64`** — 上游按 wave32 GPU 给 gfx906 选了
   BLOCK=16，wave64 上半宽浪费。
4. **rocBLAS GEMV 段错误规避** — gfx906 上 fp32/fp64 `matrix×vector` 直接 core dump，改成走 GEMM；
   以 `.pth` 注入实现"全进程启动即生效"（不是 sitecustomize，容器 ENV 语义踩过坑）。
5. **`HSA_OVERRIDE_GFX_VERSION=9.0.6`** + 编译缓存按 W4A16 内核选择隔离（见 §2 表第 10 行）。

同时确立一条架构决策：**不再从源码编 PyTorch/Triton/FA**（早期尝试过，版本对不上且从未真正使用），
改为"官方 ROCm 7.2.4 complete 基底 + 打补丁 + 源码装 vLLM"。这条配方就是 `docker/Dockerfile.rocm_gfx906`
的两条路径（Path B `--target rebuild` 离线重建 → Path A `--target final` 叠 entrypoint）。

### 1.2 功能层：让 27B 真的能用

- **A1 GDN 正确性**：44/44 GDN 内核 vs naive fp32 参考全过 ⇒ 0.28 栈不需要 GDN fallback。
- **A2 长上下文基线 + 内核形状覆盖**（prefill/decode/KV 三个方向的起点数字）。
- **A3 多模态**：这是**交付硬要求**（测试模型本身是 Qwen3.5-VL 架构，27 层未量化 vision tower）。
  图像、视频端到端验通；`--limit-mm-per-prompt` 关掉 mm 只是性能测试隔离手段，**不得进交付配置**。
  ViT 数值抽查：GPU fp16 vs HF 参考 rel-L2 1.24e-2、实现级差约 0.8%、语义全对 ⇒ 非阻断。
  **一处真 bug 是客户端用法**：serving 接口要求 `{"type":"video_url",...}` 且必须是真实 mp4，
  帧列表写法直接 400。
- **B3 CUDA graph**：capture-safe 的 `gptq_gemm_out`（C++ 3 文件 + bridge 持久 slab）。
- **B4 custom all-reduce — 判定不做**：内置实现在本硬件可用且 52 轮变尺寸压测无 stale 竞态，
  但**性能与 RCCL 持平**。Z100 作者拿到 +8.5% 依赖 xGMI，RP VII 的 PCIe P2P 上不存在这个差距
  ⇒ 交付保持 `--disable-custom-all-reduce`（少一个 stale-data 风险面）。

### 1.3 性能层：#20 的四个阶段（P0 归因 → P0.5 换调度 → P1 原型 → P2 落地）

见 §3.2–§3.5。**方法学上是这份报告最有复用价值的部分**：先测"该不该做"（P0 归因 + busy ratio 路由），
再做"最便宜的 20 ms"（P0.5，一处谓词），然后才写内核（P1 三原型 171 组合），最后按实测缩窄范围
（P2 只做两族）。**中途有三次"设计假设被实测推翻"**，都记在 §4。

### 1.4 交付层：让别人也能用

- `docker/entrypoint/{serve.sh,selfcheck.sh,QUICKSTART.md}` → 镜像内 `gfx906-serve` / `gfx906-selfcheck`。
  每条默认值都带实测理由（§2 表第 7 行），首行 `[serve] ...` 打印解析后的真实配置。
- 两条构建路径都带**静态断言**（补丁必须 COPY 进去、`/root/.cache` 必须为空、符号必须存在），
  配 `prepare_build_context.sh` 与运行期 `selfcheck.sh`，四道一致。
- 编译缓存**不随镜像分发**（601 MB 只省 ~5 分钟，且可能与内核选择串台）。
- 镜像归档与 `docker load` 回读验证已做一次（这是 #18 的第一份实测数据）。

---

## 2. 改了什么、为什么改（逐项）

**内核/语义级改动（性能层的三个真改动 + 一个"决定不做"）**

| # | 改动 | 位置 | 为什么（实测根因） | 效果 | 进镜像方式 | 验证 |
|---|---|---|---|---|---|---|
| 5 | capture-safe 融合 GEMM | C++ 3 文件 + W4A16 bridge（预分配 slab，**调用前必须 `zero_()`**） | `new_zeros` 在 graph 捕获期失败；内核用 `atomicAdd` 累加 ⇒ 复用缓冲未清零 | eager 14.6 → graph **22.0**（+50%） | 补丁文件 COPY + 断言 | B3 验收 |
| 8 | M=1 稠密 fp16 GEMM 换 skinny | `vllm/model_executor/layers/utils.py` 的 `rocm_unquantized_gemm_impl`，权威副本 `docker/patches/layers_utils.py`（md5 `55379cbd…`，基线 `43a4addb…`） | gfx906 被上游 `_ON_GFX9` 挡在 `use_skinny` 门外 ⇒ 48 次 `in_proj_ba`（318 µs/次，**耗时与 N 无关、与 mclk 无关** ⇒ Tensile 在 M=1 选了串行 K 循环，不是带宽）+ `lm_head`（6 982 µs）= **22.2 ms/step = 41.4%** | step 53.663 → **33.008 ms**（1425/1000 口径 18.6 → 30.3 tok/s；交付态实测 **32.4**）；`lm_head` 达 93% `BW_eff` | 同一谓词，交付默认不依赖 env（`VLLM_ROCM_USE_SKINNY_GEMM` 未写进镜像 ENV ⇒ 代码默认 True） | G1–G8 八道门 |
| 9 | 两族手写 wave64 GEMV | 新内核 `docker/patches/p2_w4a16_gemv.hip`（md5 `af1535d6…`，`p2_gemv` + `p2_reduce`，splitK=10 单累加器）+ `container_src/wna16_exllama.py` 四段补丁（ctypes shim、`direct_register_custom_op`、加载期 `_p2_part`、白名单路由） | W4A16 GEMV 只跑到 `BW_eff`(931 GB/s) 的 **33%**；P1 实测 V3 原型在 (5120,8192)/(5120,7168) 两族 1.92×/1.95× 距探针尺，而 incumbent 是 2.25×/2.21× ⇒ 只有这两族**现状就能赢** | 同机双臂 **32.3 → 33.3 tok/s（+1.0 = +3.1% = 930 µs/step）**；设备侧 812–826 µs | Path B 里 hipcc **现编** `.hip` ⇒ `.so` 由镜像自己产出 | G-P0…G-P6 七道门 |
| 9b | **M 判断与 env 读取必须在 custom op 的 impl 里** | 同上，`apply_weights` 只留装载期常量 `_p2_route` | `apply_weights` 在 **dynamo 追踪区内**，形状在 trace 那一刻被折叠、False 分支烤进编译产物 ⇒ 首跑 0 命中且与 eager 对照差 64 倍 | 修复后每步 `p2_gemv` **64.0/step/rank**、incumbent 256→192 | — | G-P3 run1 FAIL → run2/3 各 15/15 |
| 4 | `.pth` 注入的 GEMV 修复 | 容器内 `gfx906_gemv_fix.py` + `.pth` | fp32/fp64 `matrix@vector` 在 gfx906 直接段错误 | 崩溃消失 | COPY + 断言 | 全链路冒烟 |
| B4 | custom all-reduce **不做** | `serve.sh` 默认 `--disable-custom-all-reduce` | PCIe P2P 上 RCCL 已持平，无 xGMI 可吃 | 0 收益，少一个风险面 | 默认参数 | 52 轮压测 |

**交付参数级改动（`docker/entrypoint/serve.sh`，116 行，每条默认值的理由都写在镜像内 QUICKSTART §5）**

`--dtype float16`（无原生 BF16 矩阵路径；FP16 实测比 BF16 快约 2.8×）· `--attention-backend TRITON_ATTN`
+ `--mm-encoder-attn-backend TRITON_ATTN`（gfx906 **无 MFMA**，ROCm 原生 attention 不可用）·
`FULL_DECODE_ONLY` + capture sizes `[1,2,4,8,16,24,32,48]`（graph 让 decode +50%；4-bit fused GEMM
只覆盖 M≤50）· `--max-num-seqs 48`（必须同时 ≤ capture 上限与 GDN/Mamba 状态块数）·
`--gpu-memory-utilization 0.90` · `--max-model-len 16384` · `TP` = `renderD*` 计数 · MM 按
`config.json` 是否有 `vision_config` 自动选 · **`VLLM_CACHE_ROOT` 按 W4A16 内核选择隔离**：vLLM 的编译
缓存键含模型/并行/图配置但**不含内核选择**，而两种内核对 4-bit 权重做了不同重排，混用会在 profile
期报 `expected size 640==5120, stride 8192==1024`。可覆盖 env：`TP DTYPE MAX_LEN GPU_MEM_UTIL
MAX_NUM_SEQS PORT HOST SERVED_NAME QUANTIZATION MM EAGER CUDAGRAPH_MODE CAPTURE_SIZES
CUSTOM_ALL_REDUCE MODELS_ROOT` + 内核级 `VLLM_WNA16_P2_GEMV`（≈−3%）/ `VLLM_WNA16_EXLLAMA`（慢约 8×，仅排查）。

### 2.1 环境层逐项：机制与"不改会怎样"

这一层的本质是**把 4 个硬件事实 + 1 个上游缺失 + 1 个上游 bug 固化成可回放配方**：

| 事实 | 后果 | 我们的处置 |
|---|---|---|
| gfx906 **无 MFMA** | ROCm 原生 attention（语言模型侧与**视觉塔侧各一处**）不可用；W4A16 只能走 ExLlama/Triton | `--attention-backend TRITON_ATTN` + **`--mm-encoder-attn-backend TRITON_ATTN`**（漏第二个 ⇒ A3 多模态验收失败）；§2 表第 9 行的手写 wave64 内核 |
| 发行版 ROCm 7.2.4 **没有 gfx906 kernel** | ① TP2 在 RCCL 初始化段错误（`hipErrorInvalidKernelFile`）② 普通 matmul 段错误 | ① 覆盖 **torch 自带的那份** `torch/lib/librccl.so.1.0`（RCCL 2.27.7 gfx906 重编版，11 337 072 B / sha256 `5c5544a3…`，官方库留 `librccl.so.official-backup`）② 补 **156 个 Tensile `hsaco/co/dat` 到 `torch/lib/rocblas/library/`**（反直觉点：**起作用的是 torch 那份，`/opt/rocm` 里那份不需要**） |
| 上游把 gfx906 **有意排除在 `_ON_GFX9` 之外** | 稠密 M=1 GEMM 拿不到 skinny 快路径 | 不放宽 arch 谓词，只开一条 **M=1 例外**（§2 表第 8 行） |
| **rocBLAS sgemv/dgemv 在 gfx906 段错误** | 任何 fp32/fp64 `torch.mv` / `addmv` / 带 1-D 操作数的 `matmul` **直接杀掉整个进程** | `.pth` 注入 `gfx906_gemv_fix`：用 `torch.library` 的 Dispatcher 覆盖 aten CUDA kernel，`mv` 改写成 `(mat @ vec.unsqueeze(1)).squeeze(1)` |
| vLLM **compile-cache key 不含 W4A16 内核选择** | 两种内核对 4-bit 权重做了不同重排并烘成 static input ⇒ 混用缓存在 profile 期死于 `expected size 640==5120, stride 8192==1024` | `serve.sh` 按内核选择分缓存根 + **两条构建路径末尾各有"`/root/.cache` 必须为空"断言** |

**`.pth` 三个必须记的细节**：① 只在 `gcnArchName` 含 `gfx906` 时生效，否则彻底 no-op；②
**`_LIB = torch.library.Library("aten","IMPL")` 必须是模块级全局 —— 它被 GC 会撤回已注册的 impl**；
③ 顶部 `try/except: pass` 保证永不破坏 import，并导出 `APPLIED` 供 selfcheck 回读。
**为什么不用 `sitecustomize.py`**：Ubuntu 自带 `/usr/lib/python3.12/sitecustomize.py`，会在 `sys.path`
上遮蔽我们那份（根目录 `sitecustomize_gfx906.py` 是这条被否路线的残留，但它发明了今天仍有效的旋钮名
`VLLM_GFX906_NO_FIX=1`）。

**Triton attention 的两个经验值**（出处就是 Z100 适配笔记）：`BLOCK=32`（上游给所有 AMD GPU 统一降到
16）与 `VLLM_TRITON_ATTN_NUM_SEGMENTS=64`（笔记原文 `64 segments is optimal (+26% 8K decode), 128 gives
no further gain`）。前者是文件 patch、后者是 ENV；守卫分别是 `grep -q is_gfx906` 与 selfcheck 的 ENV 比对。
改写脚本 `docker/fix_gfx906_block.py` 已退场（全仓无引用），现在由 build-context 携带成品。

### 2.2 W4A16 桥接与 C++ 算子（性能层最底下的一块地基）

- **`ops.gptq_gemm_out`（B3 CUDA graph 的前提）**：schema 把 `c_out` 声明为 `Tensor!`（caller-owned、
  in-place），**整个 op 不做任何分配**；`temp_dq` 允许是空张量，只有大 M 的 reconstruct 路径需要它，
  而那条路**永远不出现在被捕获的图里**。不改则 `ops.gptq_gemm` 走 stable-ABI 的 `aten::empty` ⇒
  捕获期失败 ⇒ 只能 `enforce_eager`，decode 少掉约 50%。四处登记：`csrc/libtorch_stable` 的
  `q_gemm.cu` / `ops_stable.h` / `torch_bindings.cpp` / `_custom_ops.py`（仓库镜像副本在 `container_src/`）。
- **一条容易咬人的语义分叉**：ExLlama 融合内核**用 `atomicAdd` 累加**进 `c` ⇒ 桥接层调用前**必须
  `c_out.zero_()`**（对固定地址的 persistent slab 做 `zero_()` 是 capture-legal 的）；而 **P2 内核是
  OVERWRITE、无 atomicAdd ⇒ 走 P2 分支时禁止 `zero_()`**。两条语义在同一个函数里分叉。
- **桥接层的三行重排 + 三个非显然约定**：`weight_packed/weight_scale/weight_zero_point` 转置成
  `[K//8,N] / [K//G,N] / [K//G,N//8]`；nibble 必须做 `gptq_shuffle` 的 kernel-外等价 interleave
  （ExLlama 从位 `(i//2)*4+(i%2)*16` 读连续 k，不是 GPTQ 的顺序 `4*i`）；**`g_idx` 必须是空张量**
  （内核把它当 act-order **置换表**消费，不是 group index）；zeros **不被 shuffle 打乱**（内核用朴素
  LSB-first 解）；用 **GPTQv2 zero-point 编码**（stored nibble == 真实 zp）⇒ AWQ 的含 0 零点可精确表示，
  绕开 GPTQv1 "+1" 无法表示 `zp==0` 的 checkpoint bug。选它而非 Triton 的理由：M=1 decode 形状上
  **快 8–17×**，而 W4A16 GEMM 占 decode step 的 ~70%。

### 2.3 交付配方：权威副本、两条路径与两道"反向防呆"断言

**这是本项目最容易踩错的地方，必须先讲清楚**：`docker/patches/` 在仓库里**只有 2 个文件**
（`layers_utils.py`、`p2_w4a16_gemv.hip`），而 `rebuild` 阶段 `COPY` 的另外 5 个补丁
（`wna16_exllama.py`、`linear__init__.py`、`triton_decode_attention.py`、`gfx906_gemv_fix.py`、`.pth`）
**在仓库里根本不存在** —— 它们由 `prepare_build_context.sh` 用 `docker cp` 从**运行中的容器反向生成**到
`docker/build-context/patches/`。手写权威源在 `container_src/`。

⇒ **build context 自动跟随容器，仓库 ≠ 容器**。改 python 补丁后进镜像的唯一正确顺序是
**仓库 → `docker cp` 进容器 → `prepare_build_context.sh` → Path B → Path A → 镜像内 md5 复验 →
真机 selfcheck**。为堵"容器里还是旧文件"这个已烧掉一个测量周期的 bug 类，`p05_install.sh`
**拒绝安装交付配方不 COPY 的文件**（而这条守卫自己也曾失效：早期用 sed 做 ERE 转义，`&` 在替换侧展开成
整个匹配 ⇒ 连在配方里的文件都被判 reject，**守卫全盲**）。**唯一例外是 P2 的 `.hip`：从仓库 `cp` 而不是
从容器抽**，脚本注释明写"这一步会覆盖 `$CTX/patches`，**stale build-context 会遮蔽新源** ⇒ 事后必须比 md5"。

**两条路径的分工与 context 差异**：Path B（`--target rebuild`，context = `docker/build-context/`，含
6.8 GB 的 `dist-packages.tar.gz`，解包 ~18 G）与 Path A（`--target final`，context = `docker/`，几秒的
薄分层）。因为 context 不同，**同一个 Dockerfile 对"COPY 路径必须存在"的要求也不同** ⇒ 只跑 Path A 时
`docker/patches/` 有那两个文件就够，这就是"改一个 python 补丁只要几秒"的成因。但 **Path A 只重 COPY
`layers_utils.py` 一个**（其余靠 base 携带），同时 `rm -rf /root/.cache/vllm*`。

**`SNAPSHOT_IMAGE` 这个名字有历史包袱**：语义是"**Path B 的产物** `vllm-gfx906:rebuilt`"，刻意**不能**是
`docker commit` 出来的开发快照，三条实测理由写死在注释里 —— 快照带 ~16 GB `/root/.cache`（pip 9.7 G +
vLLM 编译 3.6 G + **1.8 GB AOT W4A16 缓存，其键不含内核选择 ⇒ 一种模式的缓存会弄坏另一种**），且
`:phase3-working` 早于 B3/B4/C1，**连配方自己的静态断言都过不去**。

**两道"反向防呆"断言值得原文引用**（同类交付里少见）：① `test -z "$(ls -A /root/.cache)"` 放在
**最后一步**（改断言列表时不要作废上面那个 18 G 解包层），防的是"有人拿 `docker commit` 当 base"；
② `nm -D --defined-only … | grep -qw p2_gemv` 的 **`-w` 是载荷性的** —— 没有 `extern "C"` 时 mangled 的
`_Z14p2_gemv_kernel…` 仍**含**这个子串，`grep -q` 会对一个 ctypes 取不到符号的 `.so` 绿灯（容器内验证过）。

**P2 内核为什么在镜像里现编**：dev 容器里有一份为跑门测手编的 `.so`，它会随 `dist-packages.tar.gz`
一起进来，**无法保证源是本仓库这份 `.hip`** ⇒ `RUN rm -rf ${SP}/gfx906 && hipcc --offload-arch=gfx906 -O3 …`。
`.hip` 头部还留了一条"进镜像必须换头文件"的记录：**`#include <hip/hip_fp16.h>`，不是
`<cuda_fp16.h>`**（后者在 ROCm 7.2.4 下 file not found；P1 能过只因 torch 的 `load_inline` 给了 shim `-I`）。
另有两条**纯实测**写进注释的内核事实：单累加器（multi-acc 全面输 1.0–1.5 µs 而寄存器零代价 ⇒ 串行 dot2
链已被同 SIMD 的其他 wave 隐藏）、`n = min(..., N-1)` 的 **clamp 而非发散退出**。

**一处已修的配方级缺陷（当时会静默毁掉交付）**：运行时 ENV 原先**只写在 Path A 的 final 阶段** ⇒
重建出的 `:rebuilt` 里没有 `HSA_OVERRIDE_GFX_VERSION` / `VLLM_WNA16_EXLLAMA`，表现为**静默退回 Triton
W4A16（3.9 tok/s）**。现在两条路径共用同一份声明，`serve.sh` 里还留了一层 `:-` 兜底。

**版本三件套（交付态）**：vllm `0.28.0+rocm724` / torch `2.13.0+rocm7.2`（hip `7.2.53211`）/
triton `3.8.0` / torchvision `0.28.0+rocm7.2` / transformers `5.16.1`；镜像内 `libp2gemv.so` 26 536 B，
`nm -D` ⇒ `T p2_gemv`。

### 2.4 P2 分派的九条集成约束（内核层与交付层的交界，每条都是踩出来的）

1. **形状白名单是 post-partition 的**（TP2 分片后：`8704 = 17408/2`、`3072 = 6144/2`），源码注释原文
   **"A wrong assumption here fails SILENTLY —— zero hits, zero error, zero gain"** ⇒ 必须拿引擎形状
   日志对撞，不能靠推导。
2. **`M == 1` 判断绝不能写在 `apply_weights` 里**：它在 dynamo 追踪区内，会按**首次被 trace 的 prefill
   形状**求值并把 False 分支烘进产物 ⇒ 每个被捕获的 decode 图重放 incumbent（实测 compiled 下 0 次 vs
   eager 下 64 次/step/rank）。解法＝塞进 custom op 的 **impl**（dynamo 不进 op 体）—— 与 §2 表第 8 行
   的 vLLM 自家 skinny 分派**同一手法**。
3. **op 返回 `None` 并把 `out` 声明进 `mutates_args`**：返回"同时也是输入"的张量会触发
   `output must not also be an input … may not alias any inputs`（且将被改成 error）。
4. **热路径只允许纯 python 的 int/list 操作**：任何返回非 Tensor 的 torch 调用（例如
   `is_current_stream_capturing`）会以 `torch.* op returned non-Tensor` **杀掉引擎初始化**。
5. **`CDLL` 是懒绑定**：装载成功不代表符号存在 ⇒ 必须显式 `getattr(_P2_LIB, "p2_gemv", None)` 探一次并
   缓存，否则 `AttributeError` 会推迟到**引擎内、捕获期**第一次调用才炸；`except (OSError,
   AttributeError) ⇒ _P2_OK=False` ⇒ **默认安全回退**到 `gptq_gemm_out`。
6. **split-K 单一真源**：`_P2_SPLITK` 同时决定 `part` 行数与传参（"比 part 行数大的 splitk = 内核无法
   检测的越界写，它只看到裸指针"）。10 是 Task 0 实测最优，而 `.hip` 的整除约束 `splitk | K/GROUP(=40)`
   正是 P1/P2 那个"2 的幂"网格**完全错过最优点**的原因。
7. **`GROUP` 是内核唯一查不了的布局事实**：`p2_gemv` 校验 `K%128 / N%64 / words%splitk`，然后按
   `[K/128, N]` 读 scales、`[K/8, N]` 读 qweight ⇒ 一个 `group_size == -1` 的 per-channel 层若落在白名单
   形状上只有 1 行 scales，warm-up 会读到末尾之后 39 行**且无人报错**。因此**同一个谓词**还要 gate
   `size_bits==4 / not has_g_idx / act_type==float16`（注释直言：拆成两条谓词本身就是 correctness
   hazard —— 分配了不路由浪费内存、路由了没分配崩在 `None` 上、group 错配是**静默**错误）。
8. **预分配 + 捕获外 warm-up**：无条件分配会给**每一个** WNA16 层挂一份 `(splitk, N)` fp32
   ≈ **91 MB/卡** ⇒ 只在白名单层分配；并用一次 `torch.cuda.synchronize()` 强制 HIP code object 在捕获
   **之外**加载（"so a first-call lazy load cannot become a capture failure"）。
9. **kill-switch 在 impl 内读 `VLLM_WNA16_P2_GEMV`** ⇒ 交付镜像内实测生效（纯净容器 off 臂 launch = 0）。

---

## 3. 测试数据

### 3.1 #20-P0 归因：为什么值得动手（1425/1000 钉频、TP2、batch1、64 个干净 decode step）

`BW_eff` 三工作集实测 865.9 / 920.0 / **931.2 GB/s**（4 GiB 已进平台期 ⇒ 不是 L2 假象；= 官方峰值
1024 的 91%）。**计算口径 GPU busy = 0.89 ≥ 0.8** ⇒ 按 spec 路由到"内核问题"这一行，不是调度/主机问题。

| bucket | µs/step | 占比 | launches/step |
|---|---|---|---|
| `rccl_wait` | 25 557.2 | 33.4% of window | 8 320（peer-wait，profiler 放大，交付 step 里只 ≤5.9 ms） |
| **`dense_fp16`** | **22 242.2** | **41.4%** | 3 136（49/step）→ **P0.5 打的就是它** |
| **`w4a16_gemv`** | **19 991.7** | **37.2%** | 16 384（256/step，5 个形状） |
| `elementwise` | 4 086.5 | 7.6% | 43 456 |
| `gdn`(triton) / `attention` / `memcpy` / UNCLASSIFIED | 798.2 / 537.7 / 128.8 / 7.5 | 1.5/1.0/0.2/0.0% | 6 144 / 3 072 / 1 600 / 64 |

W4A16 逐形状（256 次调用按计数加权 = **`BW_eff` 的 33%**，最坏 20%、最好 58%；完美地板 6 788 µs
⇒ **头寸 13.8 ms/step**）；graph 内 **per-node 边际 2.209 µs、replay 固定开销 16.8 µs** ⇒ Split-K
归约可用但要优先 persistent 形式。`dense_fp16` 拆开：`in_proj_ba` 15 260 µs（48/step）+
`lm_head` 6 982 µs（1/step）——**更大且不需要写 CUDA 的头寸** ⇒ 决策是"开工 #20，但前面插入 P0.5"。

### 3.2 #20-P0.5 八道门（全过，已进镜像）

| 门 | 判据 | 结果 |
|---|---|---|
| G1 | LLMM1 可跑、确定性 stable、rel < 3×Tensile | ✅ 两形状 |
| G2 | `in_proj_ba` ≥4×、`lm_head` ≥70% `BW_eff` | ✅ **67×** / **93.0%** |
| G3 | 真权重 rel ≤ 3×Tensile 且 < 5e-3 | ✅ 1.93× / 1.35×（argmax 翻转 0/64，margin 大一个量级） |
| G4 | 门谓词精确（M>1 / bias / 非连续 / k>8192 / m%4 / 非 gfx906 / env=0 全不接管） | ✅ 10/10 |
| G5 | `ab_off` 复现 53.7±3 ms 且 `ab_on` ≤ 36.0 ms | ✅ 53.663 / **33.008** |
| G6 | 贪心前 32 token 一致且无语义跑偏 | ✅ **256/256 逐字一致**（两臂 md5 同 `7db6f866…`） |
| G7 | 桶表 UNCLASSIFIED < 5% 且 `w4a16_gemv` 不变 | ✅ 0.0% / +0.2%（残差 326 µs = 1.6%） |
| G8 | 两条构建路径静态断言全过 + selfcheck 无 FAIL | ✅ 并在**交付镜像内**跑门 10/10 |

裸测：`in_proj_ba` 4.6 µs vs Tensile 309.4（67×）；`lm_head` 1 468.3 vs 6 637.4（4.5×，865.9 GB/s
= 93.0% `BW_eff` ⇒ **这一项已无肉可割**）。归因表 `dense_fp16 22 207.2 → dense_gemv 1 707.5 µs`。

### 3.3 #20-P1：三原型判定 + 标尺重标定（1700/1000 锁频、graph 内、171 行有效数据）

**结论：非 kill；窄 P2（只做两族）成立。** 主表（"尺"= 每形状三种 load-only 探针的最优值，v3 定尺）：

| 形状 | calls | 字节地板 µs | **探针最优 µs（尺）** | incumbent-rot µs | V3 最优 µs | inc/尺 | 胜者 |
|---|---|---|---|---|---|---|---|
| 5120×17408 | 64 | 56.90 | **63.5**（x4, 702 GB/s） | 90.93 | 115.33 | 1.43 | inc |
| 8704×5120 | 64 | 28.45 | **38.2**（mlp, 584） | 116.19 | 167.96 | 3.04 | inc |
| **5120×8192** | 48 | 26.78 | **35.1**（mlp, 597） | 78.99 | **67.34** | 2.25 | **原型** |
| **5120×7168** | 16 | 23.43 | **33.5**（mlp, 548） | 74.14 | **65.17** | 2.21 | **原型** |
| 3072×5120 | 64 | 10.04 | **20.3**（mlp, 387） | 41.50 | 44.38 | 2.04 | inc（微差） |

收益换算（分母统一为交付态实测 32.4 tok/s ⇒ step 30.86 ms，精确式 `32.4/(1−µs/30860)`）：
分派现状 **702.7 µs = +0.76**；调到 1.3× **1 148.7 µs = +1.25**；探针上限 **2 759.7 µs = +3.18**（未必可达）。

**这一轮被实测推翻的四个设计假设**（都是硬价值）：① "字节地板"不成立（load-only 只到 33–74%，
且标量探针自身 MLP 不足 ⇒ 尺换成 mlp/x4，比标量高 18–59%）；② V1 的 7e-7 噪声声明 —— 实测
**3.3 fp16 ULP，超 0.5 ULP 门 6.6 倍 ⇒ V1 判死，主因是噪声（逐字去偏 ≈20 条/字），不是 6% 的
VALU 超支**；③ "Split-K 重复读元数据"（实为 1 份）；④ LUT 退化预测。**op/weight 指标退役**：它标定在
已被证伪的"VALU 天花板 ≈ 字节地板"上。**L2 判据重标定（最重要新事实）**：对 fp64 零自由度参考，
**V3 RMS 0.024–0.027 ULP（MAX 1）过 0.5 门 18× 余量，而 incumbent 自己 RMS 69–219 ULP（MAX 14 828）
⇒ 0.5 门按原文连产线内核都要拒，门必须写成"≤ 产线基线的若干分之一"**。

### 3.4 #20-P2 七道门（六门过，G-P6 字面 FAIL 0.1）

| 门 | 判据 | 结果 |
|---|---|---|
| G-P0 | 两族 graph 内 ≤60.8 / ≤57.0 µs（1700/1000） | **逐族 FAIL ⇒ 用户改口径放行**：8192 = **60.51 PASS**、7168 = **58.21 FAIL**（1.274×）；累计 **1 142.1 µs = 承诺的 99.4%**。`multi-acc` 是回归（+1.0~1.5 µs，已回退）；真最优 **splitK=10 在计划网格之外**（sk 须整除 K/40 ⇒ 原网格只剩 {4,8}，照抄会误判 1.15×） |
| G-P1 | 生产 `.so` 四臂独立门 | **PASS**：wiring 1.22e-04、ULP 0.0292/0.0205、与原型 `torch.equal` 逐位一致；资源 **VGPR 19 / 10 waves/SIMD（gfx906 上限）/ 0 spill / LDS 256 B**；顺手修掉一个 **4% 真回归**（`#pragma unroll 8` 压在运行期边界循环上）；**新事实：7168 graphed 计时 ±0.7 µs 机器态双稳**（非代码、非时钟、非热） |
| G-P2 | 路由矩阵 + shape CSV + 分派 ≈64/step + capture 冒烟 | **PASS 21/21**（7 行 21 断言，含计划原文没有的"探针行"——**没有它，`.so` 静默缺失时所有负向断言空过**）。门的"牙齿"由四臂变异证实：丢白名单 ⇒ 8 行红；`scales↔qzeros` 传反 ⇒ **硬 GPU fault**；删 `mutates_args="out"` ⇒ **门仍绿 ⇒ 证伪计划断言** |
| G-P3 | 引擎 A/B：on ≤31.9±0.3 ms 且 ≥30.6 tok/s，分歧可归因 | **run1 FAIL ⇒ 修复（M/env 进 op impl）⇒ run2/3 各 15/15**：off 32.776 / on 32.120 ms、31.13 tok/s、`p2_gemv` 64.0/step/rank、incumbent 256→192、greedy 256/256 逐位一致。**可信读数 = 设备侧 812–826 µs/step**（wall 自带 ±0.2 ms 噪声 ⇒ 判回归看设备侧账） |
| G-P4 | L2（RMS ≤0.05 且 ≤ incumbent 1/10）+ L3（margin 判据） | **双 PASS**。L2 8/8：P2 **RMS 0.0118–0.0331 / MAX 1 / 逐位可重现** vs incumbent **RMS 29.7–94.1 / MAX 6 592 / 自身 71–74% 位置变动** ⇒ INC/P2 ≥ **1301×**。L3 四条件全绿（`M0_swap 0.0469 ≤ M0_floor 0.0469`，**等号过门 = 落在带边缘不是余量充裕**）；跨臂翻转 2/1015 = 0.20% |
| G-P5 | 两路径镜像 selfcheck + 纯净容器 A/B | **PASS**：全新容器 **23 ok / 0 FAIL**、五道 fence 逐字对齐、`verify_p2_blocks.py` 镜像内 `PLAN-BLOCKS-OK`；纯净容器 A/B 15/15，**off 臂 `p2_gemv` launch = 0** ⇒ **kill-switch 在交付镜像里真的回退** |
| G-P6 | 交付态 serve（auto 时钟）batch1 ≥ **33.4 tok/s** | **字面 FAIL 0.1，A/B 成立**（见下表） |

**G-P6 双臂**（同一 `:latest`、同一全新容器、两卡 `level=auto` 硬守卫、开箱 `gfx906-serve`、与
`p05_serve_probe.sh` 逐字同探针）：

| 臂 | REQ2_WARM(64 tok) | DECODE256（3 次） | 备注 |
|---|---|---|---|
| **on（默认，无 env）** | **33.3** | 33.5 / 33.4 / 33.5 ⇒ mean 33.5 | 冷卡 43 °C 下取；就绪 690 s、`torch.compile` 387.51 s、KV 67 356 |
| off（`VLLM_WNA16_P2_GEMV=0`） | **32.3** | 32.5 / 32.4 / 32.3 ⇒ mean 32.4 | 掉电后由同一可重入脚本续跑；复用同容器 AOT 缓存 ⇒ 就绪 430 s、compile 63.68 s |
| 参照：P2 前同探针 | 32.4 | 32.5 / 32.6 / 32.6 | ⇒ 旧基线**同机可复现**（差 0.1） |

⇒ **delta +1.0 tok/s = +3.1% = 930 µs/step（30.960 − 30.030 ms）**，比设备侧 812–826 µs **还多 13%**
⇒ **serve 侧无隐藏折损**，"1425 钉频收益平移到 auto 档"由假设升为实测。但门 ≥33.4 需要 924 µs
（分母 32.4）～1 020 µs（分母 32.3），而**该收益的算术顶是 33.18–33.28** ⇒ **33.3 踩在顶之上而门仍
过不去；过不了的是门的定法，不是实现**。用户裁定门**不放宽** ⇒ 按字面 FAIL 入档，QUICKSTART §6 里
**引用的交付基线仍是 32.4**，33.3 只记作"含 P2 的实测值"。兑现率：tok/s 口径 80%、µs 口径 71–72%。
唯一不对称处方向**偏保守**（热态吃亏的是 off 臂：它紧随 junction 106 °C 的 graph capture，余温 65–70 °C）。

### 3.5 为什么设备侧只兑现 71–72%（本层最重要的机制发现）

**差额全在分母，不在内核。** P2 单次比承诺还快（实测 `p2_gemv` 54.3 + `p2_reduce` 5.1 = **59.4 µs**
≤ G-P0 门 60.5/58.2），但**被换掉的 incumbent 在引擎内只值 72.0 µs/调用**，而门用的是微基准分母
78.99/74.14 µs —— 权重长期驻留 L2 后分母变小，同一份内核能换回的时间就少了 22%。
⇒ **想把 1 148.7 µs 真兑现，得去动"被替换项的定价"（L2 驻留），而不是 P2 本身**。这条直接决定了
§5 的排序。

### 3.6 稳定性与热（会污染任何数字，必须一起报）

- 宿主有**静默掉电史**：`journalctl -b -1` 尾部无 shutdown 记录、`user.journal corrupted or
  uncleanly shut down`、**所有**容器（含无关开发容器）同一秒 `Exited (255)`。
- Task 6 窗口内死两次；黑匣子（5 s sysfs 采样）取到 run1 **junction 107 °C > `temp2_crit` 105 °C /
  每卡 218 W**、run2 **106 °C / 265 W（未跳闸）**。`power1_cap` 已等于 `power1_cap_max` = 190 W
  ⇒ **宿主侧没有"再压低一点"的旋钮**。缓解：交付态计时只在冷卡窗口取、长作业 `docker exec -d` + 日志、
  脚本做可重入（完成臂跳过、容器复用不 `rm -f` 以保住 542 MB 缓存、失败原地重试）——已被 08:20 的
  一次成功续跑验证。
- 热保护会直接改性能读数：连续压测把结温顶过 100 °C 时同一配置测到 15.9 而不是 22.0 tok/s。
- **钉频必须同时钉 sclk 和 mclk**：`manual` 下不写 `pp_dpm_mclk` 会停在 level 0 = **350 MHz**
  （理论 358 GB/s），且 sclk 写入看起来完全正常、无报错。

### 3.7 交付与分发实测（#18 的第一份数据）

| 量 | 实测 |
|---|---|
| `docker save` 原始流 | 22 070 416 384 B |
| 归档（`save \| pigz -p 6`） | 21 948 506 214 B ⇒ **省 0.6%（1.01×）**，261 s |
| 归档自包含性 | 32 个成员、成员字节和 = 原始流的 **100.0%**（三个 ~7.3 GB blob 占 99.6%）；全量回读 21 s |
| `docker load` 回读 | image ID 一致 |
| **机制** | 宿主是 **containerd 镜像存储** ⇒ `inspect .Size` = 压缩态之和 = save 流字节；`docker images` = 解包态（69.8 GB）。**两者差 3× 是正常现象** |
| **规划口径** | **传输按 ~22 GB，接收端磁盘按 ~70 GB；"再压一遍"是白工** |

---

## 4. 已知限制与未闭合（诚实账）

**功能/性能**
1. **G-P6 字面未达标（差 0.1）**，且门本身高于实测算术上限（诚实账第 9 条）⇒ 对外引用基线仍是 32.4。
2. **7168 一族逐字面未达 1.3×（1.274×）**，靠"累计收益 ≥ 99%"的用户放行口径进来；只上 8192 一族是 +0.959。
3. **收益上限被分母锁死**：incumbent 在引擎内比微基准便宜 ⇒ W4A16 可见收益停在 ~0.8 ms/step。
4. **W4A16 仍是全场唯一大头**：P0.5 后占 step 60.2% / 占纯计算 73.2%，P2 后仅小幅改善。
5. **#20-P0 桶表在交付时钟（auto）下重跑仍未做** ⇒ 所有百分比置信度受此限制（收益的 1425→auto
   平移已在 P2 处被 A/B 实测支持，但桶表本身没重取）。
6. 其余三族（17408/8704/3072）**判给 incumbent**：17408 上 incumbent 距尺仅 1.43×，8704 族 5.3
   waves/CU 的 occupancy 短板是可见输因 ⇒ 宽 P2 前提已被削弱，不是"白捡"。
7. **M>1 未覆盖**：本内核族只做 M=1；capture sizes 里 2/4/8/…/48 全部留在 ExLlama。

**测量与可复现**
8. KV 容量**冷臂比 warm 高 3.6%**（67 356 vs 64 929，warm 两次同值 ⇒ 非运行间噪声，**未定位**）；
   容量规划按低的算。warm 臂 `torch.compile took` 从 C2 的 31 s 变 67 s，**原因未查**（端到端未受影响）。
9. 7168 graphed 计时 **±0.7 µs 双稳** ⇒ 任何 <1 µs 的单次差异在这两张卡上不构成结论。
10. L2 的 margin 判据是**等号过门**（量化在 1/64 与 3/64 格点上）⇒ 带边缘，不是余量充裕。
11. 交付态两臂热态不严格相同（方向偏保守）；未做 8 并发的 P2 A/B。
12. **P2 计划最大缺陷不在内核而在分派位置**（M 判断写进追踪区），**计划原文没写、评审没抓到、实测抓出来的**
    ⇒ 任何"运行期形状/开关判断"都必须落在 op impl 里，这条已进记忆为通用坑。

**工程/交付**
13. **58 项改动未 commit、未 push**（8 modified + 50 untracked）。**必须一起进仓库的交付件只有 6 个**：
    `docker/patches/p2_w4a16_gemv.hip`（**Path B 编 `.so` 的唯一源，目前未跟踪**）+ 已跟踪的
    `Dockerfile.rocm_gfx906`(+33) / `prepare_build_context.sh`(+6) / `entrypoint/selfcheck.sh`(+3) /
    `entrypoint/QUICKSTART.md`(+19) / `container_src/wna16_exllama.py`(+169)。其余 28 个 `p2_*.py|sh` 与
    `logs/p2/` 是 dev/测量工具，`.gitattributes`（`* text=auto eol=lf`）属"仓库级但不进镜像"——
    **没有它，Windows 检出注入 CRLF 后宿主用 ssh 跑这些脚本会全线 `bash\r: command not found`**。
    **一处工作树模糊地带**：`logs/p2/*.json` 与 `logs/p2/l3/` 既未被跟踪也未被忽略，提交前要先定。
    仓库根另有 10 个 8-31 时代的诊断笔记（`CRITICAL_FIX_BASED_ON_Z100.md` / `ROOT_CAUSE_ANALYSIS.md` /
    `NEXT_STEPS.md` 等）未跟踪，多数已被 `BUILD_GUIDE.md` 吸收 ⇒ 归档或删除，别当交付物。
13b. **git 里没有相位分辨率**：全仓只有 4 个 commit，且 `3009064` 把 8-30 → 9-07 的"从跑通到可交付镜像"
    整段压成一次快照 ⇒ "哪个文件何时落地"只能靠任务清单 + 文件时间戳重建（本报告 §1 的相位划分即此来源）。
13c. **`docker/BUILD_GUIDE.md` 的"性能参考"表已落后于现状**（还写着"当前 14.6 / 目标 ~33 tok/s"）⇒
    不要引用那张表；交付态数字以 `QUICKSTART.md` §6 为准。同类文档漂移还有一处：`docker-environment`
    记忆里"从源码编 PyTorch/Triton"那半已废弃，活配方只有两路径。
14. **#18 分发未落地**（只有第一份体积实测）；#24 纯净容器开箱最终验收 blocked by #18。
15. CUDA graph + 视频的复验在收尾（`serve.sh` 默认已 `video:1`，图模式下异常先 `EAGER=1` 对照）。
16. MTP 投机解码**已划掉**：TP2 实测显著负收益（我们和 Z100 作者各测一次，方向一致）。
17. **未做模型广度**：全部结论建立在 Qwen3.8-27B-AWQ + 一个 0.6B TP1 冒烟上；bf16 直发模型需重跑数值门。

---

## 5. 未来计划与可能方向

> **本节已被 `2026-09-10-gfx906-research-verification-and-replan.md` 的 §3 取代**（那份是对外部调研
> digest + David-Lzy fork + 上游 PR 的逐条核实）。留在这里的理由：它是"#20 收口时我们自己的看法"，
> 而修订能显示哪些判断被实测改写了。**下列凡与那份冲突处，以那份为准**；本节内已就地标注三处被改判的项
> （§5.2 的同口径对照已做掉、§5.3 的 #21 已关单、§5.4 的 M-RoPE 转为不做；§5.1 第 3 条**仍是未做**）。

**排序原则**：先做"已实测的机制指向的、不需要新内核的"，再做"需要新测量确认头寸的"，
最后做"作者有、我们未必有的结构性工程"。每条标出**依据强度**。

### 5.1 立刻可做、零技术风险（半天级）

1. **提交并推送当前 58 项**（含 `docker/patches/p2_w4a16_gemv.hip` 与 `logs/p2/`）——这是回滚点，
   也是"改动可追溯"的交付标准；**`.hip` 目前是未跟踪状态，而它是 Path B 编 `.so` 的唯一源 ⇒ 漏提交等于
   配方不可回放**。需你明确授权才做；仓库/推送路线见记忆 `github-repo-and-push-route`，
   提交须行内 `git -c` 身份。
2. **落地 #18 分发**：口径已定档（传 ~22 GB / 对端磁盘 ~70 GB，别再压）。可选 registry 推送 vs
   `docker save` 归档，取决于对方网络。**一条节奏约束**：`QUICKSTART.md` / `selfcheck.sh` 是随镜像分发的
   （`COPY entrypoint/`）⇒ **改一次 entrypoint = 一整轮 B→A + 一次 22 GB 重导**（P0.5 当天就因此滚了三次
   镜像 ID）⇒ 分发前把待改的文档攒够一批再走。
3. **交付时钟（auto）下重跑 P0 桶表**（遗留第 5 条）—— **仍未做，且现在比原来更值得做**。
   `logs/p0/p05_buckets_ab_on.log` 首行写得很清楚 `pin=1425.0MHz` ⇒ 它的 µs/step 与 `%wall`
   全部是**钉频那把尺**（33.01 ms = 30.3 tok/s），**不是交付步**（auto serve ≈ 30.0 ms = 33.4 tok/s）。
   核实文档 §6 这次实测只把**步总数**换到了 auto 口径，**内核级切分一次都没在 auto 档取过**
   ⇒ 那张表的每一个百分比都要按"偏保守的分母"读（分母大 ~10%）。
   而 §6.1 新暴露的"attention 在 16K 占 15.9%（短 ctx 只有 1.6%）"**恰恰是一张 auto 档 16K 桶表**才能定死的事
   ⇒ 这一项与核实文档 §3 序 3（#63）**是同一次测量**，合并做。

### 5.2 大头 A：Prefill 的结构重写（依据强度：作者的实测很强，但**口径未对齐**）

Z100 作者在大 M 上把量化 GEMM 换成"**先去量化 → 稠密 GEMM**"：QKV/Gate/Down 在 M=8192
**3.22–3.58×**，整模型 2K **+156.8%**、8K **+96.5%**。我们的 prefill 只有
**8K 346.6 / 16K 235.0 tok/s（eager 口径）**，作者生产表是 8K **468**——但那是
**HTTP + graph + FP8 KV** 口径，**不可直接比**。

**那个"同口径对照"已经做掉了**（核实文档 §6.3，交付镜像 + 全新容器 + 开箱 `gfx906-serve` + graph + auto 时钟）：
prefill warm = **2K 434.8 / 8K 338.1 / 16K 249.9 tok/s**。与 eager 口径的 346.6 相比只差 2.5%
⇒ **graph 对 prefill 几乎没有贡献**（合理：prefill 不是 launch-bound），当初担心的"口径未对齐"不是大头。
对上作者的 8K **468** ⇒ 差距 **1.38×**；decode 侧对应的差距是核实文档 §5.5 的 **1.34×**。

⇒ **一个值得追但尚未成立的观察**：prefill 这边量到 1.38×，decode 那边（核实文档 §5.5，fork 的
phase-115 `44.74` vs 我们 `33.3`）是 1.34× —— 两个几乎同一个数。**但这两个数不可相减当作证据**：
它们来自**两个不同的外部来源**（468 出自 Z100 文章语料，44.74 出自 GitHub fork 的 phase 文档）、
**两个不同的模型版本**（3.6 vs 3.8）、且 fork 语料**没有任何 prefill 数**（核实文档 §5.4）。
所以"一个共同因子（档位/栈版本/板型）"目前是**假设**，数值巧合是它的全部证据强度。
⇒ **能真正检验它的只有一件事，而且很便宜**：核实文档 §3 序 5 —— 在**我们这台机**上按 fork 的
C1 协议（固定 128 token + C8 并发）复测 decode。测出来若 ≈33 tok/s ⇒ fork 那 44.74 与我们的差是
**口径/硬件**造成的，prefill 的 1.38× 同理，本节头寸保留原估；若我们这台机按他们协议也只跑出
他们数字的 0.7× ⇒ 差距真在内核里，才轮到本节与 §5.5 花重编去 A/B。**顺序上先测再立项，别反过来。**
风险：大 M 反量化会引入一次额外的权重展开带宽/显存往返，gfx906 无 MFMA ⇒ 稠密 GEMM 走的是
Tensile/rocBLAS 的 fp16 路径（**M=1 的串行 K 病理在大 M 上不存在，但 P0.5 的教训是"先看路由谓词"**）。

### 5.3 ~~大头 B：#21 手写 HIP Attention~~ → 已判决关单；剩下的是一句"换后端"冒烟

**当初写的"立项前要先补一个长 ctx decode 归因，否则重演 P1 的分母错用"已经做掉了**
（核实文档 §6：交付镜像 + 全新容器 + 开箱 serve，2K **33.36** / 8K **31.95** / 16K **28.56** tok/s）。
结果两头都清楚了：

- **没有塌方**。digest/fork 那个"gfx906 不修 split-KV 就掉到 0.24 tok/s"的恐慌对我们不成立
  —— 我们本来就在 64 段 split-3D 路径上（§1.11(a)，源码级已证）。
- **1.6% 那个头寸是短 ctx 的数**（P0 桶表：交付口径 513.6 µs / 33.0 ms = 1.6%；本节上面引的 537.7 µs
  是 1425 钉频口径，同一桶的两个标尺读数、别混用），16K 实测是 **15.9% 步时 /
  边际带宽利用率 ≈9.0% peak**，压到 HBM 地板的上限是 28.56 → **33.35 tok/s（+16.8%）**。
  ⇒ 本节原来那句"attention 不是大头"**只在交互档成立，长 ctx 档要乘 10**。
- **仍判定不手写**（2–3 天、0.28 上我们还没有一次成功的自写 attention 内核），因为同样的头寸
  有"改一个 CLI 开关"的通道：v0.28 的 `ROCM_ATTN` 已是**纯 Triton、支持 head 256**（§1.11(b)，
  源码已证、运行未证）。⇒ 幸存项是 **#63**：一次 16K 内核分桶 trace（确认那 5.04 ms 真在 attention 里）
  + 一次 `--attention-backend ROCM_ATTN` 冒烟。**这一项还会顺带检验"gfx906 必须 TRITON_ATTN"这条交付默认。**
- 作者那两条踩坑（**KV packed-view stride** ⇒ 必须用真实非连续视图 + changing-input 图回放；
  缺 `__syncthreads()` 的 NaN 竞态）**照抄进 #63 的验收**，无论最后是换后端还是写内核都用得上。

**#23 FP8 KV（与本节解耦了）**：冒烟已过，受控对照实测**容量 +58.8%，不是翻倍**
（fp16 67,356 tokens @ block 784 vs fp8 106,977 @ 1568，同容器同 `Available KV cache memory: 2.66 GiB`）。
少掉的那一半是 **hybrid 页对齐**吃掉的：GDN 页与 attention 页共同决定 `block_size`，fp8 只压 attention 半边。
⇒ 它的定位从"与 #21 强耦合"改成**独立的容量杠杆**（服务核实文档 §6.3 那组 TTFT：16K 首 token 58.7–64.9 s，
并发/长 ctx 的可交付性瓶颈在容量与 TTFT，不在 attention 延迟）。**质量与速度都还没定价**，
且外部语料里也没有 fp8 KV 的数值质量数据 ⇒ 若要把 fp8 变成交付默认，必须先补质量档。

### 5.4 中头寸：GDN 融合与 #22 M-RoPE（作者数据）

GDN：`scatter_add` → 直接写（**+16%**）、变长因果卷积融合（**168.5 → 6.8 ms** 单项、8K 模型级 +14.9%）。
我们的 GDN(triton) 桶在 1425 口径只 798.2 µs/step = 1.5%（交付口径 847.3 µs = **2.6%**），**看起来不值**
——但作者是在 prefill/GDN 占比更高的形状下测的 ⇒ 与 §5.2 一起在同口径归因表里重新定价。
⇒ **GDN 这边真正开着的不是融合，是稳定性**：核实文档 §1.3 判定 readout 落进 fp16 缓冲的溢出风险
在我们配置下**条件为真**（状态存 fp32、readout cast 到 fp16），对应行动项是 **#62 幅值探针**
（必须覆盖 prefill/chunk），不是把内核合并。顺带掉出一个**容量杠杆**：
`--mamba-ssm-cache-dtype fp16` 可回收 **1.81 GB/卡**进 KV 池，但它会把溢出提前到状态存盘那一步
⇒ 两个必须一起读，不能单独采纳容量收益。

**#22 M-RoPE：改判为不做**（核实文档 §3 关单表）。作者 +5.5% 那个数是 fork 里
`VLLM_ROCM_ENABLE_GFX906_QWEN36_FUSED_QK_ROPE_GATE` 那条路，**默认关、且只为 Qwen3.6 那套形状服务**，
连它配套的 Triton `scf.if` fat-pointer 补丁也是为该融合而生（§5.7）。我们没有任何证据说
Qwen3.8 的 M-RoPE 有缺口——它在 P0 表里落在已归因的 `elementwise` 12.4% 内（§1.9b），
而那一桶现在的靶子是"减 kernel 数量 + P2 侧省掉 `zero_()`"，不是 RoPE。

### 5.5 若继续 W4A16（#20 剩余）

优先级明确低于 §5.2/§5.3，但有三条**已知、便宜**的路径：
1. **改"被替换项的定价"**（L2 驻留，§3.5）——这是唯一能把已承诺的 1 148.7 µs 真兑现的杠杆。
2. **`raw[UNROLL]` 多条累加器已被实测证伪为回归**，不要再试；V3 剩余 1.6–1.7× 距尺的差距要靠
   occupancy/访问模式（8704 族 5.3 waves/CU 那类问题在两族上没有对应证据）。
3. **persistent/last-block reduce** 省掉 64 次/step 的 `p2_reduce` 节点（per-node 2.209 µs ⇒ 上限 ~0.14 ms/step，
   与 930 µs 的当前收益同量级，值得算一次账再决定）。

### 5.6 明确不做

**MTP 投机解码**（两轮实测负收益）· **custom all-reduce**（PCIe P2P 上 0 收益 + 风险面）·
**BF16/FP8 权重**（缺矩阵指令 —— 注意与 §5.3 的 **FP8 KV** 不是一回事：KV 存取不需要矩阵指令，
冒烟已过并实测容量 +58.8%）· **再从源码编 PyTorch/Triton**（已废弃路线）·
**#21 手写 HIP attention**（核实文档 §6.2：16K 实测无塌方 ⇒ 恐慌前提不成立；同头寸有"改一个 CLI 开关"
的通道 #63，2–3 天手写属重复造轮子）· **KVarN**（digest 的"0.7×"是算术错，110/200 = **0.55×**，
且 fork 40 篇文档零命中无旁证）· **`q_gemm.cu` gfx906 重调**（⏸ **条件挂起**而非永久不做：
它在 `csrc/` 且 `CMakeLists.txt:1252` 要求单 arch 重编 ⇒ 只有核实文档 §3 序 5 的同机复测
证明那 1.34× 真在 stock 算子里，才重启这一项）。

### 5.7 与 Z100 知识库的差集（一句话版）

作者 24 k 行 / ~80 文件的改动里，**我们已对齐**：graph、wave64 W4A16 内核、Tensile/RCCL 补充、
FP16-only 决策、MTP 否定、TRITON_ATTN 替代路径。**我们未做**：大 M prefill 结构重写、
M-RoPE 融合、GDN 融合、启动时间工程（Startup Plan 指纹缓存 + page-cache 预取，
作者权重加载 47.12 s）。**状态已变的三项**：FP8 KV 冒烟已过并实测**容量 +58.8%**（不是 §5.3 原写的"翻倍"，
质量未定价）；手写 HIP attention **已判决关单**（长 ctx 实测无塌方，幸存路线是换后端 #63）；
GDN 的开放项从"融合"改成了"readout 溢出探针 #62"。**我们比作者多做的**：完整两路径镜像配方 + 开箱自检 +
端到端多模态验收 + 把每个数字钉在门和标尺上（作者报告里没有"尺"这一层）。
**关键约束（要分清两类，别一刀切）**：他们的 vLLM 基线 commit 是 `399f97424e61`（0.27.2rc1.dev26），
**与我们的 0.28.0 不同** ⇒ **`csrc/` 里的 C++/HIP 补丁不可搬**（`LLMMB4`、`q_gemm` 重调、
`CMakeLists.txt:1252` 的单 arch 重编要求，见核实文档 §2）；
但**纯 python / 纯 Triton / 上游已合并的 PR 是可以直接取的**——#54146 连它那 50 行回归测试一起就是本轮
唯一"现成轮子 + 现成验收门"（核实文档 §5.2、§5.6）。**判据是"这个物件的载体在哪一层"，不是"版本对不对"。**
他们引用的论文/文档中真正影响我们决策的只有两类（gfx906 无 MFMA 的能力边界、coherence/all-reduce
相关文档用于佐证 B4 的"不做"判断），其余是背景。

---

## 6. 复现与证据索引

| 层 | 位置 |
|---|---|
| 归因（P0） | `docs/superpowers/plans/2026-09-04-gfx906-w4a16-p0-report.md`、`logs/p0/`、宿主 `/home/x/vllm-work/p0/` |
| P0.5 | `2026-09-08-gfx906-p05-report.md` + `logs/p0/p05_*.log`；探针 `p05_skinny_probe.py` / `p05_accuracy.py` / `p05_gate_test.py` / `p05_ab.sh` / `p05_cold_serve.sh` / `p05_serve_probe.sh` |
| P1 | `2026-09-09-p1-report.md`（v3/v4）+ `2026-09-08-p1-nibble-map.md`（L0 位表）+ `logs/p1/`；`p1_l0_mapping.py` / `p1_floor.py` / `p1_l1_bench.py` / `p1_gap_probe.py` |
| P2 | `2026-09-09-p2-report.md` + `logs/p2/`（Task 6：`p2_t6_serve_run{1,2}.log`、`p2_t6_probe_{on,off}.txt`、`p2_t6_thermal_run{1,2}.log`）；门脚本 `p2_plan_selfcheck.py` / `p2_route_gate.py` / `p2_gate0.py` / `p2_ab_gate.py` / `p2_l2_gate.py` / `p2_l3_{run,gate,ctrl}.py` / `verify_p2_blocks.py` / `p2_t5_{preflight,ab}.sh` / `p2_t6_serve.sh`（可重入双臂）/ `p2_t6_probe.sh <OUT> <arm> <cname>` |
| 交付配方 | **活的只有 4 件**：`docker/Dockerfile.rocm_gfx906`（238 行，两条路径各挂**一条 14 检查的断言链**）·
`docker/prepare_build_context.sh`（96 行 / 7 步抽取器）· `docker/entrypoint/{serve.sh(115),selfcheck.sh(125, 23 项),QUICKSTART.md(145)}` ·
`docker/patches/{layers_utils.py(425),p2_w4a16_gemv.hip(121)}`。**决策记录**（不进镜像）：`docker/BUILD_GUIDE.md`(168)。
**已退场、不要用**：`Dockerfile.rocm_gfx906.from-source.bak`（从源码全量构建，7 stage，从未使用；但里面存着
`NUM_SEGMENTS=64` 与 `PYTORCH_ROCM_ARCH` 的原始出处）、`build_gfx906.sh` / `deploy_and_build.sh`（调用形式已不被
现配方接受，唯一留存价值是国内 base 镜像源轮询）、`fix_gfx906_block.py`（改动仍活，但换了实现方式）、
`check_server.sh` / `test_vllm_basic.py` |
| 手写补丁权威源 | `container_src/`（`wna16_exllama.py` 486 行 + `q_gemm.cu`/`ops_stable.h`/`torch_bindings.cpp`/
`_custom_ops.py` + GDN 三件 + `fla_ops/`）——**经容器反向进 build-context，见 §2.3** |
| 设计/spec | `docs/superpowers/specs/2026-09-03-gfx906-w4a16-wave64-kernel-design.md`、四份计划 `docs/superpowers/plans/2026-09-0{4,8,9}-*` |

**两条工具链坑必须随报告一起传下去**（否则复现会静默得到空结果）：往容器喂脚本一律 `docker exec -i`
且**不要与 `| tee` 混用**（heredoc 会绑到流水线最后一段）；**driver 日志名必须带 run 号或用 `>>`**
（可重入脚本用同名 `>` 会覆盖上一 run 的证据 —— 产物存在 ≠ 证据存在）。
