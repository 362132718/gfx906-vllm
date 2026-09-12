# vLLM on gfx906 —— MI50 / Radeon Pro VII 可用的 vLLM 交付镜像

把 **vLLM v0.28.0** 完整移植到 **AMD gfx906（GCN 5.1 / wave64）**：Radeon Pro VII、Instinct MI50/MI60 等每卡 16 GB HBM2、**无 MFMA、无原生 BF16/FP8 矩阵指令**的卡。

交付物是一个 Docker 镜像：ROCm 7.2.4 + torch 2.13.0+rocm7.2 + triton 3.8.0 + vLLM 0.28.0，内置全部 gfx906 适配（RCCL、Tensile、W4A16 ExLlama 桥接、Triton BLOCK=32、rocBLAS GEMV 规避、M=1 手写 GEMV、MoE 内核调优配置），并带 `gfx906-selfcheck` / `gfx906-serve` 两个开箱入口。

> **模型权重不在镜像内**，需自行挂载。

---

## 交付物

| 物件 | 说明 |
|---|---|
| `vllm-gfx906:latest`（镜像 ID `c31a8b65273c`） | 69.8 GB / 29 层；`docker save` 归档 22.07 GB，**自包含已验证**（manifest 引用的 30 个 blob 全部在档） |
| `vllm-gfx906-c31a8b65.tar` | 上面那份归档，`docker load -i` 即可导入，**不依赖本机任何已有 blob** |
| 本仓库 | 配方（Dockerfile 两条路径 + prepare 脚本 + 补丁源）与测量/验收工具链 |

⚠ 归档 22 GB，但**解包后约 70 GB** —— 磁盘按 70 GB 预算，别按归档大小算。

---

## 快速开始

### 1. 导入并起容器

```bash
docker load -i vllm-gfx906-c31a8b65.tar        # → vllm-gfx906:latest

docker run -d --name vllm-gfx906 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v /path/to/models:/models:ro \
  -p 8000:8000 \
  vllm-gfx906:latest sleep infinity
```

### 2. 自检（应打印 `SELFCHECK PASSED`）

```bash
docker exec -it vllm-gfx906 gfx906-selfcheck        # 只查文件不碰 GPU：--no-gpu
```

逐项确认 RCCL、Tensile 补丁、W4A16 内核、GEMV 修复与 GPU 实际行为。

### 3. 起服务

```bash
docker exec -it vllm-gfx906 gfx906-serve /models/Qwen3.6-35B-A3B-AWQ-4bit
```

不带路径时自动在 `/models` 下找唯一模型。**首次启动 7–13 min 是正常的**（torch.compile）；编译产物在容器可写层，`docker stop/start` 保留，`docker rm` 后才重编。

### 4. 调用（OpenAI 兼容）

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<模型名>","max_tokens":64,"messages":[{"role":"user","content":"你好"}]}'
```

模型名 = `SERVED_NAME`（默认模型目录名），`curl localhost:8000/v1/models` 可查。图像/视频走 `image_url` / `video_url`（视频必须是真实 mp4 文件，可 base64 data URL）。

---

## 实测性能（交付镜像 `c31a8b65`，TP2，2×gfx906，默认参数）

| 模型 | 单流 decode | 8 路并发聚合 | KV tokens | 文本 | 图片 | 视频 |
|---|---|---|---|---|---|---|
| **Qwen3.8-27B**（dense，AWQ W4A16） | **34.2 tok/s** | — | 67,356 | ✓ | **6/6** | **3/3** |
| **Qwen3.6-35B-A3B**（MoE，AWQ 4bit） | **46.3 tok/s** | **84.2 tok/s** | 57,344 | ✓ | **6/6** | **3/3** |

> 口径：p8 验收探针（HTTP API、ctx≈1.6K、64 token、warmup + 3 次取中位）。
> 27B 在 QUICKSTART 里的交付基线是 **32.4 tok/s**（64-token 档、三次取均值）—— 两者探针不同，不要直接混比。

### MoE 这 46 tok/s 是怎么来的（同口径三点矩阵）

| 配置 | 步长 | 单流 tok/s |
|---|---|---|
| 默认 | 36.93 ms | 27.08 |
| + `shared_expert_gate` 的 **m==1 快路径** | 26.38 ms | 37.91 |
| + **MoE 内核调优配置** | 21.72 ms | **46.03** |

两项都是**零重编**改动（一个 python 分支 + 一个 kernel config JSON），合计 **+70%**。
27B 的对应改进（P0.5 skinny GEMV）已在镜像里默认开启。

---

## 默认参数及理由（都可 env 覆盖）

| 默认 | 原因 |
|---|---|
| `--attention-backend TRITON_ATTN` | gfx906 无 MFMA，ROCm 原生 attention 内核不可用 |
| `--dtype float16` | 无原生 BF16 矩阵路径；FP16 实测比 BF16 快约 2.8× |
| `FULL_DECODE_ONLY` + capture sizes `[1..48]` | CUDA graph 让 decode +50%；4-bit fused GEMM 只覆盖到 M≤50 |
| `--max-num-seqs 48` | 必须同时 ≤ capture sizes 上限与 GDN/Mamba 状态块数（见下） |
| `--disable-custom-all-reduce` | PCIe P2P 上 RCCL 延迟已持平，custom AR 无收益 |
| 有 `vision_config` 时 image/video 各 1 + `TRITON_ATTN` | 图像与视频实测可用 |

可覆盖项：`TP DTYPE MAX_LEN GPU_MEM_UTIL MAX_NUM_SEQS PORT HOST SERVED_NAME QUANTIZATION MM EAGER CUDAGRAPH_MODE CAPTURE_SIZES CUSTOM_ALL_REDUCE MODELS_ROOT`；额外参数原样透传给 `vllm serve`。
内核级开关：`VLLM_WNA16_P2_GEMV=0` 关手写 GEMV（−3%）、`VLLM_WNA16_EXLLAMA=0` 退回 Triton W4A16（慢 ~8×，仅排查）。

---

## 模型兼容矩阵

| 层级 | 模型 | 依据 | 承诺 |
|---|---|---|---|
| **T1 已验收** | Qwen3.8-27B dense（AWQ W4A16，48 GDN + 16 full attn） | 端到端实测 | 34.2 tok/s；图像/视频可用 |
| **T2 已验收** | **MoE** `Qwen3.6-35B-A3B-AWQ-4bit`（30 GDN + 10 full attn，256 专家 top-8+1 共享） | 端到端实测 | 46.3 tok/s；图像/视频可用 |
| T3 结构同构 | 其他 W4A16 / GPTQ-Int4 dense | 结构同构 | 可试，未定测 |

---

## 常见问题排查

| 现象 | 处理 |
|---|---|
| NCCL/RCCL 初始化段错误 | `librccl` 不是 gfx906 构建 → 看 `gfx906-selfcheck` 前两项 |
| 普通 matmul 段错误 | Tensile gfx906 补丁缺失 → `selfcheck` 的 kernel 计数项 |
| 输出乱码 | W4A16 桥接或 `gptq_gemm_out` 缺失 → `selfcheck` 后四项；再 `EAGER=1` 对照 |
| **decode 只有个位数 tok/s** | 走了 Triton W4A16（实测 3.9 tok/s）而非 ExLlama 桥接 → 确认 `VLLM_WNA16_EXLLAMA=1`，日志应出现 `Using WNA16ExllamaKernel` |
| **`max_num_seqs (N) exceeds available Mamba cache blocks (M)`** | GDN/线性注意力每个 decode 序列占 1 个 Mamba block，块数只取决于权重后的剩余显存（与 `MAX_LEN` 无关）。`gfx906-serve` 默认 48 在安全线内；若调低 `GPU_MEM_UTIL`/开视觉塔/换更大模型，按报错里的 M 下调。上游还有一招：**降 `--max-cudagraph-capture-size`（默认 512）可保住 max-num-seqs 的同时缩小图池**。`EAGER=1` 不检查此项 |
| **输出全是同一个 `!`**（短 prompt 正常，长上下文才出现） | 上游 #55291/#54146：GDN readout 在 SSM state cache 为 fp32 时溢出 fp16 → inf → NaN。**本镜像已实测不触发**（三层证据：内核级 44/44 PASS；14K 无 collapse；14K 下 3 个短问题全答对并自然 EOS）。仍遇到则 `EAGER=1` 对照，并把 `fused_sigmoid_gating.py` 的 readout 缓冲改 fp32。**另注意：若要省内存改 `--mamba-ssm-cache-dtype`，请用 `bfloat16` 而非 `fp16`** —— bf16 保留 fp32 的指数位，从构造上不会溢出（上游 Qwen3.5 生产配方即用 bf16） |
| **prefix cache 命中率近 0** | GDN 混合模型默认 `mamba_cache_mode=align` 只缓存 1–2 个尾部块，上游 Qwen3.5/3.6 系列已知（修复 PR #36649 尚未合并，且其 "all" 模式**要求** fp32 SSM cache）。align + 前缀缓存上游标注为**实验特性** |
| 启动 profile 报 `expected size 640==5120, stride 8192==1024` | 编译缓存与 W4A16 内核选择串台 → 用 `gfx906-serve` 起即自动隔离；手工起可清 `~/.cache/vllm*` |
| 视频请求 400 | 内容块必须是 `{"type":"video_url","video_url":{"url":...}}` 且指向真实 mp4；帧列表写法不被接受 |
| OOM | 降 `MAX_LEN` / `GPU_MEM_UTIL`，或 `--limit-mm-per-prompt` 里 image/video 设 0 |
| 想逐项排除图执行问题 | `EAGER=1` vs 默认，对比 `[serve]` 行与 tok/s |

**给要打补丁的人**：AOT/torch.compile 追踪期若报 `'NoneType' object has no attribute 'size'` 并炸掉 `MM=1`（视觉开启）的引擎初始化 —— 检查是否对 vLLM 的 `BasevLLMParameter` 调了张量方法（如 `weight.reshape(-1)`）。它会走 `parameter.py __torch_function__` 派发；**必须用广播等价写法**（详见 `docker/patches/layers_utils.py` 内注释与 `moe_tuning/verify_m1_gate.py`）。

---

## 从源码构建

两条路径，**先 B 后 A**：

```bash
# 路径 B（权威，从官方 ROCm 基底离线重建）—— 从一个运行中的 dev 容器抽取 build-context
bash docker/prepare_build_context.sh <container>
docker build -f docker/Dockerfile.rocm_gfx906 --target rebuild \
    -t vllm-gfx906:rebuilt docker/build-context/

# 路径 A（默认 target，FROM :rebuilt，秒级）—— 只改 python 补丁或入口脚本时用
cd docker && docker build -f Dockerfile.rocm_gfx906 -t vllm-gfx906:latest .
```

**⚠ 三个会静默产出错镜像的坑**（2026-09-11 实测）：

1. **抽取源容器必须选对** —— 默认值 `vllm-gfx906` 可能是陈旧的。构建前先跑逐条断言体检：
   `bash moe_tuning/assert_probe.sh <container>`，**全过才动手**。
2. **宿主 `docker/` 与仓库 `docker/` 是两份**，`prepare` 会从容器抽、路径 A 会从宿主 COPY ⇒ **两边都要同步**（md5 核对）。
3. **`prepare` 的 `[7/7]` 若发现 tar 已存在会跳过重打** ⇒ 容器侧改动会静默丢失，必须 `FORCE=1`，并从 tar 里取出文件比对 md5。

构建耗时参考：`prepare` ~5 min（含重打 7.3 GB tar）→ 路径 B ~9 min（19 步，含断言）→ 路径 A ~20 s。

---

## 仓库结构

```
docker/                    构建配方：Dockerfile.rocm_gfx906（两条路径）、prepare_build_context.sh、
                           patches/（补丁源）、entrypoint/（QUICKSTART + selfcheck + serve）
container_src/             补丁的仓库权威副本（A 类正向安装的源端）
moe_tuning/                MoE 内核调优器与产物、逐进程算子普查、m==1 验证、
                           GDN 长上下文探针、断言体检、三点矩阵驱动、bf16 SSM 探针
p8_accept*.sh              27B/35B 开箱三模态验收
p8_verify_tar.sh           交付 tar 自包含性校验
p5_decode.py / p5_kernels.py / p5_span.py / p7_peer_bw.py / lcpp_ref.sh
                           测量方法学（前缀缓存/warmup/trace 三坑的解法）
docs/superpowers/          各阶段计划、交接文档、性能归因报告
```

---

## 已知限制

- **不要用 BF16/FP8 权重**：gfx906 缺相应矩阵指令，请用 FP16 或 W4A16/W8A16
- **27B 级别需 TP2**：单卡 16 GB 放不下权重 + KV
- 极小 batch 的 4-bit fused GEMM 存在 1-ULP 跨运行差异（atomicAdd 累加）；贪心解码实测稳定
- **GDN + CUDA 图的 `--max-num-seqs` 硬上限**（见排查表）；此约束只作用于 `FULL_DECODE_ONLY`
- **`--enable-expert-parallel` 在 2 卡 PCIe（无 XGMI）上是负收益，不要开**；上游也确认 EP 与混合 GDN 架构存在死锁（issue #41862）
- PP（`--pipeline-parallel-size 2`）实测比 TP2 慢 18%，对单流 decode 无益
- **MTP 投机解码未启用**：TP2 上实测负收益
- 宿主机 `dmesg` 有 amdgpu 掉卡、或长任务静默失联：gfx906 平台已知稳定性问题，服务用 `-d` 后台跑并落日志
- V2 model runner 与混合 GDN 架构不兼容（上游 issue #38041）⇒ 保持默认 V1

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `docker/entrypoint/QUICKSTART.md` | 面向使用者的完整上手与排查表（镜像内同路径） |
| `docker/BUILD_GUIDE.md` | 构建配方与运维注意事项 |
| `docs/superpowers/2026-09-11-moe-tuning-results.md` | MoE 性能归因、GDN 风险核查、P7/P8 全过程 |
| `docs/superpowers/plans/2026-09-10-gfx906-v3-deliverable-image-plan.md` | 交付镜像总计划（P0–P8） |
| `moe_tuning/` | 调优器、产物 config、普查钩子、探针与验收脚本 |

## 致谢

基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)（v0.28.0）与
[nlzy/vllm-gfx906](https://github.com/nlzy/vllm-gfx906) 的社区工作；MoE 内核调优器取自上游
`benchmark_moe.py`。所有 gfx906 特定补丁与测量方法学见本仓库。
