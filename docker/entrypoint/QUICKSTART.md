# vLLM on gfx906 — 快速上手

**适用硬件**：gfx906 / GCN 5.1 / wave64 —— Radeon Pro VII、MI50、MI60、S7150、Z100 等（每卡 16GB HBM2，无 MFMA、无原生 BF16/FP8 矩阵指令）。

**镜像内容**：ROCm 7.2.4 基底 + torch 2.13.0+rocm7.2 + triton 3.8.0 + vLLM 0.28.0（源码构建，含 gfx906 补丁：RCCL、Tensile、W4A16 ExLlama 桥接、Triton BLOCK=32、rocBLAS GEMV 规避）。

**模型权重不在镜像内**，需要自行挂载。

---

## 1. 起容器

```bash
docker run -d --name vllm-gfx906 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v /path/to/models:/models:ro \
  -p 8000:8000 \
  vllm-gfx906:latest sleep infinity
```

如果拿到的是 tar 包，先导一次：`docker load -i vllm-gfx906.tar`

归档本身约 **22 GB**（层在 containerd 存储里已是压缩态，再 gzip 只能省 0.6%），但**解包后要 ~70 GB
磁盘** —— 准备磁盘请按 70 GB 算，不要按归档大小算。

## 2. 自检

```bash
docker exec -it vllm-gfx906 gfx906-selfcheck
```

逐项确认 gfx906 关键件（RCCL、Tensile 补丁、W4A16 内核、GEMV 修复）以及 GPU 实际行为，最后打印 `SELFCHECK PASSED` 再继续。只想查文件不碰 GPU：`gfx906-selfcheck --no-gpu`。

## 3. 起服务

```bash
docker exec -it vllm-gfx906 gfx906-serve /models/Qwen3.8-27B-OBLITERATED-AWQ-W4A16
```

不带路径时会自动在 `$MODELS_ROOT`（默认 `/models`）下找唯一模型。第一行 `[serve] ...` 打印解析后的实际配置，方便确认。

**首次启动慢是正常现象**：默认开启 CUDA graph，torch.compile 要几分钟。交付镜像（无任何编译缓存）实测：**冷启动 710s 就绪**（其中编译 384s），**有缓存 420s**（编译 67s）。两个数都是在含 P0.5 改动的镜像上、全新容器 + 交付时钟（sclk auto）重测的：冷启动与 P0.5 之前的 711s/384s 逐项一致；warm 臂总时长仍是 420s，但日志里的 `torch.compile took` 从之前的 31s 变成 67s（**该项构成变化的原因未细查**，端到端等待时间没变）。编译产物落在 `~/.cache/vllm-w4a16-exllama`（552MB），位于容器可写层——`docker stop/start/restart` 会保留，只有 `docker rm` 后新建容器才会再编译一次。若改了 `MAX_LEN`/`TP`/图模式/内核选择导致编译键变化，或想跳过编译对照性能，可用 `EAGER=1`。

## 4. 调用

```bash
# 文本
curl -s localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"<模型名>","prompt":"The capital of France is","max_tokens":32,"temperature":0}'

# 图像（模型带 vision tower 时）
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<模型名>","max_tokens":64,"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"http://<可访问地址>/img.png"}},
        {"type":"text","text":"描述这张图"}]}]}'

# 视频：必须是真实视频文件（mp4）；帧列表不被 serving 接口接受
# base64 装载示例：VID=$(base64 -w0 clip.mp4)
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<模型名>","max_tokens":64,"messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"data:video/mp4;base64,'"$VID"'"}},
        {"type":"text","text":"视频里什么颜色的什么形状在往哪个方向移动？"}]}]}'
```

模型名 = `SERVED_NAME`（默认为模型目录名）。`curl localhost:8000/v1/models` 可查。

## 5. 默认参数及理由

`gfx906-serve` 里每个默认值都是在这块卡上实测出来的，用 env 覆盖即可：

| 默认 | 原因 |
|---|---|
| `--attention-backend TRITON_ATTN` | gfx906 无 MFMA，ROCm 原生 attention 内核不可用 |
| `--dtype float16` | 无原生 BF16 矩阵路径；FP16 实测比 BF16 快约 2.8× |
| `FULL_DECODE_ONLY` + capture sizes `[1..48]` | CUDA graph 让 decode 提升约 50%；4-bit fused GEMM 只覆盖到 M≤50 |
| `--max-num-seqs 48` | 必须同时 ≤ capture sizes 上限与 GDN/Mamba 状态块数 |
| `--disable-custom-all-reduce` | PCIe P2P 上 RCCL 延迟已持平，custom AR 无收益且多一个 stale-data 风险面 |
| 有 `vision_config` 时 `--limit-mm-per-prompt {"image":1,"video":1}` + `--mm-encoder-attn-backend TRITON_ATTN` | 图像与视频都实测可用；vision tower 同样不能走 MFMA 内核 |
| GEMV 修复进程启动自动生效 | rocBLAS fp32/fp64 matrix×vector 在 gfx906 直接段错误（`.pth` 注入，非 sitecustomize） |

可覆盖项：`TP DTYPE MAX_LEN GPU_MEM_UTIL MAX_NUM_SEQS PORT HOST SERVED_NAME QUANTIZATION MM EAGER CUDAGRAPH_MODE CAPTURE_SIZES CUSTOM_ALL_REDUCE MODELS_ROOT`

内核级开关不走 `gfx906-serve` 的参数，直接以 env 传给容器里的 python：`VLLM_WNA16_P2_GEMV=0`
关掉两族 M=1 手写 GEMV（默认开，退回 ExLlama 融合内核，代价约 −3%）、`VLLM_WNA16_EXLLAMA=0`
退回 Triton W4A16（慢约 8×，只用于排查）。

`gfx906-serve` 还会按 W4A16 内核选择隔离编译缓存目录（`~/.cache/vllm-w4a16-exllama` / `...-triton`）。原因：vLLM 的编译缓存键包含模型/并行/图配置但不包含内核选择，而两种内核对 4-bit 权重做了不同重排，混用缓存会在启动 profile 阶段直接报 stride 不匹配。

例：单卡小模型 `TP=1 MAX_LEN=8192 gfx906-serve /models/Qwen3-0.6B`；怀疑图执行时 `EAGER=1 gfx906-serve ...` 对照。量化方式默认由模型 `config.json` 自动识别，需要时可 `QUANTIZATION=compressed-tensors`。

## 6. 实测基线（Qwen3.8-27B AWQ W4A16，TP2，2×gfx906）

- decode：**serve 路径在交付时钟（sclk auto，不钉频）实测 batch1 32.4 tok/s**（64 token 请求；256
  token 长解码三次 = 32.5 / 32.6 / 32.6，取满 3 次即 **32.6**）；8 并发合计 **66.3 tok/s**
  （每流 8.3）。64 token 这一档与 P0.5 之前的 22.0 tok/s 是**同一探针口径** ⇒ 可直接比，**+47%**。
  eager 时代的 14.6 tok/s 到 graph 后 22.0，是这一档更早的基线。
- **#20-P2（本镜像已含，默认开启；`VLLM_WNA16_P2_GEMV=0` 可逐项退回 ExLlama 融合内核）**：
  (5120,8192)/(5120,7168) 两族的 M=1 AWQ-W4A16 GEMV 换上手写 wave64 内核。**同机、同镜像、同容器、
  同探针的双臂 A/B**（交付时钟 auto、全新容器、开箱 `gfx906-serve`）：
  **on = 33.3 tok/s**（64 token 档；256 token 三次 33.5 / 33.4 / 33.5 ⇒ mean **33.5**）
  vs **off = 32.3 tok/s**（256 三次 32.5 / 32.4 / 32.3 ⇒ mean **32.4**）⇒ **delta +1.0 tok/s（+3.1%）**，
  换成每步时间即 **930 µs**，与引擎内 device-trace 测到的 **812–826 µs/step** 相符（serve 侧没有
  隐藏折损）。off 臂 32.3 与 P2 前那次 32.4 只差 0.1 ⇒ 旧基线在同一台机器上可复现。
  开工前写死的验收门是 **≥33.4** ⇒ **字面差 0.1 未过**，而这一步收益的算术上限是
  **33.18–33.28**（取中点 819 µs：`32.4/(1−819/30860)=33.283`）⇒ 设备侧那份时间已全额转成吞吐，
  过不了的是门不是实现；门按用户裁定**不放宽**。**因此本节引用的交付基线仍是 32.4**，33.3 只作为
  "含 P2 的实测值"记录在此；复现口径 = §4 的 completions 调用（`temperature=0`，64 / 256 token
  两档，取满三次）。两臂热态不严格相同（on 在冷卡 43 °C、off 在 capture 后余温 65–70 °C），
  方向上只会让 +1.0 偏保守。
- **P0.5（本镜像已含）**：M=1 稠密 fp16 GEMM（GDN `in_proj_ba` + `lm_head`）改走 ROCm skinny
  `LLGemm1_kernel`。离线 harness 口径（1425/1000 MHz 钉频、batch1、`FULL_DECODE_ONLY`、贪心 256
  token 逐字不变）step **53.663 → 33.008 ms = 18.6 → 30.3 tok/s（+62.6%）**。
  此前按 sclk 比例从 30.3 外推交付态约 36 tok/s，**实测只有 32.6 —— 外推偏乐观约 10%**：decode 里
  访存受限的那半不随 sclk 缩放，且外推拿的是离线 harness、实测拿的是 serve（多一层 API/调度/分词
  开销）。两个口径不要混用。满载结温超 100°C 会降频，取数请等冷卡。
- **长上下文 decode（serve/graph/交付时钟 auto，batch1，64 token，每档三次取 warm 均值）**：
  2K **33.36** / 8K **31.95** / 16K **28.56** tok/s ⇒ **2K→16K 只降 14.4%**，gfx906 上没有
  "长上下文必塌方"那一类退化（分段 split-KV 路径默认可用）。对应步时 29.98 / 31.30 / 35.02 ms。
- **TTFT 同批**：2K **4.7 s** / 8K **23.5–24.7 s** / 16K **58.7–64.9 s**（冷启动另加约 720 s）。
  长上下文场景先撞墙的是这个，不是 tok/s。
- **上下文上限的边界行为**：prompt + 请求输出一旦超过 `--max-model-len 16384` 返回 **HTTP 400**
  （不是截断、不是静默降级）⇒ 客户端要预留输出 token 预算。
- prefill（A2 基线，eager 口径）：2K **442.8** / 8K **346.6** / 16K **235.0** tok/s
- prefill（**serve + graph + 交付时钟**，与上面长上下文 decode 同批、warm 均值）：
  2K **434.8** / 8K **338.1** / 16K **249.9** tok/s。与 eager 口径逐档比 **−1.8% / −2.5% / +6.3%**
  （两批 prompt 长度不完全相同）⇒ 没有系统性差异，即 **CUDA graph 对 prefill 没有可测贡献**
  （decode 那一档才有）；对外引用 prefill 用这一行。
- KV 容量：默认图像配置下 **66,749 tokens**（16K 上下文可 4.07 并发）；纯文本 18K × 5 并发实测 92,842 tokens
- KV 容量（`gfx906-serve` 路径）：冷臂 **67,356**（4.11×）/ warm 臂 **64,929**（3.96×，两次启动给出同一数字）/ P2 A/B 复用容器且 AOT 缓存命中的那次 **64,322**（3.93×）。**冷高 warm 低这 3.6% 未定位**，别把 KV 当成一个恒定值来容量规划
- 带图请求：全热态单张图问答 5.0s（关键词判定 6/6）
- 带视频请求（eager 口径，合成"红方块左右移"）：4 / 8 / 16 帧 = prompt 311 / 551 / 1031 tokens，4.3 / 5.6 / 7.3s，方向与颜色判定 3/3
- 首次带视觉启动慢（ViT Triton JIT）；镜像不含 Triton 缓存，但已包含在上面的冷启动计时内

## 7. 已知限制

- **视频输入可用**（eager 口径实测：16 帧约 1k prompt tokens、7.3s，颜色/形状/方向判定 3/3）。帧数或分辨率太大时可限流：`MM='{"image":1,"video":1}'` 配合 `--mm-processor-kwargs '{"size":{"longest_edge":4194304,"shortest_edge":65536}}'`（`longest_edge` 越小，profile 与单次视频的 token 越少）。CUDA graph + 视频的复验仍在收尾，若图模式下有异常先用 `EAGER=1` 对照
- **MTP 投机解码未启用**：TP2 上实测负收益
- **不要使用 BF16/FP8 权重**：gfx906 缺相应矩阵指令，请用 FP16 或 W4A16/W8A16
- **27B 级别需 TP2**：单卡 16GB 放不下权重 + KV
- 极小 batch 的 4-bit fused GEMM 存在 1-ULP 跨运行差异（内核用 atomicAdd 累加）；贪心解码输出实测稳定
- 若宿主机上 `dmesg` 有 amdgpu 掉卡记录，或长任务静默失联：这是 gfx906 平台的已知稳定性问题，建议服务用 `-d` 后台跑并落日志
- **GDN/线性注意力 + CUDA 图：`--max-num-seqs` 有硬上限。** 每个 decode 序列占 1 个 Mamba cache block；16K + `GPU_MEM_UTIL=0.90` 下只有 **53 个**，而 vLLM 默认是 1024 ⇒ 图捕获会直接拒绝启动：`ValueError: max_num_seqs (N) exceeds available Mamba cache blocks (M)`。`gfx906-serve` 的默认 `MAX_NUM_SEQS=48` 正好在安全线内；若调低 `GPU_MEM_UTIL`、开视觉塔或换更大模型，请按报错里的 M 值下调。此约束只作用于 `FULL_DECODE_ONLY`（`EAGER=1` 不检查）。注意 block 数只取决于**权重之后的剩余显存**，与 `MAX_LEN` 无关
- **MoE 模型可用（`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` 实测）**：TP2 可加载可生成，专家走 Triton WNA16 内核。实测单流 **48.9 tok/s**（16K、ctx≈1.6K、图模式、`MM=0`；开视觉塔略低且更吃显存），8 路并发聚合 71–75 tok/s。⚠️ 专家占权重 76.5%，**每 token 每卡只读 ≈2.8 GB** ⇒ 瓶颈是每步固定开销而非带宽；`--enable-expert-parallel` 在 2 卡 PCIe（本机**无 XGMI**，跨卡 2 跳）上是负收益，**不要开**。另：PP（`--pipeline-parallel-size 2`）实测比 TP2 慢 18%（21.2 vs 25.0 tok/s），对单流 decode 无益

## 8. 排查

| 现象 | 处理 |
|---|---|
| 起服务在 NCCL/RCCL 初始化时段错误 | `librccl` 不是 gfx906 构建 → 看 `gfx906-selfcheck` 前两项 |
| 普通 matmul 段错误 | Tensile gfx906 补丁文件缺失 → `selfcheck` 的 kernel 计数项 |
| 输出乱码 | W4A16 桥接或 `gptq_gemm_out` 不在 → `selfcheck` 后四项；再 `EAGER=1` 对照 |
| **输出全是同一个 `!`**（且短 prompt 正常，**长上下文到一定长度后才出现**） | 这是**另一种病因**。上游 vLLM #55291/#54146：GDN（线性注意力）层的 readout 在 SSM state cache 为 fp32 时累加值超出 fp16 上限 65504 → 变 `inf` → `inf/inf = NaN` → 贪心解码退化成单 token 无限重复。**本镜像已实测不触发**，三层证据：① 内核级 `gdn_verify.py` **44/44 PASS**（0 nan）；② 14K 上下文无 collapse（`longest_run=1`）；③ 14K 上下文下 3 个短问题**全答对且自然触发 EOS**（`len=7~9`）。若你仍遇到该症状，改用 `EAGER=1` 对照，并把 `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py` 里 readout 缓冲 dtype 改 fp32（即 #54146 那行）后重启 |
| 启动 profile 阶段报 `expected size 640==5120, stride 8192==1024` | 编译缓存与当前 W4A16 内核选择串台（缓存键不含内核选择）→ 用 `gfx906-serve` 起即自动隔离；手工起可清 `~/.cache/vllm*` |
| decode 只有个位数 tok/s | 多半是走了 Triton W4A16（实测 3.9 tok/s）而不是 ExLlama 桥接（交付时钟 serve 路径实测 **32.4 tok/s**，P0.5 前是 21.9）→ 确认 `VLLM_WNA16_EXLLAMA=1`，看日志 `Using WNA16ExllamaKernel` |
| 视频请求返回 400 Bad Request | 内容块必须是 `{"type":"video_url","video_url":{"url":...}}` 且 url 指向真实视频文件（mp4，可用 base64 data URL）；帧列表写法不被 serving 接口接受 |
| OOM | 降 `MAX_LEN` / `GPU_MEM_UTIL`，或 `--limit-mm-per-prompt` 里 image/video 设 0 |
| 想逐项排除图执行问题 | `EAGER=1` vs 默认，对比 `[serve]` 行与 tok/s |
