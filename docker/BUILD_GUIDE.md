# gfx906 vLLM 环境：构建与运行指南（已验证版本，2026-08-31）

## 当前状态：全链路验证通过

- 模型：`Qwen3.8-27B-OBLITERATED-AWQ-W4A16`，TP2，TRITON_ATTN，compressed-tensors W4A16
- ExLlama WNA16 内核：输出与 Triton 逐字符一致，贪心解码确定性 `True`
- 性能（vs Triton 基线）：
  | 场景 | ExLlama | Triton | 加速 |
  |------|---------|--------|------|
  | batch1 decode | 14.6 tok/s | 6.1 tok/s | 2.40× |
  | batch8 decode | 74.2 tok/s | 44.7 tok/s | 1.66× |
- 环境版本：vllm 0.28.0+rocm724 / torch 2.13.0+rocm7.2 (hip 7.2.53211) /
  triton 3.8.0 / torchvision 0.28.0+rocm7.2 / transformers 5.16.1

## 环境里的四处 gfx906 定制（丢任何一处都会坏）

1. **gfx906 RCCL** → `dist-packages/torch/lib/librccl.so.1.0`（11,337,072 B，
   sha256 前缀 `5c5544a3`），符号链 `librccl.so → librccl.so.1 → librccl.so.1.0`；
   官方库留作 `librccl.so.official-backup`。原生 ROCm 7.2.4 librccl 没有
   gfx906 内核，NCCL 初始化会段错误。
2. **156 个 Tensile 补丁文件** → `dist-packages/torch/lib/rocblas/library/`
   （torch 自带 rocBLAS 的库目录；`/opt/rocm/lib/rocblas/library` 里没有也
   不影响）。
3. **vLLM 补丁文件**（均在 `dist-packages/vllm/` 下）：
   - `model_executor/kernels/linear/mixed_precision/wna16_exllama.py`
     （含 `gptq_shuffle` nibble 交错 + 空 `g_idx` 置换）
   - `model_executor/kernels/linear/__init__.py`（内核注册，`VLLM_WNA16_EXLLAMA=1` 门控）
   - `v1/attention/ops/triton_decode_attention.py`（gfx906 保持 BLOCK=32）
4. **运行环境变量**：`HSA_OVERRIDE_GFX_VERSION=9.0.6`（必需）、
   `VLLM_TRITON_ATTN_NUM_SEGMENTS=64`、`VLLM_WNA16_EXLLAMA=1`

## 运行容器（已验证命令）

```bash
docker run -d --name vllm-gfx906 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 16g --ipc=host \
  -v /home/x/models:/models:ro \
  -v /home/x/vllm-work:/work \
  -p 8000:8000 \
  vllm-gfx906:latest sleep infinity
```

`HSA_OVERRIDE_GFX_VERSION=9.0.6` 等运行时变量已烘进镜像 ENV，不必再 `-e`。

## 镜像内交付入口（2026-09-02 起）

| 命令 | 作用 |
|---|---|
| `gfx906-selfcheck [--no-gpu]` | 逐项校验 gfx906 关键件（RCCL/Tensile/W4A16/GEMV 修复）+ 真实 GPU 行为，全绿打印 `SELFCHECK PASSED` |
| `gfx906-serve [模型目录]` | 用实测通过的默认参数起 `vllm serve`；量化方式与视觉模态按模型 `config.json` 自动识别；全部默认值可用 env 覆盖 |
| `/opt/gfx906/QUICKSTART.md` | 面向使用者的快速上手（启动、调用、默认参数理由、已知限制、排查表） |

源文件在 `docker/entrypoint/`，Dockerfile 两条构建路径都 `COPY entrypoint/ /opt/gfx906/` 并软链到 `/usr/local/bin`，`prepare_build_context.sh` 会把它们抽回 build-context。

模型路径（本机）：`/home/x/models/vllm/Qwen3.8-27B-OBLITERATED-AWQ-W4A16`。
默认参数（写死在 `serve.sh`，不再需要手工传）：`dtype=float16, tp=GPU 数,
attention_backend=TRITON_ATTN, disable_custom_all_reduce=True,
cudagraph_mode=FULL_DECODE_ONLY + capture sizes<=48, max_num_seqs=48,
有 vision_config 时 limit_mm_per_prompt={"image":1,"video":1} +
mm_encoder_attn_backend=TRITON_ATTN`。回退对照用 `EAGER=1`。
`VLLM_ROCM_USE_SKINNY_GEMM` **故意不写进镜像 ENV / serve.sh**：代码默认 True，
gfx906 的 M=1 稠密 GEMV 快路径（P0.5）开箱即生效，需要对照时才手工设 `=0`。

## 环境固化（防止重建丢失）

### 快照（仅作开发回滚点，不作交付基底）

```bash
# 警告：必须 --no-pause！docker commit 默认会暂停容器，
# 本机（冻结持有 /dev/kfd 的进程）会永久挂死。
nohup docker commit --no-pause vllm-gfx906 vllm-gfx906:<milestone>-working \
  > /tmp/commit_snapshot.log 2>&1 &
```

`b3-graph-working` / `b4-gemvfix-working` / `c1-entrypoint-working` 是里程碑留痕。
**它们不能当交付基底**：commit 携带容器可写层里的 ~16 GB `/root/.cache`（pip 9.7G +
vLLM 编译 3.6G + 1.8G AOT W4A16，而缓存键不含内核选择），且 `phase3-working`（2026-08-31）
早于 B3/B4/C1，连 `Dockerfile` 自己的静态断言都过不去。⇒ 交付走下面的路径 B。

### Dockerfile（可复现配方）

`Dockerfile.rocm_gfx906` 提供两条路径，**先 B 后 A**：

- **路径 B（权威，从官方 ROCm 基础镜像离线重建）**：先用
  `prepare_build_context.sh` 从运行中容器抽取 `build-context/`
  （RCCL、156 个 Tensile 文件、`patches/` 下的 vLLM python 补丁、
  ~18G dist-packages 打包），然后：
  ```bash
  bash docker/prepare_build_context.sh          # 在 GPU 主机上运行
  docker build -f docker/Dockerfile.rocm_gfx906 --target rebuild \
      -t vllm-gfx906:rebuilt docker/build-context/
  ```
- **路径 A（默认 target）**：`FROM vllm-gfx906:rebuilt`，几秒的分层——重断言运行时
  ENV、从**仓库**重新 `COPY` `patches/layers_utils.py` 与 `entrypoint/`、跑同一组静态
  自检与"`/root/.cache` 必须为空"的泄漏断言：
  ```bash
  cd docker && docker build -f Dockerfile.rocm_gfx906 -t vllm-gfx906:latest .
  ```
  ⇒ 只改 python 补丁或入口脚本时用它；换了 C++/依赖则必须回到路径 B。

旧的"从源码全量构建"方案（gitee 镜像、从未使用、版本不符）归档在
`Dockerfile.rocm_gfx906.from-source.bak`。

### C2 交付构建验证（2026-09-02）

交付以**路径 B（从 base 离线重建）**为准，产物 `vllm-gfx906:rebuilt`（已同时打为 `:latest`）。

- 抽取：`bash docker/prepare_build_context.sh vllm-gfx906` → `build-context/` 共 7.0G（其中 dist-packages 包 6.8G）
- 构建：`docker build --target rebuild -t vllm-gfx906:rebuilt build-context/`，15 步全过，含 gfx906 关键件自检与"交付镜像内 `/root/.cache` 必须为空"的泄漏断言
- 体积：`inspect .Size` 22.07GB（= containerd 存储里压缩层 blob 之和，也等于 `docker save` 的流字节数）
  / `docker images` 69.8GB（解包态）。比快照派生的路径 A 产物（25.6GB / 83.1GB）小约 3.5GB
  ⇒ **传输按 22 GB 算，对端磁盘按 70 GB 算**，两者不是一回事（机制见 P0.5 复测那节）
- 全新容器（只带 QUICKSTART 里的 run flag + 只读模型挂载）实测：`gfx906-selfcheck` 全绿 → 冷启动 **711s**（torch.compile 384s）→ `Using WNA16ExllamaKernel`、KV 66,749 tokens、文本正确、warm decode **22.0 tok/s**、图像 6/6；第二次启动 **420s**（compile 31s，557M 缓存留在容器可写层）
- 顺带修掉配方缺陷：运行时 ENV 原先只写在路径 A 的 final 阶段，重建出的镜像里没有 `HSA_OVERRIDE_GFX_VERSION` / `VLLM_WNA16_EXLLAMA`（表现为静默退回 Triton W4A16 内核）→ 现在两条路径共用同一份 ENV 声明
- 编译缓存不进镜像（Dockerfile 里有断言兜着）；开发容器里的三份缓存留着自用
- 验证用的一次性容器名为 `vllm-gfx906-rc`，用完 stop，可 `docker rm -f` 重建

### P0.5 后的交付态复测与镜像归档（2026-09-08）

- 当前交付产物：`:latest` = **`b5931bb5c742`**（Path A，基于 Path B 产物 `:rebuilt` = `0d2dc5a67297`）。
  这一轮只改了随镜像分发的 `entrypoint/QUICKSTART.md`，流程按 P0.5 定的规矩走：仓库 → `docker cp`
  进开发容器 → `prepare_build_context.sh` → B → A → 镜像内 md5 复验 → 真机 `gfx906-selfcheck`
  （20 ok / 0 FAIL），三方 + 镜像内 md5 一次通过。**教训**：改 `docker/entrypoint/*` 一次的代价是
  一整轮 B→A 构建 + 一次 22 GB 重导（今天为这份文档走了三轮，ID 因此滚了三次）⇒ **攒够一批再走。**
- **交付时钟（sclk auto，不钉频）全新空缓存容器**：冷启动 **710s / torch.compile 384.32s**，
  同容器 warm **420s / 67.41s**（缓存 552 MB 留在可写层）⇒ 与 C2 的 711/384 一致，P0.5 换内核没有
  给启动加成本。warm 的 compile 项从 C2 的 31s 变成 67s 而总时长不变，**该项构成变化未查**。
- **serve 路径 decode 实测 32.4 tok/s**（64 token 请求，与 C2 那个 22.0 同探针 ⇒ **+47%**）；
  256 token 三次 32.5 / 32.6 / 32.6；8 并发合计 66.3 tok/s。KV：冷臂 67,356 / warm 臂 64,929
  （warm 两次同值），**这 3.6% 的冷热差未定位** ⇒ 容量规划按低值。
- **归档（回滚点，也给 #18 定量）**：`/home/x/image-export/vllm-gfx906-b5931bb5c742.tar.gz`
  = 21,948,506,214 B，`docker save` 原始流实测 22,070,416,384 B ⇒ **pigz 只省 0.6%（1.01×），261s**。
  自包含性证明 = tar **32 个成员、成员字节和 22,070,390,489 B = 原始流的 100.0%**，三个 ~7.3 GB blob
  占全文 99.6%；`pigz -dc | tar -tvf -` 全量回读 21s（顺带验流未损）；`docker load` 回读 ID 一致。
  目录里另有两份同尺寸归档（`1e18c8ea6b8b`、`6ec669c822e8`），只差那份文档，可按需删。
- **为什么 .Size 与 docker images 差 3 倍、为什么压不动**（一条机制解释两个现象）：宿主 docker 用
  **containerd 镜像存储**（`Storage Driver: overlayfs` + `driver-type: io.containerd.snapshotter.v1`，
  media type `oci.image.index.v1+json`），`docker save` 导出的是 `blobs/sha256/<digest>` 形式的
  **已压缩层 blob**（所以 tar 里没有 `layer.tar`）。`.Size` 是压缩态之和 = save 流字节数；
  `docker images` 那栏是解包态。⇒ 传输按 22 GB、对端磁盘按 70 GB，再压一遍是白工。
- 载体脚本（都在仓库根，宿主 `/home/x/vllm-work/p0/`）：`p05_cold_serve.sh`（冷+warm+首次归档）、
  `p05_serve_probe.sh`（tok/s，model id 从 `/v1/models` 解析）、`p05_ship_doc.sh`（entrypoint 推送
  到位 + 两条路径重建 + 复验）、`p05_export_image.sh`（原始流测长 + 归档 + 回读）。
  日志：`logs/p0/p05_cold.log`、`logs/p0/p05_probe.log`。

## 运维注意事项（血泪教训）

1. **主机不稳定**：192.168.10.11 会无内核日志地整机掉电。长任务一律
   `docker exec -d ... > /tmp/run.log 2>&1` 脱离 SSH 跑，再轮询日志。
2. **不要 `docker pause` / 默认暂停式 `docker commit`**：冻结 + /dev/kfd
   会挂死 containerd 的 diff 服务（dockerd 零 I/O 空等）。已验证
   `--no-pause` 正常。若 containerd 已被拖死，`systemctl restart containerd docker`
   后 `docker start vllm-gfx906` 可恢复（容器可写层不受影响）。
3. **CUDA Graph 不可用**：`ops.gptq_gemm` 走 torch stable-ABI 分配
   （`aten::empty`），捕获期失败。`enforce_eager=True` 为默认（确定性）；
   `compilation_config.cudagraph_mode="NONE"` 更快（batch1 ~20 tok/s）但
   贪心输出不确定。
4. Tensile 补丁若需手工恢复：`cp /work/gfx906-supplements/* \
   /usr/local/lib/python3.12/dist-packages/torch/lib/rocblas/library/`。

## 性能参考（Z100 文章目标，尚未全部达成）

| 场景 | 目标 | 当前 |
|------|------|------|
| 27B-INT4 decode TP2 | ~33 tok/s | 14.6 tok/s（batch1，ExLlama） |
| 8K prefill TP2 | ~399 tok/s | 未测（长上下文待做） |
