# 恢复 / 重建手册（vllm-gfx906 on AMD gfx906）

> 项目已于 2026-09-18 收摊冻结。本文是**唯一的恢复入口**。
> 冻结版本：`vllm-gfx906:latest` = `2f6d5bf6b47e`（含 P0.6 修正）

---

## 0. 先说结论：什么能重建，什么不能

| 问题 | 答案 |
|---|---|
| 只靠这个 GitHub 仓库能重建镜像吗？ | **不能。** 见 §3 的循环依赖 |
| 需要一个"根产物"吗？ | **需要。** 一份 `docker save` 出来的 tar |
| 有 tar 之后能完整重建吗？ | **能。** 见 §4 |
| 镜像里有没有模型权重？ | **没有。** 权重需另配（见 §6） |

**一句话**：**tar（1 个文件）+ 本仓库 = 完整可重建。** 服务器上其余 vLLM 产物都可以删。

---

## 1. 冻结了什么

| 项 | 值 |
|---|---|
| 镜像 tag | `vllm-gfx906:latest` |
| 镜像 ID | `sha256:2f6d5bf6b47e…` |
| 离线归档 | `images/vllm-gfx906-latest-2f6d5bf6.tar`（`docker save`，约 20 GB） |
| vLLM / torch | `0.28.0+rocm724` / `2.13.0+rocm7.2` |
| triton / transformers | `3.8.0` / `5.16.1` |
| 基座 | `rocm/dev-ubuntu-24.04:7.2.4-complete`（**公开可 pull**） |
| 交付入口 | `/opt/gfx906/{serve.sh,selfcheck.sh,QUICKSTART.md}`，symlink 为 `gfx906-serve` / `gfx906-selfcheck` |
| 目标硬件 | gfx906 / GCN 5.1 / wave64（Radeon Pro VII、MI50、MI60、S7150、Z100） |
| 验证过的模型 | `Qwen3.8-27B-OBLITERATED-AWQ-W4A16`（TP2，TRITON_ATTN，W4A16 ExLlama） |

---

## 2. 从 tar 立即恢复（不重建，最快）

```bash
docker load -i images/vllm-gfx906-latest-2f6d5bf6.tar
docker images | grep vllm-gfx906            # 应看到 2f6d5bf6b47e

# 静态自检（不需要 GPU、不需要模型）
docker run --rm --entrypoint bash vllm-gfx906:latest -lc \
  'md5sum /usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py;
   gfx906-selfcheck --help 2>&1 | head -5'

# 起服务（需要 GPU + 模型目录挂到 /models）
docker run -d --name vllm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models \
  -v /path/to/logs:/logs \
  vllm-gfx906:latest gfx906-serve /models/<模型目录> --api-key <KEY>
```

> `docker load` 之后就能用。**这是"留下一个可以使用的镜像"的最小充分条件。**

---

## 3. ⚠ 为什么不能只靠本仓库重建（循环依赖）

`Dockerfile.rocm_gfx906` 有两条构建路径：

- **Path B**（`--target rebuild`，权威、离线）：
  `FROM rocm/dev-ubuntu-24.04:7.2.4-complete` → apt 装依赖 → **`COPY dist-packages.tar.gz`**（解压 ~18 G）→ COPY `rccl/` `supplements/` `patches/` `entrypoint/` → 用 `hipcc` 现编 `p2_w4a16_gemv.hip`
- **Path A**（默认 target，秒级）：`FROM vllm-gfx906:rebuilt` → 只重 COPY 几个补丁文件

**Path B 需要 `build-context/`，而 `build-context/` 只能从一个"已经建好的容器"里抽出来：**

```
prepare_build_context.sh [7/7]:
    docker exec <运行中的容器> tar -czf /tmp/dist-packages.tar.gz -C / \
        usr/local/lib/python3.12/dist-packages usr/local/bin
```

⇒ **要有 tar 才能造镜像，要有镜像才能造 tar。**

而且下面这些**必需输入都不在 git 里**（`.gitignore` 明确排除了 `build-context/` 与 `dist-packages.tar.gz`，二进制也放不下）：

| 输入 | 体积 | 在 GitHub？ |
|---|---|---|
| `dist-packages.tar.gz` | 6.8 G | ❌（`.gitignore`） |
| `supplements/`（156 个 Tensile gfx906） | 142 M | ❌（二进制） |
| `rccl/librccl.so.1.0` | 11 M | ❌（二进制） |
| vLLM 补丁源文件 × 5 | 91 K | ✅ **已于 2026-09-18 补齐** |

**但**：上面 4 项**全部可以从镜像里 1:1 再生**（2026-09-18 逐字节验证，见 §7）。所以**镜像 = 唯一的根产物**。

---

## 4. 从 tar 完整重建（改补丁 / 换基座时用）

```bash
# 1) 恢复镜像
docker load -i images/vllm-gfx906-latest-2f6d5bf6.tar

# 2) 起一个"活容器"当素材源（要有 /dev/dri + /dev/kfd 才认得出卡；
#    只是抽文件的话不加也行）
docker run -d --name vllm-gfx906 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --entrypoint bash vllm-gfx906:latest -c 'sleep infinity'

# 3) 生成 build-context（在仓库的 docker/ 目录下跑）
cd docker
FORCE=1 bash prepare_build_context.sh vllm-gfx906
#    产物：build-context/{dist-packages.tar.gz, rccl/, supplements/(156),
#                        patches/, entrypoint/, versions.txt}

# 4) Path B 全量重建（约 20–40 min，18 G 解压是主要耗时）
docker build -f Dockerfile.rocm_gfx906 --target rebuild \
             -t vllm-gfx906:rebuilt build-context/

# 5) Path A 出交付 tag（秒级）
docker build -f Dockerfile.rocm_gfx906 -t vllm-gfx906:latest .

# 6) 自检（不需要 GPU）
docker run --rm --entrypoint bash vllm-gfx906:latest -lc gfx906-selfcheck
```

**要点**
- `FORCE=1` 是必须的，破两个**静默出错**的陷阱：① `prepare_build_context.sh` 见 tar 已存在即跳过重打；② `p05_install.sh` 的「未 COPY 即拒装」门会对 tar 沿线文件误拒。
- 第 2 步的容器名要和 `prepare_build_context.sh` 的参数一致（默认 `vllm-gfx906`）。
- 若想跳过 `dist-packages.tar.gz` 重生（省 ~10 min），把它留在 `docker/build-context/` 里即可（该目录**不要**提交进 git）。

---

## 5. 补丁清单（Dockerfile 会 COPY 进镜像的，全部已在 `docker/patches/`）

| 文件 | md5 | 作用 |
|---|---|---|
| `platforms_rocm.py` | `221a70134a8d22f9d411cc0f7871dc10` | **P0.6**：gfx906 不再被判定为 CDNA（`_ON_CDNA` 子串匹配 `"gfx9" in "gfx906"` 曾使 `supports_fp8()` 错返 True）；并补 fp8 KV 显式拒绝 + 每次启动打 `ROCm FP8 posture` 日志 |
| `layers_utils.py` | `ddb9765fed0fd253…` | **P0.5**：M=1 dense GEMV → ROCm skinny `LLGemm1_kernel`（decode step 53.66 → 33.01 ms，18.6 → 30.3 tok/s） |
| `wna16_exllama.py` | `0a146c02eaf3e696…` | compressed-tensors W4A16 → `ops.gptq_gemm(use_exllama=True)` 桥接 + P2 的 M=1 通路 |
| `linear__init__.py` | `42e243693b55fa2b…` | W4A16 kernel 注册（**此前唯一副本只存在于镜像里**，2026-09-18 抽回仓库） |
| `triton_decode_attention.py` | `a5a0ac3f2a64fd55…` | gfx906 上 `BLOCK=32` |
| `gfx906_gemv_fix.py` / `.pth` | `ef07381791dd48a2` / `722a18dab978dbc5` | rocBLAS `matrix*vector` segfault 规避（靠 `.pth` 自动生效，**不能用 `sitecustomize.py`** —— Ubuntu 自带同名文件会遮蔽） |
| `p2_w4a16_gemv.hip` | `af1535d67012d54e…` | P2：手写 wave64 W4A16 GEMV（split-K=10），**在镜像构建时现编**成 `gfx906/libp2gemv.so` |

**纪律**：Dockerfile 会 COPY 的文件，`prepare_build_context.sh` 必须从**仓库**取（不能从容器 `docker cp`），否则一个未经审阅的副本可能遮蔽已审计的版本。2026-09-18 已把这一组全部改成仓库来源。

---

## 6. 模型权重（不在镜像内）

镜像里**没有任何权重**。恢复后自备模型并挂到 `/models`。

- 冻结时验证过的目标：`Qwen3.8-27B-OBLITERATED-AWQ-W4A16`（Dense AWQ-W4A16、`head_dim=256`、64 层 = 48 GDN + 16 full attn、VLM）
- 冒烟用的最小模型：`Qwen3-0.6B-real`（1.5 G）
- 生产启动参数（27B 档）与 35B MoE 档见仓库 `docs/superpowers/` 与 `host_tools/start-*.sh`
- 只想要"能跑起来"：`gfx906-serve /models/<任意 AWQ-W4A16 目录>`，启动横幅会打印实测的 KV 容量

---

## 7. 2026-09-18 的可再生性验证记录（这些数字是本文的根据）

| 验证项 | 方法 | 结果 |
|---|---|---|
| Tensile gfx906 supplements | 镜像 `torch/lib/rocblas/library` vs 主机 `backup-rocm-tensors` | **156/156 文件名、字节、md5 全一致** |
| 同上，修复后的 [2/7] 回退分支 | 起无 GPU 容器实跑抽取 | **抽到 156 个，与镜像内 md5 `0 不一致`，与主机 md5 `0 不一致`** |
| `librccl.so.1.0` | 镜像 vs 主机 | **md5 `282fe4362505e23ddfda4c3580522f8a` 一致** |
| 3 个补丁源文件 | 从镜像抽出 vs `container_src/` 归档 | `wna16_exllama.py` / `triton_decode_attention.py` **IDENTICAL**；`linear__init__.py` 仓库原本缺失，已补 |
| P0.6 是否在镜像内 | `md5sum` + `grep` | `221a7013…`，`cdna-fix hits: 1`，`fp8-guard hits: 1` |
| `/root/.cache` 是否干净 | 构建期断言 | `cache empty: 0`（交付镜像不夹带编译缓存） |
| 基座可得性 | Dockerfile `FROM` | `rocm/dev-ubuntu-24.04:7.2.4-complete` 公开可 pull |

**当时发现并修掉的缺口**：`prepare_build_context.sh` 的 [2/7] 回退原本指向 `/work/gfx906-supplements` —— 该目录只存在于当年的 dev 容器，**交付镜像里没有**。原逻辑会在回退时静默产出空的 `supplements/`，Path B 于是构建出一个**没有任何 gfx906 kernel 的 rocBLAS**（在 gfx906 上等于不可用）。已改为从 `torch/lib/rocblas/library` 抽 gfx906 文件，并加了「必须恰好 156 个 + 必须含 `TensileLibrary_lazy_gfx906.dat`」的硬断言。

---

## 8. 已知限制（不要重新踩）

- **无 FP8 / 无 MFMA / 无原生 BF16**：gfx906 是 GCN5。`--dtype float16` 是硬要求（BF16 GEMM 慢 1.81×）。fp8 KV 已被平台层显式拒绝。
- **NVFP4 / MXFP4 权重不可用**：所有原生 kernel 都是 CUDA-only，唯一兜底是纯 Triton 模拟（ALU 瓶颈）。
- **`docker commit` 默认会 pause 容器，在本机会永久挂起** —— 必须 `--no-pause`。
- **交付镜像不夹带编译缓存**：构建期有 `test -z "$(ls -A /root/.cache)"` 断言。缓存约 600 MB/种 W4A16 kernel 模式，且 vLLM 的 compile-cache key 不覆盖 kernel 选择 ⇒ 一个模式的缓存会拖垮另一个。
- **`--kv-cache-memory-bytes` 一旦设定就忽略 `GPU_MEM_UTIL` 并跳过显存 profiling**，字节数绑定当时的 free memory ⇒ 换卡/加共租/改 `MAX_LEN` 都要重看日志建议值。
- **绝不同时开 KV offload 与 `int8_per_token_head`**：scale 内联在 cell padding 尾部而 offload 用跨层共享缓冲 ⇒ stride 错位、静默覆盖相邻 K/V ⇒ **不崩不 NaN，只是流利而自信的错答案**。
