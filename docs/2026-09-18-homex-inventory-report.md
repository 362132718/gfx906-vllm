# /home/x 盘点报告（2026-09-18 12:4x，只读，未改动任何文件）

主机 `ssh x@192.168.10.11` ｜ 单盘 `/dev/nvme0n1p2` ext4 915 G ｜ 已用 **225 G**，可用 **644 G**（26%）
扫描方式：4 轮 `find/du/ls/docker inspect`，全部只读。**未移动、未删除、未修改任何文件。**

---

## 0. 结论速览

| 盘面 | 数字 |
|---|---|
| 根分区 | 915 G 总 / **225 G 已用** / 644 G 可用 |
| `/home/x` | **58 G** |
| Docker（镜像 + 构建缓存） | 约 **165 G** ← 真正的占用大头 |
| 其中可直接回收（保守口径） | **约 43 G** |
| 其中可回收（激进口径，含清空构建缓存） | **约 90 G** |

**真正的"乱"不在体积，在 `/home/x` 顶层散落着 49 个文件**（49 个里含 1 个 20.55 G 的 tar，其余是探针 payload / 探针脚本 / 启动器混在同一个平面）。

**但比"整理"更紧急的，是下面 §1 的三件事。**

---

## 1. ⚠ 三件必须先确认的事

### 1.1 所有大模型都不在了 —— llama.cpp 现在 100% 起不来

`/home/x/models` 从上次实测的 **138 G 掉到 1.6 G**，目录 mtime = **今天 12:41**（同一时刻 `.xterminal` 12:43）。全盘（含 `sudo find /`）已无任何 >100 M 的 `.gguf`。

而 `/home/x/llama-swap/` 的 4 个后端脚本**全部硬编码指向已删除的文件**：

```bash
# Qwen3.6-35B-A3B-DSV4Pro-…-Q4-imatrix.sh
exec …/build-rocm724-fix/bin/llama-server \
    -m /home/x/models/Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-LynnStyle-Q4-imatrix.gguf \
    --mmproj /home/x/models/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf \
```

四个 GGUF（上次实测共 85.7 G）+ 两个 mmproj 全都不存在 ⇒ **llama-swap 的 4 个模型一个都拉不起来**。

vLLM 侧的权重也一起没了：`models/vllm/` 现在只剩

| 残留 | 体积 | 状态 |
|---|---|---|
| `Qwen3-0.6B` | 11 M | **只有 tokenizer，无权重**（不完整） |
| `Qwen3-0.6B-real` | 1.5 G | 完整（唯一的完整模型） |
| `MiniCPM-o-4.5-awq` | 168 K | **只剩 `test_assets/`**，权重没了 |

已消失的：`Qwen3.8-27B-OBLITERATED-AWQ-W4A16`、`Qwen3.6-35B-A3B-AWQ-4bit`、4× GGUF、2× mmproj。
HF/ModelScope 缓存里也没有备份（`.cache/huggingface` 仅 44 M）。

**本机无法恢复**：无 Trash（`~/.local/share/Trash` 空）、无 btrfs 快照、无 LVM、无第二块盘。

> ❓ **这次删除是你自己做的吗？** 你上一条消息说"后面只使用 llama 就可以了"，但同时 llama 的 GGUF 也一起没了 —— 如果是有意的（打算重新下载），没问题；如果不是，这是本次唯一的事故点，需要立刻停下所有操作。

### 1.2 昨天那次镜像重建**成功了**（这条是好消息）

`/home/x/logs/image_build_20260917.log` 尾部：

```
#16 naming to docker.io/library/vllm-gfx906:latest done
=== [6/6] 验收：新镜像内 rocm.py 应与 payload 逐字节一致 ===
221a70134a8d22f9d411cc0f7871dc10  …/vllm/platforms/rocm.py
cdna-fix hits: 1
fp8-guard hits: 1
cache empty: 0
```

⇒ `vllm-gfx906:latest` = `2f6d5bf6b47e`，P0.6 补丁（`_ON_CDNA` + fp8 KV 显式拒绝）**已逐字节固化进镜像**，`/root/.cache` 干净。

⚠ **但当前容器挂在旧镜像上**：`docker inspect vllm` → `c31a8b65273c` = `vllm-gfx906:pre-cdnafix-20260917`。也就是说新镜像**还没做过运行时验收**（容器内的 P0.6 是当初直接打进容器层的那份，不是镜像里的）。

### 1.3 容器里现在什么都没跑

```
容器 vllm:  Up 2 hours ｜ 进程只有 sleep infinity ｜ 0.0.0.0:8000 只是 docker-proxy
/dev/shm:   空（offload 的 mmap 已清）  ｜  /home/x/logs/kv-tier: 1.5 G 遗留缓存
```

---

## 2. `/home/x` 逐项占用与性质

### 2.1 大项（>1 M）

| 项 | 体积 | 性质 | 处置建议 |
|---|---|---|---|
| `comfyui/` | **18 G** | 图像生成（models 占 18 G） | **保留**（与两个 LLM 项目无关） |
| `vllm-gfx906-c31a8b65.tar` | **20.6 G** | `docker save` 归档 | **可回收**（同一镜像本机已有 tag） |
| `vllm-work/` | 8.2 G | vLLM 构建链 + 12 个阶段实验目录 | 归档；其中 `docker/build-context/` 7.0 G 可回收 |
| `vllm-cache/` | 3.2 G | torch AOT 编译缓存 | **可回收**（可重建） |
| `models/` | 1.6 G | 容器 bind mount ⇒ `/models` | 保留目录结构 |
| `logs/` | 1.5 G | 容器 bind mount ⇒ `/logs` | `kv-tier/` 1.5 G 可回收 |
| `.rustup/` | 1.4 G | Rust 工具链 | **可回收**（两个项目都不用 Rust） |
| `llamacpp-gfx906-slot-save/` | 1.1 G | llama.cpp 源码 + **3 个 build 目录** | 保留最新 `build-rocm724-fix`，另两个可回收 |
| `archive_old_work/` | 941 M | 8 月的归档（已归档过一轮） | 保留；内含 324 M triton 缓存可回收 |
| `.lmstudio/` | 504 M | LM Studio 客户端（含 node/deno 运行时） | 视是否还用 LM Studio |
| `vllm-src/` | 288 M | vLLM 源码副本 | 归档 |
| `下载/` | 263 M | ROCm 6.4.4 旧包 + deb | **可回收**（当前栈是 7.2.4） |
| `snap/` | 188 M | snap 应用（firefox/mesa/gnome…） | 无头服务器基本不用 |
| `.cache/` | 157 M | 其中 pip 96 M / HF 44 M | pip 可回收 |
| `backup-rocm-tensors/` | 142 M | ❓来历不明 | **需你确认** |
| `.local/` | 98 M | 含 lib 96 M | 保留 |
| `.cargo/` | 75 M | Rust crates | 可回收（随 `.rustup`） |
| `bin/` | 23 M | `llama-swap` 二进制 | **保留** |
| `rccl-gfx906-2.27.7-rocm7.2.4/` | 11 M | RCCL 构建产物 | 归档 |
| `.config/` | 11 M | | 保留 |
| 其余：`reinstinct` 3.6 M / `model_benchmark` 556 K / `llama-swap` 24 K / `llm-cache` 32 K | <5 M | | 见 §4 |

### 2.2 空目录与垃圾文件

| 项 | 数量 | 体积 |
|---|---|---|
| 空目录 | **195 个** | 0 |
| `__pycache__` 目录 | 1 335 个 | — |
| `*.pyc` | 7 728 个 | 93 M |
| 7 个桌面标准目录（公共/模板/视频/图片/文档/音乐/桌面）+ `image-export` + `ccache` | 9 个 | 全空（0） |
| `*.orig/*.bak/*.rej/*~` | 101 个 | 全在 `.cargo/registry/` 内（**Rust 源码自带的，不是你的**）+ `archive_old_work/*.bak-*` 3 个 |
| 日志文件 | 274 个 | 0.03 G（**体积可忽略**） |

> 注：`*.orig` 的 101 个里 101 个都在第三方 crate 源码里，**不要动**。

### 2.3 Docker（真正的大头）

| 项 | 体积 | 说明 |
|---|---|---|
| `vllm-gfx906:latest` | 2f6d5bf6 | 23 h 前 | **新镜像，含 P0.6** ✅ |
| `vllm-gfx906:rebuilt` | 204e414f | 23 h 前 | 同批产物，**tag 冗余** |
| `vllm-gfx906:pre-cdnafix-20260917` | c31a8b65 | 6 天前 | 回滚锚点，**当前容器在用** |
| `rocm/dev-ubuntu-24.04:7.2.4-complete` | 92f309c5 | 3 月前 | 29.2 G，重建镜像的基座 |
| 容器 `vllm` | 270 MB | 只有 `sleep infinity` |
| **Build Cache** | **55.5 G** | `docker system df` 报可回收 7.5 G |
| 镜像页报总量 | 110.4 G | 三层 `vllm-gfx906` 大量共享层 |

---

## 3. 分层处置建议

### A 档：必须保留（删了就回不来 / 后续还要用）

```
/home/x/bin/llama-swap                                    ← llama-swap 二进制
/home/x/llama-swap/                                       ← 配置 + 4 个后端脚本
/home/x/llamacpp-gfx906-slot-save/                        ← llama.cpp 源码 + build-rocm724-fix（当前后端用的就是它）
/home/x/start-llama-swap.sh, stop-llama-swap.sh
/home/x/models/ , /home/x/logs/                           ← 容器 bind mount，目录本身不能删
/home/x/models/vllm/Qwen3-0.6B-real/                      ← 唯一完整模型（冒烟用）
/home/x/comfyui/                                          ← 18 G，你的另一条线
/etc/cron.d/amdgpu-fancontrol, amdgpu-powercap, amdgpu-pp-fan-patch   ← 风扇/功耗，单卡也还要
/home/x/.ssh/ , .gnupg/ , .config/ (不含缓存) , .local/ , .pki/
```

### B 档：建议归档到一个目录（不删，只是从顶层挪走）

顶层现在有 **49 个散落文件**，按用途可分 4 组：

| 组 | 文件模式 | 个数 | 建议去向 |
|---|---|---|---|
| 启动 / 运维 | `start-vllm.sh` `stop-vllm.sh` `start-27b.sh` `start-35b.sh` `build_image_cdnafix.sh` | 5 | `vllm-archive/bin/` |
| 探针脚本 | `longctx_probe*.sh` `prefix_hit_*.sh` `kv-reuse-probe.sh` `probe_*_inner.sh` | 8 | `vllm-archive/probes/` |
| 请求 payload | `lc_*.json` `lc2_*.json` `p_*.json` `p_long_a*.json` `pv_*.json` `ph_*.json` `mm_*.json` | 24 | `vllm-archive/payloads/` |
| 补丁 / 元数据 | `patch_rocm_cdna.py` `.vllm-api-key` | 2 | 补丁入 `vllm-archive/patches/`；api-key 留原地（会被容器脚本读） |
| 保留在顶层 | `start-llama-swap.sh` `stop-llama-swap.sh` | 2 | **不动** |

目录级归档：

```
vllm-work/            → vllm-archive/vllm-work/
vllm-cache/           → 可回收（见 C 档）或 vllm-archive/vllm-cache/
vllm-src/             → vllm-archive/vllm-src/
rccl-gfx906-2.27.7-rocm7.2.4/ → vllm-archive/
archive_old_work/     → 保持原位（它本身就是归档）
```

**建议的最终顶层形态**：

```
/home/x/
├── models/            (bind mount → /models)
├── logs/              (bind mount → /logs)
├── llama-swap/        ← llama.cpp 后端脚本
├── llamacpp-gfx906-slot-save/   ← llama.cpp 源码+build
├── bin/               ← llama-swap 二进制
├── start-llama-swap.sh  stop-llama-swap.sh
├── comfyui/
├── vllm-archive/      ← vLLM 全套（含 bin/ probes/ payloads/ patches/ + 三个目录）
├── archive_old_work/
└── (常规目录)
```

### C 档：可安全回收（给出精确路径，等你逐项确认）

| # | 路径 | 释放 | 为什么安全 |
|---|---|---|---|
| 1 | `/home/x/vllm-gfx906-c31a8b65.tar` | **20.6 G** | `docker save` 归档；同镜像本机已以 tag `pre-cdnafix-20260917` 存在 |
| 2 | `/home/x/vllm-work/docker/build-context/dist-packages.tar.gz` | **6.8 G** | `prepare_build_context.sh` 可重打（代价 ~30–60 min） |
| 3 | `/home/x/vllm-cache/vllm-w4a16-exllama/` | **3.2 G** | torch AOT 缓存，可重建（代价：下次冷启回到 ~11 min） |
| 4 | `/home/x/logs/kv-tier/` | **1.5 G** | offload fs 层缓存，纯缓存；腿 0 实验的产物，结论已入库 |
| 5 | `/home/x/.rustup/` + `/home/x/.cargo/` | **1.5 G** | Rust 工具链，vLLM/llama 均不用 |
| 6 | `docker builder prune`（构建缓存） | **7.5–55.5 G** | 视 `--all` 与否；重建镜像会更慢 |
| 7 | `/home/x/.lmstudio/` | 504 M | LM Studio 客户端；若已弃用 |
| 8 | `/home/x/archive_old_work/vllm028_triton_cache/` | 324 M | triton 缓存 |
| 9 | `/home/x/下载/rocblas-6.4.4-1-x86_64.pkg.tar.zst` | 256 M | ROCm **6.4.4** 的包，当前栈 7.2.4 |
| 10 | `/home/x/snap/`（+ `/var/lib/snapd` 的 loop 挂载） | 188 M | 无头服务器（firefox/mesa/gnome snap） |
| 11 | `/home/x/backup-rocm-tensors/` | **142 M** | ❓来历不明，**需先确认** |
| 12 | `/home/x/models/vllm-cache/triton/` | 132 M | triton 缓存 |
| 13 | `/home/x/.cache/pip/` | 96 M | pip 包缓存 |
| 14 | 全部 `*.pyc`（7 728 个） | 93 M | 可重建 |
| 15 | 195 个空目录 | 0（但清爽） | 纯空 |
| 16 | `docker rmi vllm-gfx906:rebuilt` | 0（共享层） | 只是去 tag |
| 17 | `archive_old_work/vllm028_src/vllm-0.28.0.tar.gz` | 40 M | 与解压目录重复 |
| | **保守小计（不含 #6）** | **≈ 35 G** | |
| | **含 `docker builder prune -a`** | **≈ 83 G** | |

---

## 4. 单卡 + llama.cpp 的连带影响（供你决策，不是待办）

如果确定"只用 1 张卡 + llama.cpp"，除了上面的目录整理，还有两处**配置层面**要改：

1. `llama-swap/*.sh` 里的 `-sm layer --tensor-split 44/56` 是**双卡**参数，单卡要删掉（否则 split 无意义）。
2. 后端脚本引用的 GGUF 已不存在 ⇒ 无论删不删文件，**llama-swap 目前都不可用**，需要先重新下载模型。

`/etc/cron.d/` 下三个 AMD 脚本（`amdgpu-fancontrol`、`amdgpu-powercap`、`amdgpu-pp-fan-patch`）**不要动** —— 单卡同样需要它们维持 130 W 运行点与风扇曲线；注意其中两个脚本有竞态历史（后写者需等前写者）。

---

## 5. 我没做的事

- 没有删除、移动、重命名**任何文件**
- 没有停容器、没有动 Docker
- 没有清理空目录（195 个）
- 没有动 `.cargo/registry` 里的 101 个 `*.orig`（第三方源码自带）
- **没有动那个 20.6 G 的 tar**（虽然它是最值钱的一块空间）

## 6. 需要你回答的两个问题

1. **§1.1 的模型删除是你做的吗？**（12:41，138 G → 1.6 G）如果是，我按"重新下载"来规划；如果不是，请立刻告知，我暂停一切操作。
2. **C 档 17 项，按"保守口径（≈35 G）"还是"含清空构建缓存（≈83 G）"执行？** 我会按批（每批 ≤10 项）逐项列出路径请你确认后再动。

---
*报告生成：2026-09-18 ｜ 扫描轮次：4×只读脚本（`host_tools/audit_homex_1..5.sh`）*
