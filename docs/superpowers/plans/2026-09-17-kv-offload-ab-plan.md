# KV 多级 offload A/B 计划（2026-09-17 联网 + 源码核实后定稿）

> 目标问题：**用 CPU 内存 + SSD 多级存储减少长前缀的 prefill 重算，到底值不值。**
> 前置条件已就绪：前缀粒度已修（`PREFIX_MATCH_UNIT=16`，命中率 0%→99.6~100%）、
> offload × per-token-head 量化的硬护栏已加（`start-vllm.sh` 拒绝同开）。
> 本文档只写**计划**，尚未执行。

---

## 一、核实结论（先定事实，再谈实验）

### 1.1 机制：官方文档口径（vLLM KV Offloading Usage Guide）

- `OffloadingConnector` 做的事是**扩展前缀缓存**：把**已产生的** KV 块卸到更大更慢的层（CPU 宿主内存 + 可选次级层），
  **命中时按需 promote 回 GPU**。GPU↔CPU 走 DMA（`cudaMemcpyAsync`）**与计算异步重叠**。
- 官方明确：**currently supports CUDA, ROCm, and XPU only** ⇒ **ROCm 在支持列表内**（我们不是"蹭"）。
- 两种 spec（由 `kv_connector_extra_config.spec_name` 选）：
  | spec | 形态 | 承载体 |
  | --- | --- | --- |
  | `CPUOffloadingSpec`（默认） | 单 CPU 层 | **pinned host memory** |
  | `TieringOffloadingSpec` | CPU 主层 + 若干次级层 | CPU 主层 = **共享 mmap 区**；次级层不支持直连 GPU，所有 GPU↔次级都要经 CPU 主层中转 |
- 关键配置项（官方参考表）：
  - `cpu_bytes_to_use`（必填）= **所有 worker 的总量**，不是每 worker。
  - `offload_prompt_only` 默认 **true** ⇒ 只卸 prompt（prefill）块，decode 块跳过。
  - `blocks_per_chunk` 默认 1；`block_size`（按 token 计）与它**互斥**，且官方注明
    `blocks_per_chunk` 是 **"models whose KV cache groups have different block sizes"** 的替代品
    ⇒ **本模型（attention + Mamba 两个 group）应当用默认的 `blocks_per_chunk`，不要用 `block_size`**。
  - `eviction_policy`：`lru` / `arc`；`store_threshold` 仅单层可用（**≥2 会被 Tiering 拒绝**）。

### 1.2 ⚠ 核实掉一个"网传结论"：hybrid 模型能不能用 offload

网上（多篇二手分析，含 skills.lc / ForceInjection 的模块解析）流传的说法是
"OffloadingConnector 不支持 HMA ⇒ 混合模型必须 `--disable-hybrid-kv-cache-manager`，否则启动报
`Connector OffloadingConnector does not support HMA but HMA is enabled`"。

**这个说法对我们不成立 —— 镜像源码实测（vLLM 0.28.0）：**

```
config/vllm.py:1750-1762
    if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
        # Auto-disable HMA only when the connector config does not support it.
        if self.kv_transfer_config is not None:
            if not KVConnectorFactory.supports_hma_config(self.kv_transfer_config):
                need_disable_hybrid_kv_cache_manager = True
```

即 **0.28 只在连接器确实不支持 HMA 时才关 HMA**。而
`kv_connector/v1/offloading_connector.py:49` = `class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)`
⇒ **HMA 保留，混合模型不会被迫走 `unify_hybrid_kv_cache_specs` 的合并路径**（那条路会把 Mamba/full-attn
group 强行统一 ⇒ 每请求 KV 需求暴涨 ⇒ 启动 OOM）。

对照：vLLM issue **#41830** 描述的正是"`--kv-transfer-config` 无条件关 HMA"这个 bug，
其提案就是上面这段 `supports_hma_config` 门 ⇒ **代码已在 0.28 落地**。二手文章描述的是修复前状态。
`factory.py:56-62` 的 raise 只在 `hma_enabled and not supports_hma(...)` 时触发，我们不会命中。

> ⚠ 结论修正记录：此前记的"HMA 硬门 ⇒ 不能 disable"方向正确，但**理由是过期的**；
> 正确理由是"0.28 已按连接器能力决定是否关，且本连接器支持 HMA"。

### 1.3 CPU 层落在哪、能不能活过重启（源码实测）

`v1/kv_offload/cpu/shared_offload_region.py`：

- 路径写死 **`/dev/shm/vllm_offload_{engine_id}.mmap`**，`os.open(..., O_CREAT|O_EXCL|O_RDWR, 0o600)`；
- 已是存在文件时打 **`Opened existing mmap file %s`** 并复用 ⇒ **CPU 层可活过 vLLM 进程重启**
  （`/dev/shm` 是 tmpfs，随**容器**生命周期，不随进程）；
- 非创建者会 `_wait_for_file_size(expected_size, timeout=30s)`，超时抛 `TimeoutError`
  ⇒ **残留的、尺寸不符的 mmap 文件会让下次启动失败**（见 3.3 对策）；
- `TieringOffloadingSpec` 继承 `CPUOffloadingSpec`，共用这个 `SharedOffloadRegion`。

**fs 次级层**：`file_mapper.py:118` 的文件名是
`<base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin`，
并有 `get_config_file_path()` 落一份参数文件（`model_name / tokens_per_hash / blocks_per_file /
tp/pp/pcp/dcp / rank / dtype / kv_cache_groups`），`_compute_base_path(root_dir, fields)` 由参数派生目录
⇒ **按 hash 定位、跨进程/跨容器可复用**，且**换配置会落到不同 base_path**（不会读错数据）。

### 1.4 ⚠ 核实到一个**真实且落在我们路径上的**风险：tiering 的 mmap 启动失败

此前我把它记为"查无权威出处"。现在有出处了（CSDN《vllm start with TieringOffloadingSpec mmap_obj error》），
且与 1.3 的源码一致 —— **只影响 tiering 路径，单层 CPUOffloadingSpec 不受影响**：

| 报错 | 触发条件 |
| --- | --- |
| `ValueError: mmap.mmap(-1, 0): [Errno 22] Invalid argument`（`_map_tier`） | 尺寸算成 0 |
| `OSError: [Errno 13] Permission denied: '.../kv_offload.bin'` | root_dir 权限 |
| `OSError: [Errno 19] Operation not supported: mmap on this filesystem` | 文件系统不支持 mmap |

**我们的风险面**：`cpu_bytes_to_use=16 GiB` 分给 2 个 worker = 每 worker 8 GiB（非 0，OK）；
`/dev/shm` 实测可用 24 GiB（够）；fs 的 `root_dir=/logs/kv-tier` 落在宿主 ext4 的 bind mount 上（支持 mmap）。
⇒ 大概率能过，但必须在**腿 0** 验证，且用 3.3 的对策排除残留文件。

### 1.5 可观测指标（A/B 的仪表盘）

CPU 层（已从源码读出常量）：
`vllm:kv_offload_stores_skipped`、`vllm:kv_offload_cpu_cache_usage_perc`、
`vllm:kv_offload_cpu_allocation_size`、`vllm:kv_offload_cpu_cache_write_usage_perc`、
`vllm:kv_offload_cpu_cache_read_usage_perc`
tiering 层：指标名是**动态拼的**（`tiering/metrics.py`），必须先起一次再从 `/metrics` 抓全名。
原有对照指标：`vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total`（现成探针已在用）。

---

## 二、待验证假设

- **H1（能不能起）**：`TieringOffloadingSpec` + fs 次级层能在本模型（hybrid + 2×16 GB + TP2）上正常启动，
  不 OOM、不撞 HMA、不撞 mmap。→ 腿 0。
- **H2（有没有效）**：**进程重启后**再发同一条长前缀，能从 CPU/fs 层 promote 回来，
  prefill 耗时显著低于重算。→ 腿 A vs 腿 B（主问题）。
- **H3（值不值）**：省下的 prefill 时间是否值得付出的代价（放弃 int8 KV ⇒ 上下文从 131072 降到 ~114688；
  CPU 层占 16 GiB 内存；fs 层只增不减要盯盘）。（在 H2 成立后由数据判定）

**为什么 A/B 的对照点选"重启"而不是"同进程内连发两次"**：
同进程内，GPU 内前缀缓存已经把命中率打到 99.6~100%（实测 51×），offload 根本轮不到出场 ——
那样测不出任何差异。offload 的**唯一增量价值**就在"GPU 内已不存在（进程重启 / 被淘汰）"的场景。

---

## 三、A/B 设计

### 3.1 三条腿

| 腿 | 配置 | 目的 |
| --- | --- | --- |
| **0 · 冒烟** | `KV_OFFLOAD_GB=16 KV_TIER_DISK_DIR=/logs/kv-tier KV_CACHE_DTYPE=none KV_CACHE_MEM_BYTES=none MAX_LEN=16384`（先小 max_len，快启动） | 只验 H1：能不能起、VMM 层指标是否出现。**不进 A/B 数据** |
| **A · 对照** | `KV_CACHE_DTYPE=none KV_CACHE_MEM_BYTES=none PREFIX_MATCH_UNIT=16 MAX_LEN=114688`，**无 offload** | 基线：重启后同一长前缀要多久（应≈冷跑） |
| **B · 实验** | 同 A，**加** `KV_OFFLOAD_GB=16 KV_TIER_DISK_DIR=/logs/kv-tier` | 重启后同一长前缀是否变快 |

⚠ **A 与 B 必须同 max_len / 同 gmu / 同 capture / 同模型**，否则不可比。
⚠ **A 与 B 共用 `VLLM_CACHE_ROOT`**（`serve.sh` 按 W4A16 选择路径），torch.compile 产物可沿用 ⇒ 两边都按 427 s 量级预期。

### 3.2 每腿的动作序列（关键在"重启"这一步）

```
1. 清场：停服务；删残留 /dev/shm/vllm_offload_*.mmap；清 /logs/kv-tier
2. 起服务（该腿配置），等就绪
3. 冷跑：发长前缀 P（~50K tokens，带唯一 nonce，max_tokens=256，读 usage.prompt_tokens 核实长度）
   → 记 T_cold、答案、/metrics 快照（offload 各计数器增量）
4. 停服务（进程消失；/dev/shm 与 fs 目录保留）
5. 重启同配置服务，等就绪
6. 热跑：发**同一条** P（同 nonce 会引入"同进程已缓存"的假阳性吗？不会——新进程，GPU 内前缀缓存已空；
   但要保证 prompt 与前一次**逐字节相同**，所以 nonce 必须复用同一字符串，长度用同一份文件）
   → 记 T_restart、答案、/metrics 快照
7. 判定：ΔT = T_cold − T_restart
```

> ⚠ 这里的 nonce 语义与既有探针**相反**：既有探针用"每请求唯一 nonce"来**排除**前缀缓存命中；
> 本 A/B 要的正是命中，所以**同一份 prompt 文件复用两次**，靠"换进程"来保证 GPU 侧是冷的。

### 3.3 预置对策（已知风险 → 动作）

| 风险 | 对策 |
| --- | --- |
| 残留 mmap 尺寸不符 ⇒ 30 s 超时启动失败 | 每腿开始前 `rm -f /dev/shm/vllm_offload_*.mmap`；且**两条腿用同一个 `cpu_bytes_to_use`** |
| mmap 报 `Errno 22 / 13 / 19` | 先退到 **单层 `CPUOffloadingSpec`**（pinned memory，无共享 mmap 区）验证"offload 本身"；单层**不跨重启**，故只作诊断，不作结论 |
| `/dev/shm` 不够 | `cpu_bytes_to_use` ≤ 16 GiB（实测 /dev/shm 可用 24 GiB），失败就降到 8 GiB（此时每 worker 4 GiB） |
| fs 层写满 | 腿 B 前后 `du -sh /logs/kv-tier` + 文件数；跑完清理 |
| 启动 OOM（HMA 未保留的老毛病） | 看日志有无 `Turning off hybrid kv cache manager`；有则说明连接器没被判为 SupportsHMA ⇒ **立刻停，回到 §1.2 复核** |
| 前缀长度不足以致命中粒度 | P 取 ~50K tokens（attention block 1552 的整数倍附近，且远大于 1552） |
| 结论被"重算变快"污染 | 两腿都用**同一份** prompt 文件；且 A 腿本身也要重启后重发一次，作为"重算有多慢"的对照 |

### 3.4 判据（事先定，避免事后解释）

- **H1 通过**：腿 0 就绪，且日志出现 `Opened existing mmap file`（第 2 次起）或 `/metrics` 出现
  `vllm:kv_offload_*`。
- **H2 通过**（主判据，需同时满足）：
  1. `T_restart(B) < 0.5 × T_cold(A)` —— 即重启后的长前缀复用至少带来 **2×** 收益；且
  2. `T_restart(B) < 0.5 × T_restart(A)` —— 与"同样重启、但没开 offload"直接比；且
  3. 腿 B 的 `/metrics` 显示 offload **读命中/load 传输计数 > 0**（不是"只是巧合变快"）；且
  4. 答案逐字正确（针检索，cold 与 warm 一致）。
- **H2 失败**：以上任一条不满足 ⇒ 结论写"不值得"，并记录是"没命中"还是"命中了但不快"。
- **H3**：若 H2 通过，再用"放弃 int8、上下文 −12%、内存 −16 GiB、盘只增不减"三项代价做取舍陈述。

---

## 四、成本与顺序

1. **腿 0 冒烟**：1 次启动（~427 s 量级）+ 一次短请求。**先做这个**，因为它同时验 §1.2 的 HMA 结论与 §1.4 的 mmap 风险。
2. **腿 A**：2 次启动 + 1 次 ~50K 冷跑（按 O(n^1.6) 外推，50K 约 **7–8 min**/次）+ 1 次重启后重发。
3. **腿 B**：同 A 的启动/请求次数，另加 fs 层落盘时间。
4. 合计约 **4 次启动 + 4 次长 prefill**，数小时量级，且**全程占用 GPU**（做完才能恢复 27B 生产档）。
5. 可选 **腿 B2**：`docker rm` 重建容器后再发 P ⇒ 只有 fs 层能救，验证"最持久的那个层"。成本最高，H2 通过后再决定。

---

## 五、我不会在计划里预设的结论

- 不预设 offload 一定正向：官方文档只说"up to 16x faster for long prompts"（那是别人的长 prompt 场景），
  我们的 prefill 单价是 O(n^1.6)，而 offload 读回也要付一次 PCIe + shm/disk 的搬运。
  **需要实测的是"搬回来"是否真的快于"重算"。**
- 不预设 fs 层一定被命中：跨重启"文件还在"不等于"索引还会找到它"，腿 B 的 /metrics 命中计数是唯一证据。
