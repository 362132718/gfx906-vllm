# W4A16 位级映射（L0 实测钉死，非推导）

状态：**G1 PASS（2026-09-08）**。证据 `logs/p1/p1_l0_mapping.log`；测试载体 `p1_l0_mapping.py`
（L0a 纯 python 逐字比对 + L0b one-hot `x = e_k` 让现成 `gptq_gemm_out` 直读每个 `(k,n)`）。

## 一个 int32 字里的 8 个 k 位置

`qweight[kw, n]` 的第 i 个 code（i = 0..7，对应 k = kw*8 + i）位于位
`(i//2)*4 + (i%2)*16`，即 k 顺序 → bit 位置：

| i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| bit | 0-3 | 16-19 | 4-7 | 20-23 | 8-11 | 24-27 | 12-15 | 28-31 |

=> 提取"连续 4 个 k"（i=0..3）**不能**用顺序掩码移位，必须按上表打乱；
   这正是 `v_dot2_f32_f16` 的 pair 边界：(i0,i1) 与 (i2,i3) 各自由
   `((w >> 4p) & 0x000F000F) + 0x64006400` 一次取出（p = 0..3 → (c0,c1)(c2,c3)(c4,c5)(c6,c7)，
   本地逐位核验过与 `dst_bits` 完全吻合，评审亦独立复核）。

## 三个张量的最终布局（内核入口所见）

- `qweight`: **[K/8, N] int32**，如上置乱。
- `scales`: **[K/128, N] fp16**，contiguous。
- `qzeros`: **[K/128, N/8] int32**，沿 N 每 8 个输出打包一 word、LSB-first（word (g, n8) 的
  第 j 个 nibble 是列 8*n8+j）；**GPTQv2 编码：存储 nibble 就是真 zp**（不减 1）。
- `g_idx`: **空张量**（无 act-order；桥接把它当 act-order *置换表* 用，不是 group index）。

检查点侧（桥接的输入）形态，供造数据/回归用：`weight_packed [N, K//8]`（沿 K、LSB-first）、
`weight_scale [N, K//G]`、`weight_zero_point [N//8, K//G]`（沿 N 打包）。
⚠️ `weight_zero_point` 必须按检查点形态喂给桥接 —— 喂 `[N, K//G]` 会在桥接自己的
`permute/reshape` 里炸（复现：`shape '[16, 2]' is invalid for input of size 256`）。

## 校验方式与结论

- L0a：合成 codes（`(k*7 + n*13) % 16`，保证每个 (k,n) 可辨识、每个 code 变化值可辨识）过桥接
  `process_weights_after_loading`，与独立实现的 `expected_qweight` 逐字比对。
  **5 个真实形状 + 1 个小形状全部 `mismatched_words=0`**（累计 22,602,240 个 int32 字）。
- L0b：`x = e_k`（one-hot）消掉所有抵消项，`gptq_gemm_out` 的输出逐 (k,n) 直读；
  与 fp32 解析参考 `(q − zp)·scale` 比，**5 个真实形状 `max_rel = 4.880e-04`** ——
  恰为 1/2048 = 一个 fp16 输出量子（内核输出 fp16、参考 fp32 的舍入下限），无任何映射类误差。
- 结论：**设计文档 §映射 的位规则与桥接实现一致；P2 内核可按本文档的位表直接实现提取。**

## harness 复用的三个坑（下次造合成层别再踩）

1. 桥接的 `_transform_param` → `permute_param_layout_` 要求参数**带 `input_dim`/`output_dim`
   （weight 还要 `packed_dim`）属性**，裸 `torch.nn.Parameter` 直接 AssertionError。
2. `BasevLLMParameter` 在 harness 里也不可用：其 `__init__` 调 `get_tensor_model_parallel_rank()`，
   没有 vLLM 分布式状态就炸。⇒ 正解是继承 `torch.nn.Parameter` 直接加这三个属性
   （`weight_loader` 只在加载期用，桥接路径不碰）。
3. `permute_param_layout_` 之后 `_transform_param` 会把参数**换回普通 `torch.nn.Parameter`**
   （Dynamo 兼容），所以"第一个被 transform 的参数"之后其余参数仍须自带属性——按
   qweight → scales → zeros 的顺序各自独立带属性即可。
