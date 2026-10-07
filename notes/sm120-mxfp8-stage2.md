# SM120 MXFP8 阶段二记录：线性注意力 TE 化、量化范围对齐官方 FP8、训练侧 MXFP8 开关

环境：8× RTX PRO 6000 Blackwell (sm120)，torch 2.11+cu129，TE 2.16.1，Megatron-core 0.16.0rc0，SGLang 0.5.15.post1（+ slime docker 补丁 + sglang-sm120-mxfp8.patch）。

## 问题记录

### P1 TELinear(parallel_mode="duplicated") 会让线性注意力梯度在 TP 内被重复求和
- 现象（读代码发现）：Megatron `TELinear` 在 duplicated 模式下给参数设置 `param.sequence_parallel = config.sequence_parallel`；`finalize_model_grads._allreduce_non_tensor_model_parallel_grads` 会对这类参数在 TP 组内做 SUM all-reduce。
- 原因：Qwen3.5 的 `HuggingfaceAttention` 在 SP 下先把序列 all-gather 成完整序列，GatedDeltaNet 在每个 TP rank 上对完整序列做重复计算（backward 用 `tensor_parallel_output_grad=False`，只切分不归约），各 rank 的权重梯度本来就相同。再按 SP 参数做 SUM 会把梯度放大 TP 倍。原来的 `nn.Linear` 参数没有这个属性，所以没有这个问题。
- 解决：`_duplicated_te_linear` 构造后把三个投影参数的 `sequence_parallel` 置为 False，保持与原 `nn.Linear` 相同的梯度归约语义。

### P2 `tests/test_qwen3_linear_attention_cu_seqlens.py` 在 HEAD 上就全部失败（与本次改动无关）
- 现象：4 个用例都报 `ValueError: invalid literal for int() with base 10: 'cpu'`。
- 原因：测试把 `torch.cuda.current_device` monkeypatch 成返回 `"cpu"`，而 `slime.utils.accelerator` 会把它当整数解析。未改动的 HEAD 上同样 4 个全部失败。
- 处理：未修复（超出范围）。为插件改动补了 TELinear stub 与 `mcore_config` 参数；去掉该 monkeypatch 的临时副本上 4 个用例全部通过。

### P3 裁剪模型上 MXFP8 与 BF16 差异很大，需要区分“量化本身”与“实现错误”
- 现象：4 层 Qwen3.5（3 线性注意力 + 1 全注意力，16 专家 top-4，随机初始化）上，MXFP8 相对 BF16 的第 0 层输出误差 0.109，logits 0.31，各组梯度余弦 0.73～0.95。
- 原因：随机初始化的路由分数近乎均匀，MXFP8 的量化噪声就足以翻转 top-k（第 0 层 22% 的 token 路由不同，第 3 层 61%），差异逐层放大。
- 判别方法：新增“仿真模式”——BF16 计算，但在每个 GEMM 前按 TE 的量化方向（TN/NN/NT 各自的累加维度）对两个操作数做 MXFP8 伪量化（`tests/_mxfp8_emulation.py`，GroupedLinear 每个专家单独补零到 32 的倍数，与 TE 的 Fp8Padding 一致）。
- 结论：仿真 vs BF16 与 MXFP8 实跑 vs BF16 的各项数字几乎相同；MXFP8 实跑 vs 仿真在第 0 层只差 0.0059 且路由完全一致，之后随少量路由翻转（第 1 层 8%）才放大。BF16 重跑差异为 0。所以整模型 MXFP8 前后向与理想 MXFP8 语义一致，差异来自 MXFP8 量化本身而非实现错误。

### P4 旧插件保存的 torch_dist checkpoint 无法被新插件加载
- 现象：用改动前插件（`nn.Linear`）保存裁剪模型的 torch_dist checkpoint，改动后插件加载时在 `dist_checkpointing/strategies/torch.py:_replace_sharded_keys_with_state_dict_keys` 报 `TypeError: object of type '_io.BytesIO' has no len()`。
- 原因：TELinear 的 state dict 多了 `_extra_state`（Megatron 将其作为 ShardedObject 请求）；旧 checkpoint 里没有这个条目，Megatron 在 `allow_partial_load=True` 下对缺失的 ShardedObject 返回未包装的 `BytesIO`，随后崩溃。用户已有的 Qwen3.5 torch_dist checkpoint（含训练中途保存、待续训的）都会受影响。
- 解决：`_duplicated_te_linear` 给三个投影注册 `register_state_dict_post_hook`（保存时去掉 `_extra_state`）和 `register_load_state_dict_pre_hook`（加载时缺省填 None，TE 的 `set_extra_state(None)` 直接返回）。这样 checkpoint 布局与原 `nn.Linear` 完全相同。代价：这三个层不保存 delayed scaling 的 amax 历史；slime 用的 MXFP8/blockwise/current scaling recipe 不使用这些元数据。
- 验证（`mxfp8_stage2_work/ckpt_compat.{py,sh}`）：旧插件保存→新插件加载、新插件保存→新插件加载、新插件保存→旧插件加载，三种组合 186 个参数全部逐位一致；两种插件保存的 checkpoint 都是 212 个键，线性注意力键集合相同。

## 实验记录

### E1 单卡裁剪模型（`mxfp8_stage2_work/model_check.{py,sh}`，结果 `model_check.json`）
- 配置：`tiny_qwen35/config.json`（num_hidden_layers=4，num_experts=16，top-4），Megatron 随机初始化（seed 1234），DDP 包装（gradient_accumulation_fusion 默认开启 → TE 走 main_grad 累加），3 条序列共 1024 token（THD 打包），交叉熵损失。
- 同一组权重依次跑 BF16、MXFP8（`config.fp8="e4m3"`, recipe mxfp8）、仿真、BF16 重跑。
- GEMM 计数：MXFP8 模式下每种布局（TN/NN/NT）各有 Linear 18 次、LayerNormLinear 1 次、GroupedLinear 8 次，全部为 MXFP8 操作数。Linear 18 = 线性注意力投影 3 层 × 3 + 全注意力 linear_proj 1 + 共享专家 4 层 × 2；LayerNormLinear 1 = 全注意力 linear_qkv；GroupedLinear 8 = 路由专家 4 层 × 2。线性注意力投影已进入 MXFP8。
- 插件一致性：BF16 下三个 TELinear 的输出与 `F.linear(input, weight)` 最大绝对差 0（9 个模块）。

| 对比 | logits 相对误差 | 各层输出相对误差 (L0/L1/L2/L3) | 路由不一致比例 (L0/L1/L2/L3) | 梯度最小余弦（各组最小值） |
| --- | --- | --- | --- | --- |
| BF16 重跑 vs BF16 | 0 | 0/0/0/0 | 0/0/0/0 | 1.000 |
| MXFP8 vs 仿真 | 0.095 | 0.0059/0.045/0.085/0.095 | 0/0.081/0.222/0.209 | 0.944（router）～0.995 |
| MXFP8 vs BF16 | 0.313 | 0.109/0.198/0.294/0.310 | 0.224/0.409/0.601/0.609 | 0.730（router）～0.951 |
| 仿真 vs BF16 | 0.312 | 0.109/0.198/0.294/0.310 | 0.224/0.412/0.610/0.613 | 0.737（router）～0.951 |

loss：BF16 12.8275，MXFP8 12.8123，仿真 12.8107。无 NaN/Inf。

### E2 补丁未测路径（单测）
- `tests/test_sm120_mxfp8_gemm.py` 新增 `test_linear_main_grad_accumulation_and_bias`、`test_grouped_linear_main_grad_accumulation`（fp32/bf16 各一）：`fuse_wgrad_accumulation=True` 时 wgrad 直接累加进 FP32 main_grad，bias 梯度来自未量化的 dy。4 个全部通过。
- 新增 `tests/test_sm120_mxfp8_sequence_parallel.py`（NUM_GPUS=2）：TE column/row parallel Linear + sequence_parallel，覆盖前向 all-gather MXFP8 输入、反向 all-gather 列方向数据后经补丁转置。实测误差（FP32）：column y 1.9e-7 / dx 6.5e-8 / dw 3.2e-7；row y 1.0e-7 / dx 1.2e-7 / dw 3.2e-7，容差 2e-5。

### E3 量化范围对齐官方后的 rollout 侧（v2 checkpoint，单卡 TP1）
- 生成：`tools/convert_hf_to_fp8.py --strategy mxfp8 --model-dir /workspace/hf_models/Qwen3.5-35B-A3B --save-dir /workspace/hf_models/Qwen3.5-35B-A3B-MXFP8-slime-v2`（36 GB）。`ignored_layers` 中的线性层与官方 `modules_to_not_convert` 一致：每层 `linear_attn.conv1d/in_proj_a/in_proj_b`、`mlp.gate`、`mlp.shared_expert_gate`，以及 `lm_head`、`embed_tokens`、`mtp.fc`、MTP 的两个 gate（其余条目为 norm、A_log、dt_bias 等非线性层参数）。
- SGLang 加载：`quant=mxfp8`，权重显存 35.27 GB（v1 36.1 GB），KV cache 1,237,787 token（v1 1,213,000）。`in_proj_qkv`+`in_proj_z` 合并为 `in_proj_qkvz` 后正常加载。
- 权重同步（`sync_sim.py`，`sync_report_v2.json`）：(a) 同步后与离线 v2 checkpoint 的输入/输出 logprob 最大差 0、输出 token 相同；(b) 推扰动权重后输入 logprob 最大差 10.7、输出改变，再推回后差 0；(c) 再同步 3 次每次差 0。FP8 张量 30,970 个（v1 30,880，多出 90 = 30 层 × 3 个线性注意力投影），36.75 GB，23.9 s。
- 精度（同阶段一方法：128 条 DAPO-Math，BF16 采样的 131,072 个回复 token 做 teacher-forcing；GSM8K test 前 300 题）：

| 配置 | mean\|Δlogp\| | p99\|Δlogp\| | KL(k3) | GSM8K |
| --- | --- | --- | --- | --- |
| BF16 decode vs prefill（参考噪声） | 0.0139 | 0.176 | 7.7e-4 | — |
| FP8 block128（量化线性注意力） | 0.0249 | 0.303 | 2.26e-3 | 96.33% |
| MXFP8 v1（线性注意力 BF16） | 0.0241 | 0.287 | 2.08e-3 | 96.33% |
| MXFP8 v2（对齐官方，线性注意力 MXFP8） | 0.0279 | 0.343 | 2.80e-3 | 95.33% |

  BF16 GSM8K 为 97.00%。v2 比 v1 多量化了 30 层的三个投影，KL 增加约 35%，GSM8K 少 3 题（300 题的标准误约 1.2 个百分点）。

- 速度（`bench.sh`，random 1024 输入 / 512 输出，单卡）：

| 并发 | v1 输出 tok/s | v2 输出 tok/s | v1 TPOT ms | v2 TPOT ms | v1 TTFT ms | v2 TTFT ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 144 | 109 | 6.4 | 8.9 | 301 | 154 |
| 16 | 986 | 964 | 13.9 | 15.4 | — | 616 |
| 64 | 2276 | 2380 | 25.0 | 23.9 | — | 1470 |

  并发 1 时 v2 的 decode 明显变慢：线性注意力三个投影在极小 M 下改走 MXFP8 GEMM，激活量化的固定开销超过了 GEMM 本身的收益。
