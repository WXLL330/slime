# SM120 MXFP8 并行一致性测试：真实实现 vs 伪量化

环境：8× RTX PRO 6000 Blackwell (sm120)，torch 2.11+cu129，TE 2.16.1，Megatron-core 0.16.0rc0。
脚本与产物：`/workspace/rl-project/slime_dev/mxfp8_parallel_work/`（`parallel_check.py`、`run_parallel.sh`、`summarize.py`、`r_*.json`、`r_*.log`）。

## 方法

- 模型：4 层裁剪 Qwen3.5 MoE（3 线性注意力 + 1 全注意力，16 专家 top-4），Megatron 随机初始化，slime Qwen3.5 插件。
- 同一进程、同一模型、同一权重、同一并行配置下依次运行：
  1. 真实 MXFP8（TE MXFP8BlockScaling + SM120 补丁），记录路由与每个 TE 模块的输入、grad_output；
  2. 伪量化 + teacher forcing（判定用）；
  3. 伪量化自由运行（只回放路由，参考用）。
- 伪量化：保留 `config.fp8`（Megatron 的专家 Fp8Padding 等路径不变），只把 `transformer_block.get_fp8_context` 换成空上下文，TE 走 BF16；每个 TE GEMM 的两个操作数按 TE 布局的累加维做 MXFP8 量化-反量化（`tests/_mxfp8_emulation.py`，与 TE 量化逐位一致），以 FP32 交给 GEMM、FP32 输出后再转回目标 dtype/缓冲区。
- 路由回放：复用 slime R3（`ENABLE_ROUTING_REPLAY=1`，真实运行 `ROUTING_REPLAY_STAGE=record`，伪量化运行 `RoutingReplay.clear_all_forward()` 后 `replay_forward`）。
- teacher forcing：所有 TE 模块（Linear / LayerNormLinear / GroupedLinear）前向输入替换为真实运行记录值（`rec + (x - x.detach())`，数值严格等于 rec，梯度对 x 为恒等），反向 grad_output 替换为真实运行记录值。于是每个 TE 模块的输出、dgrad、权重梯度之差只来自实现。
- 判定（阈值 1e-3，逐 rank 本地比较后取最差）：TE 模块前向输出、dgrad、TE GEMM 权重梯度的相对 L2 误差；路由一致率 = 1；真实运行全部 GEMM 为 MXFP8 操作数、伪量化运行全部不是，且两者 GEMM 调用一一对应；loss/logits/梯度全部有限。非 TE 参数（norm、router、gate、in_proj_a/b、conv1d 等）与自由运行结果只作参考。

## 问题记录

### Q1 阶段二记录的单卡“真实 vs 伪量化”第 0 层 0.0059 差异
- 现象：路由完全一致的第 0 层输出相对误差 0.0059，而单 GEMM 层面只有 ~1e-6。
- 原因：不是实现错误，是“整条流水线逐层比较”时量化的不连续性放大。逐模块追踪（`diag_trace.py`）：第一个 GEMM（in_proj_qkv）输入完全相同时输出只差 4.9e-6（累加顺序 + BF16 输出舍入），之后每次重新量化都会让部分元素越过 E4M3 舍入边界，误差按约平方根规律放大：out_proj 输入差 6e-6 → 输出差 8e-5；shared_experts fc1 输入差 1e-4 → 输出差 2.9e-3；fc2 输出 1.6e-2。
- 处理：判定改用逐 TE 模块 teacher forcing（见方法）；自由运行仅作参考。自由运行复现了该现象（第 0 层 5.9e-3 / 1.9e-3（带离群值），逐层放大到 ~4e-2～8e-2）。

### Q2 teacher forcing 写成 `x + (r - x).detach()` 在 BF16 下不精确
- 现象：强制后模块实际收到的输入与记录值相差 ~4e-5。
- 原因：BF16 中 `r - x` 与再加回 `x` 各舍入一次。
- 解决：改为 `r + (x - x.detach())`，`x - x.detach()` 严格为 0。

### Q3 伪量化基线自身误差：cuBLAS BF16 GEMM 在部分形状多一次 BF16 舍入
- 现象：强制输入后，experts/shared_experts linear_fc1 输出仍差 2.9e-3、dgrad 1.4e-3，其余模块 ~1e-5。对共享专家 fc1 用 FP64 精确参考比较：真实 MXFP8 与参考差 1.66e-3（恰为 BF16 输出舍入），伪量化差 2.45e-3。
- 原因：在模型外复现，M=1024/N=1024/K=2048 以及小 M（每专家 token 数）时，cuBLAS BF16 GEMM（TE 与 torch 相同）误差 2.45e-3，而 FP32 精确累加后舍入为 1.66e-3；其他形状无此现象（推测为 split-K 在 BF16 中归约）。属于伪量化基线的误差，真实实现正确。
- 解决：伪量化 GEMM 改为 FP32 输入、FP32 输出（MXFP8 值在 FP32/TF32 中精确，累加 FP32），再转换/拷贝到 TE 期望的输出 dtype 或缓冲区（wgrad 融合累加时直接用 FP32 main_grad）。修正后 fc1 输出差 1.8e-5、dgrad 1.5e-5。TE 不支持 FP32 输入 + BF16 输出（CUBLAS_STATUS_NOT_SUPPORTED），故用 FP32 输出。

### Q4 “分块方向错误”负例对随机权重不可见
- 现象：伪量化中把 dgrad 的权重按错误方向分块（`--negative-control direction`），无论是否注入 32 倍离群值，误差几乎不变（dgrad 1.5e-5 → 2e-5；离群值下 1.1e-5 → 3.6e-5）。
- 原因：E4M3 是浮点格式，块内每个元素相对精度不变；只有块内动态范围超过约 2^15 的元素才会因分块方向不同而改变（与 `notes/sm120-mxfp8-emulation-tests.md` 结论一致）。所以“方向用错”在数值上本身几乎无害，不适合衡量测试灵敏度。
- 处理：负例改为直接破坏真实实现（替换 `sm120_mxfp8.transpose_columnwise_to_rowwise`）：`scale_roll`（scale 错位一行）与 `swizzle`（对 GEMM-swizzled scale 做普通转置）。两者均被判 FAIL（误差 ~1，见结果表）。保留 `direction` 选项并记录此现象。

### Q5 `linear_qkv` 组的梯度误差（2.4e-4～3.5e-4）高于其他 TE 层
- 现象：汇总里 `linear_qkv` 组梯度误差比其他 TE 层（1e-7 级）高三个数量级。
- 排查：独立 `te.LayerNormLinear`（RMSNorm、zero_centered_gamma、离群值输入，FP32 main_grad）的 wgrad 与“对 TE 实际 BF16 归一化输出伪量化”的参考差 3.1e-8，与 te.Linear 相同；BF16 与 MXFP8 模式下 TE 输出的归一化结果逐位相同。
- 原因：该组按子串还包含了 `linear_qkv.layer_norm_weight`（LayerNorm 参数，梯度经归一化反向，来自未被 forcing 的路径），GEMM 权重 `linear_qkv.weight` 本身误差 4.9e-8。
- 处理：汇总时 TE 权重梯度只统计 GEMM 权重，layer_norm_weight 归入非 TE 参数。

## 结果（均开启 `--outliers`：每个 TE 权重 0.5% 元素 ×32，embedding 8 个通道 ×20）

见下方结果表（由 `summarize.py` 生成，持续更新）。

### 结果表（`parallel_check.py`，`summarize.py` 输出，`summary.txt`）

判定列：te_out（TE 模块前向输出）、te_dgrad（TE 模块输入梯度）、te_wgrad（TE GEMM 权重梯度，FP32 main_grad），均为所有 rank、所有模块的最大相对 L2 误差，阈值 1e-3；route 为路由一致率；gemm_ok 表示真实运行全部 GEMM 为 MXFP8、伪量化运行全部不是，且调用一一对应（每 rank 81 次：前向/dgrad/wgrad 各 27）。参考列：loss_diff、logits、layers（强制运行下残差流累积的差异）、nonTE_grad（norm、router、gate、in_proj_a/b、conv1d、embedding、output_layer 等非 GEMM 参数）、free_*（自由运行）。tok/rank 为 SP/CP 切分后每 rank 的 token 数。

| 配置 | GPU | tok/rank | loss_diff | logits | te_out | te_dgrad | te_wgrad | nonTE_grad | route | gemm_ok | free_logits | free_layers | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 单卡（无离群值） | 1 | 1024 | 1.9e-6 | 1.3e-6 | 4.1e-5 | 5.5e-5 | 1.2e-7 | 1.9e-3 | 1.0 | 是 | 1.1e-4 | 3.9e-2 | PASS |
| 单卡 | 1 | 1024 | 1.5e-5 | 2.3e-6 | 1.6e-4 | 1.3e-4 | 4.8e-7 | 2.4e-3 | 1.0 | 是 | 1.2e-4 | 4.4e-2 | PASS |
| TP2+SP | 2 | 512 | 1.9e-6 | 4.1e-6 | 1.0e-4 | 1.7e-4 | 4.9e-7 | 2.8e-3 | 1.0 | 是 | 2.2e-4 | 4.8e-2 | PASS |
| DP2 | 2 | 1024 | 1.5e-5 | 2.3e-6 | 1.6e-4 | 1.4e-4 | 4.8e-7 | 3.5e-3 | 1.0 | 是 | 1.2e-4 | 4.5e-2 | PASS |
| CP2 | 2 | 512 | 1.1e-5 | 2.0e-6 | 1.2e-4 | 1.5e-4 | 2.1e-7 | 2.6e-3 | 1.0 | 是 | 1.2e-4 | 4.4e-2 | PASS |
| EP4（DP4） | 4 | 1024 | 2.0e-5 | 2.3e-6 | 1.3e-4 | 1.8e-4 | 4.6e-7 | 4.1e-3 | 1.0 | 是 | 1.3e-4 | 4.5e-2 | PASS |
| TP2+SP+EP4（DP2） | 4 | 512 | 1.3e-5 | 4.4e-6 | 1.4e-4 | 3.7e-4 | 5.3e-7 | 3.7e-3 | 1.0 | 是 | 2.2e-4 | 4.5e-2 | PASS |
| TP2+SP+EP4+CP2（DP2） | 8 | 256 | 2.2e-5 | 4.3e-6 | 2.1e-4 | 1.9e-4 | 2.5e-7 | 4.4e-3 | 1.0 | 是 | 2.2e-4 | 4.8e-2 | PASS |
| TP2+SP+EP8（DP4，同 8 卡 GRPO 切分） | 8 | 512 | 2.8e-5 | 4.1e-6 | 1.9e-4 | 2.3e-4 | 6.3e-7 | 4.5e-3 | 1.0 | 是 | 2.2e-4 | 4.6e-2 | PASS |
| 负例 scale_roll（单卡） | 1 | 1024 | 1.5e-5 | 2.3e-6 | 1.6e-4 | 1.0 | 1.1 | 1.2 | 1.0 | 是 | — | — | FAIL（预期） |
| 负例 swizzle（单卡） | 1 | 1024 | 1.5e-5 | 2.3e-6 | 1.6e-4 | 1.4e-4 | 1.2 | 2.4e-3 | 1.0 | 是 | — | — | FAIL（预期） |
| 负例 direction（单卡，伪量化侧 dgrad 权重分块方向错误） | 1 | 1024 | 1.5e-5 | 2.3e-6 | 1.6e-4 | 1.4e-4 | 4.8e-7 | 2.8e-3 | 1.0 | 是 | 1.2e-4 | 4.4e-2 | 未检出（见 Q4） |

除“单卡（无离群值）”外均开启 `--outliers`。用 `--tokenizer-type NullTokenizer` 复跑单卡，结果与 HuggingFace tokenizer 相同（测试因此不依赖本地 tokenizer）。

解读：
- 所有并行配置下，TE 模块前向输出与 dgrad 的误差 ≤ 3.7e-4（BF16 输出舍入 + 累加顺序），TE GEMM 权重梯度 ≤ 6.3e-7（FP32 累加），路由 100% 一致，与单卡同一量级；SP all-gather 后的转置、EP 每专家补零、CP zigzag 切分、DP 拆批下真实实现都与理想 MXFP8 语义一致。
- 负例中补丁转置被破坏时误差 ~1，判定阈值 1e-3 与正常值和错误值之间各有约三个数量级余量。
- 非 TE 参数梯度（~2e-3～4.5e-3）与自由运行（logits 1e-4～2e-4、逐层 ~4e-2）不作判定：前者经过未被 forcing 的残差/归一化反向路径累积，后者是量化边界放大（见 Q1）。

## 测试固化

- `tests/_sm120_mxfp8_parallel_harness.py`：torchrun harness（精简自 `parallel_check.py`，始终注入离群值，负例保留 `scale_roll`、`swizzle`）。
- `tests/test_sm120_mxfp8_parallel_consistency.py`：`NUM_GPUS = 8`，非 SM12x 跳过、GPU 不足的配置跳过；8 个并行配置 + 2 个负例；裁剪模型配置内嵌，tokenizer 用 NullTokenizer，不依赖仓库外文件。
- CI：未注册到固定矩阵。仓库现有 GPU runner 不是 SM120（与 `test_sm120_mxfp8_gemm.py`、`test_sm120_mxfp8_sequence_parallel.py` 相同），注册后只会全部跳过；`run-ci-changed` 会按 `NUM_GPUS = 8` 调度。本地运行：`PYTHONPATH=/root/Megatron-LM python tests/test_sm120_mxfp8_parallel_consistency.py`。

### 测试运行结果

- `PYTHONPATH=/root/Megatron-LM:/workspace/rl-project/slime_dev/sglang-sm120-mxfp8/python python tests/test_sm120_mxfp8_parallel_consistency.py -v`：10 passed（8 个并行配置 + scale_roll/swizzle 两个负例），393 s。
- 回归：`tests/test_sm120_mxfp8_gemm.py` 116 passed，`tests/test_mxfp8_weight_sync.py` 18 passed，`tests/test_sm120_mxfp8_sequence_parallel.py` 2 passed。
- 本轮未修改 SM120 补丁：所有配置下真实实现与理想 MXFP8 语义一致，发现的问题（Q1～Q5）均在测试方法/伪量化基线一侧。
