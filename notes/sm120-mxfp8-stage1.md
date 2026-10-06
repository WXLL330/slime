# sm120 MXFP8 阶段 1：MXFP8 rollout（训练保持 BF16）

## 环境

- GPU：1 张 NVIDIA RTX PRO 6000 Blackwell Server Edition（sm120，96 GB）
- torch 2.11.0+cu129，SGLang 0.5.15.post1（/sgl-workspace/sglang，带 slime docker 补丁），FlashInfer 0.6.12，Triton 3.6.0，TE 2.16.1
- slime worktree：基于本地 main `624b824a`（未 rebase 上游，见问题 1）
- SGLang worktree：`/workspace/rl-project/slime_dev/sglang-sm120-mxfp8`，分支 `sm120-mxfp8`
  - 提交 `e95473177`：把 /sgl-workspace/sglang 的未提交改动（slime docker 补丁）原样镜像为基线
  - 之后的提交是本阶段的改动，导出为 `docker/patch/latest/sglang-sm120-mxfp8.patch`
  - 测试时用 `PYTHONPATH=/workspace/rl-project/slime_dev/sglang-sm120-mxfp8/python` 加载

## 问题记录

### 问题 1：无法同步上游 slime

- 现象：本机访问不了 GitHub 的 git 接口（`git ls-remote` 超时，没有 `gh`）。
- 原因：网络限制。
- 解决方案：基于本地 main 开发。上游在 #2298 之后合入了 #2327（UE8M0 scale）以及 #2316/#2320/#2321（权重更新相关重构），合并时可能在 `quantizer_fp8.py` 和两个 updater 文件上产生冲突，需要人工处理。

### 问题 2：已安装的 SGLang 不等于 git HEAD

- 现象：`/sgl-workspace/sglang` 工作区有 60 多个未提交修改和 4 个未跟踪文件（slime docker 补丁），直接在 HEAD 上建 worktree 会得到与运行环境不同的代码。
- 原因：slime 的 Dockerfile 用 `git apply` 打补丁，不提交。
- 解决方案：`git diff --binary` 导出这些修改并应用到 worktree，复制未跟踪文件，提交为基线 `e95473177`；本阶段改动单独提交，导出的补丁只包含本阶段改动，可叠加在现有 docker 补丁之后。

### 问题 3：转换工具对 Qwen3.5 BF16 checkpoint 的专家张量不生效

- 现象：BF16 checkpoint 的专家是融合 3D 张量 `mlp.experts.gate_up_proj [256, 1024, 2048]`、`mlp.experts.down_proj [256, 2048, 512]`，键名不含 `weight`，原工具会原样保留 BF16；并且原工具的 `block_fp8` 只接受 2D 张量。
- 原因：原工具按官方 FP8 checkpoint 的逐专家命名编写。
- 解决方案：新增 `split_fused_experts`，在量化前拆成 `experts.{e}.gate_proj/up_proj/down_proj.weight`（HF 中 `gate_up_proj` 前半为 gate，已核对 transformers 实现），与 slime 同步时发送的逐专家命名一致。对 block/channel/tensor 策略同样生效。

### 问题 4：MXFP8 下 SGLang 需要 `weight_block_size`

- 现象：`Fp8MoEMethod.create_weights` 在 block quant 分支直接读 `weight_block_size[0/1]`。
- 原因：`Fp8Config.from_config` 只有在配置里存在 `weight_block_size` 时才改成 `[1, 32]`。
- 解决方案：转换工具写出的 `quantization_config` 固定带 `"weight_block_size": [1, 32]`。

### 问题 5：转换工具原黑名单与 SGLang 的实际量化范围不一致

- 现象：原黑名单不排除 `shared_expert_gate`（SGLang 中为 `quant_config=None` 的 ReplicatedLinear）、视觉塔和 `mtp.fc`。
- 原因：黑名单是按 FP8 写的。
- 解决方案：`should_quantize(key, "mxfp8")` 额外排除 `linear_attn`、`shared_expert_gate`、`visual`、`mtp.fc`；单测 `tests/test_mxfp8_weight_sync.py` 逐项断言 slime 同步量化器与转换工具对每个 HF 张量的量化决策一致。block 策略保持原行为不变。

### 问题 6：本机 http_proxy 拦截 localhost

- 现象：SGLang 服务模型加载、CUDA graph 捕获都成功，但启动预热请求返回 502，服务被 kill；`curl localhost:30000` 也是 502。
- 原因：环境变量 `http_proxy=http://127.0.0.1:9999`，访问 localhost 也走了代理。
- 解决方案：服务端和所有客户端命令都设置 `no_proxy=127.0.0.1,localhost`。

### 问题 7：SGLang `/post_process_weights` 接口在此版本必然返回 400

- 现象：同步模拟推送 35 个分块均成功（HTTP 200），随后调用 `/post_process_weights` 返回 400：`'loc': ('query', 'req'), 'msg': 'Field required'`。
- 原因：此 SGLang 版本把请求结构体改成了 `msgspec.Struct`（`BaseReq`），其他权重接口都写成 `Annotated[XxxReqInput, Body()]`，唯独 `post_process_weights(req: PostProcessWeightsReqInput, ...)` 漏了 `Body()`，FastAPI 将其当作查询参数解析。该问题与 MXFP8 无关，slime 现有的 INT4（compressed-tensors）同步路径在此版本上同样会失败。
- 解决方案：在 SGLang 补丁中把签名改为 `req: Annotated[PostProcessWeightsReqInput, Body()]`。

### 问题 8：转换工具生成的 FP8 blockwise checkpoint 输出乱码（原有缺陷）

- 现象：用 `--strategy block --block-size 128 128` 生成的 FP8 checkpoint 能加载，但 “The capital of France is” 续写为乱码，GSM8K 准确率 2.3%，相对 BF16 的 KL 估计 0.52。
- 原因：原黑名单只排除 `mlp.gate.`，不排除 `mlp.shared_expert_gate`（[1, 2048]）。SGLang 中该层是 `quant_config=None` 的 BF16 层，checkpoint 里的 `weight_scale_inv` 找不到对应参数被跳过，FP8 原始数值（量级 ±448）被直接复制为 BF16，共享专家门控被放大数千倍。该缺陷在本阶段之前就存在，与 MXFP8 无关。
- 解决方案：`should_quantize` 对所有策略排除 `shared_expert_gate`；删除有缺陷的 FP8 checkpoint 后重新生成。有缺陷版本的测量结果保留为 `*_fp8_broken_gate.*`，不计入对比。

### 问题 9：BF16 服务在 teacher-forcing 打分时 OOM

- 现象：32 并发打分时 BF16 服务崩溃：`logits[input_logprob_indices]` 申请 6.62 GiB 失败。
- 原因：请求 input logprob 时，一个 prefill chunk（最多 8192 token）会物化 `[tokens, 248320]` 的 fp32 logits；BF16 权重占 65.5 GB，`mem_fraction_static=0.85` 下余量仅约 12.9 GB。
- 解决方案：打分客户端并发降到 2（三种精度统一使用），服务端配置保持不变。

## 实验记录

### E1：MoE kernel 单测（`mxfp8_stage1_work/test_moe_kernel.py`）

E=256、H=2048、I=512、top-8，与“反量化权重 + MXFP8 量化激活”的精确仿真比较：

| T | 与 MXFP8 仿真的相对误差 | 与 BF16 的相对误差 |
| --- | --- | --- |
| 1 | 1.66e-3 | 0.063 |
| 7 / 64 / 1000 / 4096 | 1.66e-3 | 0.066 |

1.7e-3 仅为 BF16 中间结果舍入，说明 kernel 计算正确；0.066 为 MXFP8 量化噪声。

### E2：服务端冒烟与 profiler 取证

- 启动：`start_server.sh <MXFP8 ckpt> <log> --fp8-gemm-backend flashinfer_cutlass --moe-runner-backend triton`（`--mem-fraction-static 0.85 --disable-radix-cache`）
- 生成：“用一句话介绍一下长城，然后计算 37*43” → 回答正确（1591），文本通顺。
- torch profiler（5 个前向步）：
  - `_mxfp8_grouped_gemm_kernel` 400 次 = 40 层 × 2 GEMM × 5 步，占 GPU 时间 21%
  - FlashInfer `DeviceGemmMxfp8GemmSm120` 500 次 = 每步 100 次（40 层共享专家 × 2 + 10 个全注意力层 × 2）
  - `_fused_moe_kernel`（BF16/FP8 Triton MoE）0 次，没有逐专家 GEMM
  - BF16 `gemvx` 来自线性注意力投影、路由 gate 和 lm_head，符合“线性注意力保持 BF16”的决策

### E3：单卡权重同步模拟（`mxfp8_stage1_work/sync_sim.py`，结果 `sync_report.json`）

路径与 slime `UpdateWeightFromTensor` 一致：BF16 HF checkpoint → `hf_to_megatron.qwen3_5_hf_tensor`（得到 TP=EP=1 的 Megatron 命名参数）→ `megatron_to_hf.convert_to_hf(..., quantization_config=mxfp8)` → slime 的 `_build_flattened_tensor_data` + `MultiprocessingSerializer`（CUDA IPC）→ `/update_weights_from_tensor`（`load_format=flattened_bucket`，1 GB 分块）→ 按 `requires_post_process_after_update` 调 `/post_process_weights` → `/flush_cache`。服务从离线 MXFP8 checkpoint 冷启动。探针为 4 条提示的全部输入 token logprob + 32 token 贪心输出。

| 检查项 | 结果 |
| --- | --- |
| 冷启动后重复探针 | 差 0.0 |
| (a) 同步原权重后 vs 离线 checkpoint | 输入 logprob 最大差 0.0，输出 token 完全一致 |
| (b) 同步扰动权重（2D 权重乘 1+0.2·N(0,1)） | 输入 logprob 最大差 8.62，输出改变 |
| (b) 再同步原权重 | 差 0.0 |
| (c) 再重复同步 2 次 | 每次差 0.0 |

每次同步 62,213 个张量，其中 30,880 个 FP8（= 40 层 × (256 专家 × 3 + 共享专家 3) + 10 个全注意力层 × 4），共 37.7 GB，耗时约 131 s（瓶颈是从 NFS 读 BF16 checkpoint）。

### E4：BF16 / FP8 blockwise / MXFP8 推理对比

公共设置：单卡 RTX PRO 6000，TP=1，`--mem-fraction-static 0.85 --disable-radix-cache --moe-runner-backend triton`；MXFP8 另加 `--fp8-gemm-backend flashinfer_cutlass`，FP8 用默认 GEMM 后端。

checkpoint：

| 配置 | 路径 | 量化范围 |
| --- | --- | --- |
| BF16 | `/workspace/hf_models/Qwen3.5-35B-A3B` | 无 |
| FP8 | `/workspace/hf_models/Qwen3.5-35B-A3B-FP8-block128-slime`（本工具 `--strategy block --block-size 128 128` 生成，已修 `shared_expert_gate`） | 全注意力、线性注意力 in_proj_qkv/z 与 out_proj、共享专家、路由专家（128×128 块，fp32 scale） |
| MXFP8 | `/workspace/hf_models/Qwen3.5-35B-A3B-MXFP8-slime`（`--strategy mxfp8`） | 全注意力、共享专家、路由专家（1×32 块，UE8M0 scale）；线性注意力保持 BF16 |

注意：FP8 比 MXFP8 多量化了线性注意力投影，两者量化覆盖面不同。

**精度**（`precision_eval.py` + `analyze_precision.py`）：128 条 DAPO-Math 提示，由 BF16 以温度 1 采样最长 1024 token（共 131,072 个回复 token），三种配置对同一序列做 teacher-forcing 打分，以 BF16 打分为基准。KL 用 k3 估计 mean(exp(d)−1−d)，d = logp_Q − logp_BF16。GSM8K 取 test 前 300 题，chat 模板、关闭思考、贪心、最长 1024 token。

| 配置 | mean\|Δlogp\| | p99\|Δlogp\| | max\|Δlogp\| | KL(BF16‖Q) | \|Δ\|>0.1 比例 | GSM8K |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 重复打分（噪声底） | 0 | 0 | 0 | 0 | 0 | 97.00% |
| BF16 decode vs prefill（参考：BF16 自身的训推式差异） | 0.0139 | 0.176 | 0.875 | 7.7e-4 | 3.70% | — |
| FP8 blockwise | 0.0249 | 0.303 | 1.496 | 2.26e-3 | 7.99% | 96.33% |
| MXFP8 | 0.0241 | 0.287 | 1.495 | 2.08e-3 | 7.75% | 96.33% |

**显存**（服务启动日志）：

| 配置 | 权重显存 | KV cache token 数 | 最大并发请求 |
| --- | --- | --- | --- |
| BF16 | 65.49 GB | 404,768 | 114 |
| FP8 | 34.19 GB | 1,266,016 | 362 |
| MXFP8 | 36.10 GB | 1,213,028 | 347 |

**速度**（`bench.sh`：`sglang.bench_serving`，random-ids，输入 1024 / 输出 512，并发 1/16/64 分别 8/64/256 个请求）：

| 并发 | 指标 | BF16 | FP8 | MXFP8 |
| --- | --- | --- | --- | --- |
| 1 | 输出吞吐 tok/s | 172.2 | 142.6 | 144.4 |
| 1 | 平均 TTFT ms | 98 | 132 | 301 |
| 1 | 平均 TPOT ms | 5.62 | 6.76 | 6.35 |
| 16 | 输出吞吐 tok/s | 813.2 | 1252.2 | 985.5 |
| 16 | 平均 TTFT ms | 1044 | 483 | 1210 |
| 16 | 平均 TPOT ms | 17.64 | 11.83 | 13.87 |
| 64 | 输出吞吐 tok/s | 1753.2 | 2482.3 | 2275.5 |
| 64 | 平均 TTFT ms | 1481 | 1216 | 1544 |
| 64 | 平均 TPOT ms | 33.52 | 23.24 | 25.01 |

解读：MXFP8 的 decode（TPOT）在各并发下都优于 BF16，略逊于 FP8；prefill（TTFT）明显慢于 FP8 和 BF16。新 MoE kernel 的分块参数（BLOCK_M=64、BLOCK_N=128、BLOCK_K=128）照搬 gfx950 版本，未针对 SM120 调优。按用户要求，本阶段不定位首 token 延迟问题。

## 未解决问题与后续

1. MXFP8 prefill 偏慢（TTFT 约为 FP8 的 2.3 倍）：待调优 `mxfp8_moe_sm120.py` 的分块参数，或改用 FlashInfer cuTile MXFP8 MoE（需升级 FlashInfer）。本阶段按要求不处理。
2. 未在真实 slime 多卡 RL 中跑过：同步只在单卡上用 HTTP + CUDA IPC 模拟（TP=EP=1）。多卡下的 TP/EP 切分、`update_weight_from_distributed` 的 NCCL 路径、以及 `--sglang-fp8-gemm-backend flashinfer_cutlass` 在 TP=2 下的 scale 重排，都需要在 8 卡环境验证。
3. 未 rebase 上游 slime（问题 1）。
4. SGLang 中 `Fp8MoEMethod` 的 `get_triton_quant_info` 现在会传 `use_mxfp8`，非 SM120 的 NVIDIA GPU 上使用 Triton MoE runner 跑 MXFP8 会得到明确的 `NotImplementedError`（以前会静默走错误的 FP8 路径）。
5. 阶段 2 衔接点：训练侧开启 MXFP8 后，量化器仍复用同一个 `mxfp8_group_quantize`（已验证与 TE `MXFP8Quantizer` 逐位一致），线性注意力仍保持 BF16，`ignored_layers` 不需要变化。

## 实验脚本与原始结果

全部位于 `/workspace/rl-project/slime_dev/mxfp8_stage1_work/`：`test_moe_kernel.py`、`sync_sim.py`（`sync_report.json`）、`precision_eval.py`、`analyze_precision.py`（`precision_summary.json`、`score_*.json`、`gsm8k_*.json`）、`bench.sh`（`bench_*.jsonl`）、`start_server.sh`、`wait_server.sh`、`patch_check.sh`、服务日志 `server_*.log`、profiler trace `prof_mxfp8/`。
