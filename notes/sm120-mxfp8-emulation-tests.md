# SM120 MXFP8 训练 GEMM：仿真参考与第 2、3 层测试

## 目标

没有 SM100 硬件，无法拿原生结果做对比。改用 MXFP8 GEMM 的数学定义作为基准：两个操作数沿累加维按 MXFP8 量化，再精确累加。在此基础上分三层验证：

1. 第 1 层，操作数：仿真量化器、转置副本都与 TE 量化结果逐位一致；
2. 第 2 层，单个 GEMM：fprop TN、dgrad NN、wgrad NT 都与仿真参考的差异在 FP32 累加误差量级；
3. 第 3 层，模块：`te.Linear`、`te.LayerNormLinear`、`te.GroupedLinear` 的前向和反向与仿真 autograd 对比。

## 文件

| 文件 | 内容 |
| --- | --- |
| `slime/backends/megatron_utils/sm120_mxfp8.py` | 被测补丁。需要显式调用 `apply_sm120_mxfp8_patch()`，import 时不会生效，也没有接入 `initialize.py`。NN 时转换 A，NT 时转换 B，统一改为 TN；转换方式是直接转置列方向的 FP8 数据和 scale |
| `tests/_mxfp8_emulation.py` | 纯 PyTorch 的 MXFP8 量化、反量化和参考 GEMM，只依赖 torch |
| `tests/test_sm120_mxfp8_gemm.py` | 第 1 到第 3 层测试，共 112 个用例，在 sm120 上约 5 到 13 秒 |

## TE MXFP8 量化语义（实测，仿真已逐位复现）

- 每 32 个元素一块，块沿量化方向排列。TE 要求两个维度都能被 32 整除，否则在 `quantizer.cpp:1383 create_tensor` 断言失败。
- scale：amax 乘以 FP32 常数 1/448 后向上取整到 2 的幂，偏置指数最大饱和到 254；全零块的指数为 0。
- 数据：`x * 2^(127-e)` 用 RNE 舍入到 E4M3，并按 satfinite 饱和到 ±448；NaN 保持 NaN（0x7F）。
- Inf：所在块的 scale 为 254，Inf 元素变成 ±448（0x7E/0xFE），同块其他元素变成 ±0。
- NaN：行方向 kernel 用 `fmaxf` 求 amax，NaN 被忽略，块内其他元素正常量化；列方向 kernel 让 NaN 传染到整块，scale 为 0xFF，整块数据都是 NaN。两条路径对 NaN 的处理不一致。
- scale 的 padding：TE 不初始化 padding 区域，里面是分配器的残留数据，唯一的例外是行方向在 K 方向上的 padding 列，总是写 0。实测 sm120 上的 cuBLAS 不读 K 方向 padding 的 scale：写入 0xFF、0x7F、0x01、0xFE，GEMM 结果都逐位不变。
- swizzle：`te.Linear` 和 `te.LayerNormLinear` 的输入及梯度 quantizer 会把 `optimize_for_gemm` 设为 True，scale 在量化时就直接写成 swizzle 格式，权重则保持普通格式。列方向 scale 的 swizzle 结果，与转置张量行方向 swizzle 的结果逐字节相同（包括 padding），所以转置时只需交换二维形状，不需要搬动数据。
- LayerNormLinear：TE 在 FP32 中做归一化，把结果写成激活的 dtype 后再量化。在 BF16 下，按“先舍入到 BF16 再量化”构造参考，误差 7e-4；按 FP32 直接量化构造参考，误差 6.5e-2。dgrad GEMM 的输出也会先舍入到激活 dtype，再做归一化的反向。`return_layernorm_output=True` 不改变任何输出和梯度，已验证逐位相同，所以测试用它取 TE 实际量化的 `ln_out`。

## 判定标准与容差

- **误差指标**：逐行相对误差的最大值。整体 Frobenius 范数会被离群行主导，掩盖错误分块方向在小值上造成的误差。
- **`GEMM_TOL = 2e-5`**：FP32 累加 K ≤ 4096 项，随机游走误差约 sqrt(K)·2^-24 ≈ 4e-6。实测最大值：单个 GEMM 1.2e-6，FP32 模块 4e-6，LayerNorm 部分 3e-7。
- **`BF16_MODULE_TOL = 2^-8 + GEMM_TOL`**：BF16 输出是 FP32 结果舍入一次得到的，与精确值的相对误差不超过半个 BF16 ulp，即 2^-8。实测最大值 2.9e-3。
- **数据**：除高斯数据外，还用了离群值数据，每隔 97 个切片放大 2^12，并且离群值放在每个操作数的非累加维上。

### 关键发现：MXFP8（E4M3）对分块方向不敏感

E4M3 是浮点格式，scale 改变 2 的整数次幂时，元素的相对舍入误差不变。所以只要元素在 E4M3 的正规数范围内，无论沿哪个方向分块，量化结果都完全相同。只有当元素比块内 amax 小大约 2^15 以上、落入次正规区或被冲成 0 时，分块方向才会影响结果。在高斯数据上，正确方向与错误方向的参考结果只差约 1e-7，测试根本区分不出来。负例因此必须用块内动态范围大的数据（类似 LLM 激活的离群通道），并配合逐行指标：

| 形状 | 实现与正确参考的误差（fprop / dgrad / wgrad） | 实现与错误方向参考的误差 |
| --- | --- | --- |
| qkv_tp2 | 2.1e-7 / 5.7e-7 / 5.1e-7 | 7.0e-3 / 1.0e-2 / 9.1e-3 |
| o_proj | 5.1e-7 / 2.6e-7 / 4.0e-7 | 1.1e-2 / 8.9e-3 / 7.0e-3 |
| expert_fc1 | 1.2e-7 / 1.0e-7 / 5.3e-8 | 9.4e-3 / 6.8e-3 / 1.1e-2 |
| expert_fc2 | 8.2e-8 / 2.0e-7 / 8.2e-8 | 7.4e-3 / 1.2e-2 / 1.9e-2 |

两者相差 3 个数量级以上。

## 变异测试（证明测试能抓到真实的错误实现）

把被测的 `transpose_columnwise_to_rowwise` 换成以下错误版本后运行整个测试套件，脚本在 scratchpad 的 `emu/mutation.py`：

| 变异 | 结果 |
| --- | --- |
| 原型写法（`dequantize()` → 转置 → 重量化） | 非 GroupedLinear 用例 28 个失败（第 1 层 8、第 2 层 16、负例 4）；GroupedLinear 用例遇到空专家时 SIGFPE 崩溃 |
| 用行方向副本构造转置 | 同上：28 个失败，GroupedLinear 崩溃 |
| swizzle 情形下仍对 scale 做普通转置 | 24 个失败（第 1 层 4、第 2 层 12、Linear 2、LayerNormLinear 4） |

正确实现：112 个用例全部通过，空专家的 GroupedLinear 也正常。

## 问题记录（现象 / 原因 / 解决方案）

1. **NaN/Inf 的仿真与 TE 不一致**
   - 现象：Inf 情形下 TE 输出 0x7E，仿真输出 0x7F；NaN 情形下行、列方向的 scale 不一致。
   - 原因：TE 的转换是 satfinite 的；行方向用 fmaxf 忽略 NaN，列方向让 NaN 传染整块。
   - 解决：仿真中先 clamp 到 ±448 再转换（NaN 不受影响）；行方向把 NaN 置 0 后求 amax，列方向直接求 amax。
2. **−Inf 用例的行方向 scale 是 247，不是 254**
   - 原因：`nan_to_num(nan=0)` 会顺带把 Inf 替换成 FP32 的最大有限值。
   - 解决：改用 `torch.where(isnan, 0, x)`。
3. **真实 `te.Linear` 反向时补丁断言失败，提示“swizzled scales”**
   - 原因：Linear 的输入 quantizer 设置了 `optimize_for_gemm=True`。之前只 grep 了 Python 代码，误以为这三个模块不会开启 swizzle。
   - 解决：实测发现列方向 swizzle 后的 scale，与转置张量行方向 swizzle 的 scale 逐字节相同，转置时只交换二维形状即可。
4. **原型在 GEMM 和模块里并不正确**
   - 原因：`MXFP8TensorStorage.dequantize()` 在行、列两份数据都存在时优先用行方向数据。反向中，权重、输入、梯度都带有两份数据，所以原型实际是“先按错误方向量化，再重量化”。原型阶段只与 BF16 对比（误差 0.038），看不出这个问题。
   - 解决：直接转置列方向副本，不再经过 `dequantize()`。
5. **错误分块方向的负例检测不出来（误差约 1e-7）**
   - 原因：见上文“关键发现”，加上 Frobenius 范数被离群行主导。
   - 解决：负例使用离群值数据，误差指标改为逐行相对误差的最大值。
6. **FP32 下 LayerNormLinear 的误差为 2.4e-4 到 3.5e-4，超过容差**
   - 原因：参考实现用 FP64 重算 LayerNorm 后舍入到 FP32，与 TE 的 FP32 结果相差 1 个 ulp；落在 E4M3 舍入边界上的元素因此跳变一整个 E4M3 步长。这是 LayerNorm 算术差异被量化放大，不是 GEMM 的错误。
   - 解决：GEMM 参考改用 TE 实际量化的 `ln_out`（`return_layernorm_output=True`，已验证逐位不影响结果），LayerNorm 本身单独与 FP64 参考比对。修正后 FP32 所有指标 ≤ 6e-7。
7. **调试脚本误把 uint8 当 FP8 使用，得到“相对误差 2.1”的假结果**
   - 原因：TE 以 uint8 存储 FP8 数据。
   - 解决：先 `.view(torch.float8_e4m3fn)` 再反量化。正式测试一律按 uint8 比较字节，不受影响。
8. **`python -c` 或脚本导入的 slime 来自主工作区**
   - 原因：环境以 editable 方式安装了主工作区的 slime。
   - 解决：测试文件把 worktree 根目录插到 `sys.path` 最前面。

## 更正：旧做法（反量化到 BF16 → 转置 → 重量化）与直接转置并非逐位一致

协调方更正了任务第 1 条的判定标准，我的实测结果与之一致：

- **主判定**：直接转置的结果，与“先转置原始高精度张量、再由 TE `MXFP8Quantizer` 按行方向量化”的结果逐位一致，数据和 scale 都比较。对应用例 `test_transposed_copy_is_rowwise_quantization_of_transpose`，覆盖普通和 swizzle 两种 scale，以及 4 种形状。在 FP32 输入 1e-36 的极小量级下也成立。
- **与旧做法的比较**：只在正常数值范围内比较反量化后的数值，以及 GEMM 输出，不比较位模式。对应用例 `test_transpose_matches_dequantize_requantize_prototype`。
- **已知现象，不作为断言**：
  - 正常量级下，旧做法有一部分块的 scale 小 1（E8M0 差 −1），对应 FP8 编码整体 +8（数值 ×2），反量化后的数值完全相同。实测：256×512 的 FP32 randn 上为 144/4096 块（3.5%），协调方实测为 139/4096；1024×2048 的 BF16 数据上为 2327 到 2911 块（约 1.1% 到 1.4%）。原因是块的 amax/scale 落在 (224, 232) 区间时会被 RNE 舍入到 224，反量化后 amax 正好是 448·2^(e−1)，重新量化时便选了更小的 scale。
  - 极小量级（约 1e-36）下，旧做法反量化到 BF16 时会下溢，数值也不同。实测 256×512 的 FP32 randn×1e-36 上有 106710 个元素不同，协调方实测为 6719 个（输入 dtype 和数据不同）。
  - 在只含列方向数据的张量上，旧做法与正确结果数值相同；在同时含两份数据的张量上，旧做法使用了错误的方向（见问题 4）。

## microxcaling 与 TE 的差异（交叉参考，不作为逐位基准）

参考代码：`/workspace/rl-project/slime_dev/microxcaling/mx/mx_ops.py::_quantize_mx`、`_shared_exponents`，以及 `elemwise_ops.py::_quantize_elemwise_core`。仿真参考不依赖 microxcaling。

| 项目 | microxcaling（OCP MX v1.0 参考实现） | TE |
| --- | --- | --- |
| scale 指数 | `floor(log2(amax)) − emax_elem`，E4M3 的 emax_elem 为 8 | amax/448 向上取整到 2 的幂 |
| 后果 | scale 可能是 TE 的一半，块内大值超过 448 后被饱和截断 | 从不截断 amax |
| 元素舍入 | 默认 `nearest`，即四舍五入（远离零）；`even` 才是 RNE | RNE |
| 饱和 | `saturate_normals=True`，超出范围的值截到 ±448 | satfinite，截到 ±448 |
| Inf 和溢出 | shared_exp 超出范围时置 NaN，整块变 NaN | scale 饱和到 254，元素饱和到 ±448 |
| NaN | 经 `torch.max` 传播 | 行方向忽略、列方向传播 |
| 次正规 | `flush_fp32_subnorms` 可选，E4M3 次正规保留（`allow_denorm=True`） | 保留 E4M3 次正规 |

实测（1024×2048，块大小 32，`custom_cuda=False`），与 TE 语义的仿真对比：

- 高斯数据：约 51% 的块（33330/65536）的 amax 被截断，0.60% 的元素不同，相对误差 1.4e-2；
- 离群通道数据：1.07% 的元素不同，相对误差 3.4e-2；
- `round=nearest` 与 `round=even` 在这些数据上结果相同，因为恰好落在中点的情况很少。

结论：microxcaling 实现的是 OCP 规范里“floor 取 scale + 饱和”的变体，与 TE 的“向上取整、不截断”不同，所以不能作为 TE 的逐位参考，只能用于理解 shared exponent 和舍入的概念。

## TE 官方数值测试（v2.16 `tests/pytorch/test_numerics.py`）

- 获取方式：git 访问 GitHub 超时，但 `curl https://raw.githubusercontent.com/...` 可用，偶发 `curl: (56)` 时重试即可。`test_numerics.py` 和 `utils.py` 下载到 scratchpad 的 `te_tests/`，用一个 `conftest.py` 在收集用例前调用 `apply_sm120_mxfp8_patch()`，这样 `is_mxfp8_available()` 返回 True，`recipe0` 即为 `MXFP8BlockScaling`。
- 命令：`CUDA_VISIBLE_DEVICES=0 python -m pytest test_numerics.py -k "recipe0 or mxfp8" -o addopts=""`
- 结果：262 个通过，144 个跳过，48 个失败，共 49.6 秒。
  - 通过：打过补丁的模块全部通过，包括 `test_grouped_linear_accuracy`（192 个，GroupedLinear 与逐专家 `te.Linear` 对比，其中一部分要求 atol=0 逐位一致）、`test_padding_grouped_linear_accuracy`、`*_save_original_input`、`test_grouped_linear_accuracy_single_gemm`。
  - 跳过：`test_grouped_gemm_grouped_tensor_mxfp8` 共 144 个，需要 cuBLAS 13 的分组 GEMM。
  - 失败 1：42 个 GPT 整模型用例（`test_gpt_selective_activation_recompute`、`test_gpt_full_activation_recompute`、`test_gpt_fp8_parameters`）全部出自 `layernorm_mlp.py:1220` 的反向，报 `CUBLAS_STATUS_NOT_SUPPORTED`。原因是补丁没有覆盖 `LayerNormMLP`，属于预期内的失败。
  - 失败 2：6 个 `test_grouped_gemm_grouped_tensor_zero_work`，报 `nvte_get_grouped_gemm_setup_workspace_size requires cuBLAS 13`。这是 GroupedTensor 分组 GEMM 新接口，TE 2.16 的 `GroupedLinear` 模块默认不走它，补丁也没有覆盖。
- 仅在 scratch 中做的实验：给 `layernorm_mlp.general_gemm` 也打上同样的补丁后，GPT 相关 MXFP8 用例 78 个全部通过，用时 11.6 秒，覆盖了 selective 和 full activation recompute 的前后一致性，以及以 MXFP8 存储的模型参数。说明转置方案在完整的 TransformerLayer 和 recompute 路径中同样成立。没有把这一点写进 `sm120_mxfp8.py`，因为 Megatron 默认的 TE spec 不用 `LayerNormMLP`，任务也限定只改三个模块。如果阶段二需要，只要多加一行赋值。

## CI

- 文件头设置了 `NUM_GPUS = 1`。测试仅在 SM100（验证 TE 原生 MXFP8）和 SM120（验证 slime 补丁）上运行，其他硬件上全部跳过。
- 没有注册到 `pr-test.yml.j2` 的固定矩阵。仓库没有声明自托管 GPU 的机型，现有 GPU runner 也没有 SM100 或 SM120，注册后只会全部跳过。`run-ci-changed` 会按 `NUM_GPUS = 1` 调度改动过的测试文件。
- 在 sm120 上手动运行：`CUDA_VISIBLE_DEVICES=0 python tests/test_sm120_mxfp8_gemm.py`

## 命令与结果

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_sm120_mxfp8_gemm.py -q   # 112 passed in 13.05s
CUDA_VISIBLE_DEVICES=0 python tests/test_sm120_mxfp8_gemm.py                # 112 passed in 4.28s
CUDA_VISIBLE_DEVICES=0 python tests/test_mxfp8_weight_sync.py               # 阶段一回归
```

## 尚未覆盖（阶段二接入时需要补）

- 张量并行加序列并行：反向中的 MXFP8 all-gather（`distributed.py::_all_gather_mxfp8`），以及 gather 之后的列方向数据转置。单卡无法测试。
- `te.LayerNormMLP` 和 `ops/*` 的融合路径没有打补丁。Megatron 默认不使用这些路径，如果启用会直接在 cuBLAS 处报错，而不会静默出错。
- bias、`fuse_wgrad_accumulation`（`main_grad`）、activation recompute 和 CPU offload 路径没有测试。
- 性能：直接转置每个反向操作数仍要多一次 FP8 数据拷贝，scale 在 swizzle 情形下不需要拷贝。
