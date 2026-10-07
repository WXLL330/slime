#!/bin/bash

# MXFP8 W8A8 rollout (SGLang) GRPO for Qwen3.5-35B-A3B on a single 8-GPU node, with optional
# MXFP8 W8A8 training (Transformer Engine MXFP8BlockScaling).
#
# Rollout runs MXFP8 (e4m3 weights/activations, UE8M0 scale per 1x32 block) on the same layers as the
# official Qwen3.5 FP8 checkpoints: full attention, linear-attention in_proj_qkv/in_proj_z/out_proj,
# shared experts and routed experts. Linear-attention in_proj_a/in_proj_b/conv1d, routers, gates,
# embeddings and lm_head stay BF16. slime re-quantizes the BF16 training weights with the same
# quantizer on every weight sync.
#
# TRAIN_MXFP8=1 also trains with MXFP8 (--fp8-format e4m3 --fp8-recipe mxfp8) on the same layers, so
# training and rollout quantize identically; TRAIN_MXFP8=0 trains in BF16.
#
# On SM120 (RTX PRO 6000 / RTX 50):
#   - SGLang needs docker/patch/latest/sglang-sm120-mxfp8.patch (grouped MXFP8 MoE kernel in the
#     Triton MoE runner).
#   - slime patches TE at startup so its MXFP8 backward GEMMs run on SM120
#     (slime/backends/megatron_utils/sm120_mxfp8.py).
#
# Before the first run, build the MXFP8 HF checkpoint that SGLang starts from (its quantization_config
# drives the MXFP8 weight sync):
#   python tools/convert_hf_to_fp8.py --strategy mxfp8 \
#       --model-dir /workspace/hf_models/Qwen3.5-35B-A3B \
#       --save-dir /workspace/hf_models/Qwen3.5-35B-A3B-MXFP8-slime-v2
# Megatron loads the BF16 HF checkpoint directly (a torch_dist checkpoint works as well).

TRAIN_MXFP8=${TRAIN_MXFP8:-1}
BF16_HF_CKPT=${BF16_HF_CKPT:-/workspace/hf_models/Qwen3.5-35B-A3B}
MXFP8_HF_CKPT=${MXFP8_HF_CKPT:-/workspace/hf_models/Qwen3.5-35B-A3B-MXFP8-slime-v2}
SAVE_PATH=${SAVE_PATH:-./ckpt/qwen3.5-35b-a3b-mxfp8-dapo-math-17k}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}  # 0 disables checkpointing
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
RUN_EVAL=${RUN_EVAL:-1}

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   # MXFP8 HF checkpoint: SGLang loads it and its quantization_config drives MXFP8 weight sync
   --hf-checkpoint "${MXFP8_HF_CKPT}"
   # BF16 weights for Megatron (HF checkpoint loaded directly)
   --load "${BF16_HF_CKPT}"
   --ref-load "${BF16_HF_CKPT}"
)
if [ "${SAVE_INTERVAL}" != "0" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_PATH}"
      --save-interval "${SAVE_INTERVAL}"
      --save-retain-interval 100
   )
fi

ROLLOUT_ARGS=(
   --prompt-data /workspace/datasets/dapo-math-17k-processed/dapo-math-17k.parquet
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=()
if [ "${RUN_EVAL}" = "1" ]; then
   EVAL_ARGS=(
      --eval-interval 20
      --eval-prompt-data aime /workspace/datasets/aime-2024-processed/aime-2024.parquet
      --n-samples-per-eval-prompt 4
      --log-passrate
      --eval-max-response-len 16384
      --eval-top-p 1
   )
fi

PERF_ARGS=(
   # num_query_groups=2, so TP must be <= 2
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384

   --transformer-impl transformer_engine
   --bf16
)
if [ "${TRAIN_MXFP8}" = "1" ]; then
   PERF_ARGS+=(
      --fp8-format e4m3
      --fp8-recipe mxfp8
   )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   # keep half of the optimizer on GPU and use bf16 adam moments to fit the ~950GB host memory limit
   --optimizer-offload-fraction 0.5
   --exp-avg-dtype bf16
   --exp-avg-sq-dtype bf16
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-35B-A3B-mxfp8
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # MXFP8 dense GEMM backend: flashinfer_cutlass or triton both work on SM120
   --sglang-fp8-gemm-backend flashinfer_cutlass
   # the SM120 MXFP8 MoE kernel lives in the Triton MoE runner
   --sglang-moe-runner-backend triton
   # --use-rollout-routing-replay
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
