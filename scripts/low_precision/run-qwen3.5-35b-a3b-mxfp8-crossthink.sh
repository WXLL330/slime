#!/bin/bash

# CrossThink-mini GRPO for Qwen3.5-35B-A3B (FP8 checkpoint dequantized to BF16) on a single 8-GPU node,
# with MXFP8 W8A8 rollout (SGLang) and optional MXFP8 W8A8 training (Transformer Engine
# MXFP8BlockScaling). Same setup as run-qwen3.5-35b-a3b-mxfp8.sh, except data, reward and
# non-thinking mode follow /workspace/datasets/crossthink-mini/README.md.
#
# E2E comparison: run once with TRAIN_MXFP8=1 and once with TRAIN_MXFP8=0; each run saves to its own
# SAVE_PATH by default.
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
# The MXFP8 HF checkpoint that SGLang starts from (its quantization_config drives the MXFP8 weight
# sync) is built from BF16_HF_CKPT on the first run if MXFP8_HF_CKPT does not exist yet.
# Megatron loads the BF16 HF checkpoint directly (a torch_dist checkpoint works as well).

TRAIN_MXFP8=${TRAIN_MXFP8:-1}
DATA_DIR=${DATA_DIR:-/workspace/datasets/crossthink-mini}
BF16_HF_CKPT=${BF16_HF_CKPT:-/workspace/hf_models/Qwen3.5-35B-A3B-FP8-Dequant}
MXFP8_HF_CKPT=${MXFP8_HF_CKPT:-/workspace/hf_models/Qwen3.5-35B-A3B-FP8-Dequant-MXFP8}
if [ "${TRAIN_MXFP8}" = "1" ]; then
   SAVE_PATH=${SAVE_PATH:-./ckpt/qwen3.5-35b-a3b-crossthink-train-mxfp8}
else
   SAVE_PATH=${SAVE_PATH:-./ckpt/qwen3.5-35b-a3b-crossthink-train-bf16}
fi
SAVE_INTERVAL=${SAVE_INTERVAL:-20}  # 0 disables checkpointing
KEEP_CKPTS=${KEEP_CKPTS:-2}  # number of latest completed checkpoints kept on disk
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
# the dataset difficulty prefilter sampled with max_new_tokens=6144; truncated responses get reward 0
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-6144}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
RUN_EVAL=${RUN_EVAL:-1}
RECOMPUTE=${RECOMPUTE:-full}  # full or selective
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
WANDB_PROJECT=${WANDB_PROJECT:-slime-dev}
if [ "${TRAIN_MXFP8}" = "1" ]; then
   WANDB_GROUP=${WANDB_GROUP:-qwen3.5-35b-a3b-crossthink-train-mxfp8}
else
   WANDB_GROUP=${WANDB_GROUP:-qwen3.5-35b-a3b-crossthink-train-bf16}
fi

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
# SGLang engines and the router listen on the node IP; keep their health checks off the HTTP proxy
NODE_IP=$(hostname -I | awk '{print $1}')
export no_proxy="${no_proxy:+${no_proxy},}${NODE_IP}"
export NO_PROXY="${no_proxy}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

if [ ! -f "${MXFP8_HF_CKPT}/config.json" ]; then
   python3 "${SCRIPT_DIR}/../../tools/convert_hf_to_fp8.py" --strategy mxfp8 \
      --model-dir "${BF16_HF_CKPT}" --save-dir "${MXFP8_HF_CKPT}"
fi

CKPT_ARGS=(
   # MXFP8 HF checkpoint: SGLang loads it and its quantization_config drives MXFP8 weight sync
   --hf-checkpoint "${MXFP8_HF_CKPT}"
   # BF16 weights for Megatron (HF checkpoint loaded directly); set LOAD_PATH=${SAVE_PATH} to resume
   --load "${LOAD_PATH:-${BF16_HF_CKPT}}"
   --ref-load "${BF16_HF_CKPT}"
)
if [ "${SAVE_INTERVAL}" != "0" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_PATH}"
      --save-interval "${SAVE_INTERVAL}"
   )
fi

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/train.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   # crossthink_reward.py lives in DATA_DIR, which is put on PYTHONPATH below
   --custom-rm-path crossthink_reward.reward_func
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   # dynamic sampling: drop groups whose samples are all right or all wrong, keep sampling until
   # ROLLOUT_BATCH_SIZE groups with a learning signal are collected
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
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
      --eval-prompt-data crossthink-val "${DATA_DIR}/val.jsonl"
      --n-samples-per-eval-prompt 4
      --log-passrate
      --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
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

   --recompute-granularity "${RECOMPUTE}"

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384

   --transformer-impl transformer_engine
   --bf16
)
if [ "${RECOMPUTE}" = "full" ]; then
   PERF_ARGS+=(
      --recompute-method uniform
      --recompute-num-layers 1
   )
fi
if [ "${TRAIN_MXFP8}" = "1" ]; then
   PERF_ARGS+=(
      --fp8-format e4m3
      --fp8-recipe mxfp8
   )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF}"
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
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
)
# without WANDB_KEY, wandb falls back to the login in ~/.netrc
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # MXFP8 dense GEMM backend: flashinfer_cutlass or triton both work on SM120
   --sglang-fp8-gemm-backend flashinfer_cutlass
   # the SM120 MXFP8 MoE kernel lives in the Triton MoE runner
   --sglang-moe-runner-backend triton
   --use-rollout-routing-replay
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
# Ray kills a worker above this host-memory fraction (default 0.95); checkpoint saves of this
# model peak at roughly 250-350GB of extra host memory on top of the CPU-offloaded optimizer
export RAY_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Keep only the KEEP_CKPTS latest completed checkpoints. A checkpoint is complete once
# latest_checkpointed_iteration.txt points at it or past it, so one still being written is never touched.
prune_ckpts() {
   local tracker="${SAVE_PATH}/latest_checkpointed_iteration.txt"
   while sleep 300; do
      [ -f "${tracker}" ] || continue
      local latest=$((10#$(cat "${tracker}")))
      ls -d "${SAVE_PATH}"/iter_[0-9]* 2>/dev/null | sort | while read -r dir; do
         [ $((10#${dir##*iter_})) -le ${latest} ] && echo "${dir}"
      done | head -n -"${KEEP_CKPTS}" | xargs -r rm -rf
   done
}
if [ "${SAVE_INTERVAL}" != "0" ]; then
   prune_ckpts &
   PRUNE_PID=$!
   trap 'kill ${PRUNE_PID}' EXIT
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${DATA_DIR}\",
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
   ${MISC_ARGS[@]} \
   ${EXTRA_ARGS}
