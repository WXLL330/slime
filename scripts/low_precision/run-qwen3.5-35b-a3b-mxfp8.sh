#!/bin/bash

# BF16 training + MXFP8 W8A8 rollout (SGLang) GRPO for Qwen3.5-35B-A3B on a single 8-GPU node.
#
# Rollout runs MXFP8 (e4m3 weights/activations, UE8M0 scale per 1x32 block) for full attention,
# shared experts and routed experts; linear attention, routers, gates, embeddings and lm_head stay
# BF16 because they are plain BF16 layers in training. slime re-quantizes the BF16 training weights
# with the same quantizer on every weight sync.
#
# On SM120 (RTX PRO 6000 / RTX 50) SGLang needs docker/patch/latest/sglang-sm120-mxfp8.patch, which
# adds a grouped MXFP8 MoE kernel; the Triton MoE runner is used and both GEMMs stay MXFP8.
#
# Before the first run:
#   1. Build the MXFP8 HF checkpoint that SGLang starts from (its quantization_config drives the sync):
#        python tools/convert_hf_to_fp8.py --strategy mxfp8 \
#            --model-dir /workspace/hf_models/Qwen3.5-35B-A3B \
#            --save-dir /workspace/hf_models/Qwen3.5-35B-A3B-MXFP8-slime
#   2. Convert the BF16 HF checkpoint to torch_dist for Megatron:
#        source scripts/models/qwen3.5-35B-A3B.sh
#        PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
#            ${MODEL_ARGS[@]} \
#            --hf-checkpoint /workspace/hf_models/Qwen3.5-35B-A3B \
#            --save /workspace/rl-project/slime_dev/slime/mg_models/qwen3.5-35b-a3b-mg

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

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
   --hf-checkpoint /workspace/hf_models/Qwen3.5-35B-A3B-MXFP8-slime
   # torch_dist converted from the BF16 Qwen3.5-35B-A3B checkpoint
   --ref-load /workspace/rl-project/slime_dev/slime/mg_models/qwen3.5-35b-a3b-mg
   # --load ./ckpt/qwen3.5-35b-a3b-mxfp8-dapo-math-17k
   --save ./ckpt/qwen3.5-35b-a3b-mxfp8-dapo-math-17k
   --save-interval 20
   --save-retain-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /workspace/datasets/dapo-math-17k-processed/dapo-math-17k.parquet
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 200
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /workspace/datasets/aime-2024-processed/aime-2024.parquet
   --n-samples-per-eval-prompt 4
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
)

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

   # training stays BF16 in this stage; MXFP8 training is stage 2
   --transformer-impl transformer_engine
   --bf16
)

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
