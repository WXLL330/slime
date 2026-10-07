"""SM120 MXFP8 training under TP/SP/EP/DP/CP matches the MXFP8 emulation.

For each Megatron parallel config, ``_sm120_mxfp8_parallel_harness.py`` runs one step of a
tiny Qwen3.5 MoE (3 linear-attention + 1 full-attention layers, 16 experts) twice in one
process: real MXFP8 (TE + the slime SM120 patch) and a teacher-forced MXFP8 emulation
(same routing, same per-module inputs and grad outputs). Every TE module's output, input
grad and weight grad must then agree up to accumulation order on every rank.

Corrupting the SM120 transpose (scales shifted by one row, or swizzled scales transposed
as if they were plain) must make the comparison fail, which shows the check is sensitive.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine")
pytest.importorskip("megatron.core")

NUM_GPUS = 8

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "tests" / "_sm120_mxfp8_parallel_harness.py"

IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
pytestmark = pytest.mark.skipif(not IS_SM120, reason="the SM120 MXFP8 patch only runs on SM12x GPUs")

# Measured worst case over all configs is ~4e-4 (TE output/dgrad, BF16 outputs) and ~6e-7
# (TE weight grads accumulated in FP32 main_grad); a broken transpose gives ~1.
TOL = 1e-3

TINY_QWEN35_MOE = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "moe_intermediate_size": 512,
        "mtp_num_hidden_layers": 0,
        "num_attention_heads": 16,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {
        "model_type": "qwen3_5_moe",
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

MODEL_ARGS = [
    "--spec", "slime_plugins.models.qwen3_5", "get_qwen3_5_spec",
    "--disable-bias-linear", "--qk-layernorm", "--group-query-attention",
    "--num-attention-heads", "16", "--num-query-groups", "2", "--kv-channels", "256",
    "--num-layers", "4", "--hidden-size", "2048", "--ffn-hidden-size", "512", "--use-gated-attention",
    "--normalization", "RMSNorm", "--apply-layernorm-1p", "--position-embedding-type", "rope",
    "--norm-epsilon", "1e-6", "--rotary-percent", "0.25", "--swiglu", "--untie-embeddings-and-output-weights",
    "--vocab-size", "248320", "--rotary-base", "10000000",
    "--moe-ffn-hidden-size", "512", "--moe-shared-expert-intermediate-size", "512",
    "--moe-router-score-function", "softmax", "--moe-token-dispatcher-type", "alltoall",
    "--moe-router-topk", "4", "--moe-layer-freq", "[1,1,1,1]", "--num-experts", "16", "--moe-grouped-gemm",
    "--moe-token-drop-policy", "probs", "--moe-router-dtype", "fp32", "--moe-permute-fusion",
    "--moe-aux-loss-coeff", "0", "--attention-output-gate", "--moe-shared-expert-gate",
    "--tokenizer-type", "NullTokenizer", "--transformer-impl", "transformer_engine", "--bf16",
    "--fp8-format", "e4m3", "--fp8-recipe", "mxfp8", "--attention-backend", "flash",
    "--attention-dropout", "0.0", "--hidden-dropout", "0.0", "--seq-length", "4096", "--global-batch-size", "64",
]  # fmt: skip

TP2_SP = ["--tensor-model-parallel-size", "2", "--sequence-parallel"]
PARALLEL_CONFIGS = {
    "single": (1, []),
    "tp2_sp": (2, TP2_SP),
    "dp2": (2, []),
    "cp2": (2, ["--context-parallel-size", "2"]),
    "ep4": (4, ["--expert-model-parallel-size", "4", "--expert-tensor-parallel-size", "1"]),
    "tp2_sp_ep4": (4, TP2_SP + ["--expert-model-parallel-size", "4", "--expert-tensor-parallel-size", "1"]),
    "tp2_sp_ep4_cp2": (
        8,
        TP2_SP
        + ["--expert-model-parallel-size", "4", "--expert-tensor-parallel-size", "1", "--context-parallel-size", "2"],
    ),
    "tp2_sp_ep8": (8, TP2_SP + ["--expert-model-parallel-size", "8", "--expert-tensor-parallel-size", "1"]),
}


def _run_harness(tmp_path, nproc, parallel_args, negative_control="none"):
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"needs {nproc} GPUs")
    checkpoint = tmp_path / "tiny_qwen35_moe"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(TINY_QWEN35_MOE))
    out = tmp_path / "report.json"
    env = dict(os.environ)
    env.update(
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        ENABLE_ROUTING_REPLAY="1",
        ROUTING_REPLAY_STAGE="record",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}", str(HARNESS),
        *MODEL_ARGS, *parallel_args,
        "--hf-checkpoint", str(checkpoint), "--out", str(out), "--negative-control", negative_control,
    ]  # fmt: skip
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    return json.loads(out.read_text())


def _worst(report, key, reduce=max):
    return reduce(rank[key] for rank in report["ranks"])


@pytest.mark.parametrize("config", list(PARALLEL_CONFIGS))
def test_real_mxfp8_matches_emulation(tmp_path, config):
    nproc, parallel_args = PARALLEL_CONFIGS[config]
    report = _run_harness(tmp_path, nproc, parallel_args)

    for rank in report["ranks"]:
        assert rank["finite"]
        assert rank["local_tokens"] % 32 == 0
        # Every TE GEMM runs on MXFP8 operands in the real step and on emulated BF16 operands
        # in the forced step, one for one.
        real = {k.rsplit(":", 1)[0]: v for k, v in rank["calls_real"].items() if k.endswith(":mxfp8")}
        emulated = {k.rsplit(":", 1)[0]: v for k, v in rank["calls_emulated"].items() if k.endswith(":bf16")}
        assert sum(real.values()) == sum(rank["calls_real"].values()) > 0
        assert real == emulated == {k.rsplit(":", 1)[0]: v for k, v in rank["calls_emulated"].items()}

    assert _worst(report, "routing_agreement", min) == 1.0
    assert _worst(report, "te_output_rel") < TOL
    assert _worst(report, "te_dgrad_rel") < TOL
    assert _worst(report, "te_wgrad_rel") < TOL


@pytest.mark.parametrize("negative_control", ["scale_roll", "swizzle"])
def test_broken_sm120_transpose_is_detected(tmp_path, negative_control):
    report = _run_harness(tmp_path, 1, [], negative_control)
    # Only the backward NN/NT GEMMs go through the transpose; the forward stays exact.
    assert _worst(report, "te_output_rel") < TOL
    assert _worst(report, "te_wgrad_rel") > 0.1
    if negative_control == "scale_roll":
        assert _worst(report, "te_dgrad_rel") > 0.1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
