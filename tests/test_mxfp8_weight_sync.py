"""CPU tests for MXFP8 rollout weight sync.

Pins two contracts:
- ``quantize_params`` with ``quant_method == "mxfp8"`` quantizes exactly the
  attention/MLP/expert projections and keeps linear attention, routers and
  gates in BF16, emitting ``weight`` + ``weight_scale_inv`` pairs.
- ``tools/convert_hf_to_fp8.py --strategy mxfp8`` quantizes the same HF tensors,
  so the checkpoint SGLang starts from and the weights slime pushes later agree
  on which modules are MXFP8 (SGLang's ``ignored_layers`` comes from the tool).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fake_mxfp8_group_quantize(x):
    n, k = x.shape
    assert k % 32 == 0
    return x.to(torch.float8_e4m3fn), torch.full((n, k // 32), 127, dtype=torch.uint8)


@pytest.fixture(scope="module")
def mxfp8_modules():
    # Mount megatron_utils without running its runtime initialization, and stub
    # the SGLang dependency module so the test needs neither Megatron nor SGLang.
    saved = {
        name: sys.modules.get(name)
        for name in ("slime.backends.megatron_utils", "slime.backends.megatron_utils.sglang")
    }
    megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    megatron_utils.__path__ = [str(REPO_ROOT / "slime/backends/megatron_utils")]
    sys.modules["slime.backends.megatron_utils"] = megatron_utils
    sglang_stub = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_stub.mxfp8_group_quantize = _fake_mxfp8_group_quantize
    sys.modules["slime.backends.megatron_utils.sglang"] = sglang_stub

    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf

    spec = importlib.util.spec_from_file_location("convert_hf_to_fp8", REPO_ROOT / "tools" / "convert_hf_to_fp8.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    yield convert_to_hf, converter

    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


QUANT_CONFIG = {"quant_method": "mxfp8", "activation_scheme": "dynamic", "weight_block_size": [1, 32]}

ARGS = types.SimpleNamespace(
    vocab_size=64,
    hidden_size=64,
    num_attention_heads=4,
    num_query_groups=2,
    kv_channels=16,
    q_lora_rank=None,
)

# (megatron param name, shape, whether it must be MXFP8 on the rollout side)
PARAMS = [
    ("module.module.decoder.layers.3.self_attention.linear_qkv.weight", (2 * 2 * 16 * 2 + 2 * 2 * 16, 64), True),
    ("module.module.decoder.layers.3.self_attention.linear_proj.weight", (64, 64), True),
    ("module.module.decoder.layers.0.mlp.experts.linear_fc1.weight5", (2 * 32, 64), True),
    ("module.module.decoder.layers.0.mlp.experts.linear_fc2.weight5", (64, 32), True),
    ("module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight", (2 * 32, 64), True),
    ("module.module.decoder.layers.0.mlp.shared_experts.linear_fc2.weight", (64, 32), True),
    ("module.module.decoder.layers.0.mlp.shared_experts.gate_weight", (1, 64), False),
    ("module.module.decoder.layers.0.mlp.router.weight", (8, 64), False),
    ("module.module.decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight", (96, 64), False),
    ("module.module.decoder.layers.0.self_attention.linear_attn.in_proj_z.weight", (64, 64), False),
    ("module.module.decoder.layers.0.self_attention.linear_attn.in_proj_b.weight", (4, 64), False),
    ("module.module.decoder.layers.0.self_attention.linear_attn.in_proj_a.weight", (4, 64), False),
    ("module.module.decoder.layers.0.self_attention.linear_attn.out_proj.weight", (64, 64), False),
    ("module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight1", (64, 32), True),
    ("module.module.mtp.layers.0.eh_proj.weight", (64, 128), False),
    ("module.module.output_layer.weight", (64, 64), False),
    ("module.module.embedding.word_embeddings.weight", (64, 64), False),
]


@pytest.mark.unit
@pytest.mark.parametrize("megatron_name,shape,quantized", PARAMS)
def test_mxfp8_sync_matches_checkpoint_layout(mxfp8_modules, megatron_name, shape, quantized):
    convert_to_hf, converter = mxfp8_modules
    param = torch.randn(shape, dtype=torch.bfloat16)

    hf_params = dict(convert_to_hf(ARGS, "qwen3_5", megatron_name, param, QUANT_CONFIG))

    weight_names = [name for name in hf_params if not name.endswith("_scale_inv")]
    assert weight_names
    for name in weight_names:
        scale_name = name.replace(".weight", ".weight_scale_inv")
        if quantized:
            assert hf_params[name].dtype == torch.float8_e4m3fn
            assert hf_params[scale_name].dtype == torch.uint8
            assert hf_params[scale_name].shape == (hf_params[name].shape[0], hf_params[name].shape[1] // 32)
        else:
            assert hf_params[name].dtype == torch.bfloat16
            assert scale_name not in hf_params
        # The offline converter must make the same decision, otherwise SGLang
        # allocates a BF16 param for a tensor slime pushes as FP8 (or vice versa).
        assert converter.should_quantize(name, "mxfp8") == quantized, name


@pytest.mark.unit
def test_split_fused_experts_keeps_gate_first(mxfp8_modules):
    _, converter = mxfp8_modules
    num_experts, inter, hidden = 3, 4, 8
    gate_up = torch.randn(num_experts, 2 * inter, hidden)
    down = torch.randn(num_experts, hidden, inter)
    prefix = "model.language_model.layers.0.mlp.experts"

    split = converter.split_fused_experts(
        {f"{prefix}.gate_up_proj": gate_up, f"{prefix}.down_proj": down, "other.weight": down[0]}
    )

    assert set(split) == {"other.weight"} | {
        f"{prefix}.{e}.{proj}.weight" for e in range(num_experts) for proj in ("gate_proj", "up_proj", "down_proj")
    }
    for e in range(num_experts):
        assert torch.equal(split[f"{prefix}.{e}.gate_proj.weight"], gate_up[e, :inter])
        assert torch.equal(split[f"{prefix}.{e}.up_proj.weight"], gate_up[e, inter:])
        assert torch.equal(split[f"{prefix}.{e}.down_proj.weight"], down[e])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
