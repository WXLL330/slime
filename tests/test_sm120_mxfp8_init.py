"""slime's SM120 MXFP8 setup in the Megatron init (``initialize._setup_sm120_mxfp8``).

TE's cross-step quantized weight cache is turned off when the trainer is offloaded between
steps, through Megatron's ``disable_parameter_transpose_cache`` config field, which
``core_transformer_config_from_args`` copies from ``args``.
"""

import dataclasses
from argparse import Namespace

import pytest

pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("megatron.training")

from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from slime.backends.megatron_utils import initialize, sm120_mxfp8  # noqa: E402


@pytest.fixture(autouse=True)
def _no_te_patch(monkeypatch):
    # The TE patch itself is covered by test_sm120_mxfp8_gemm.py; here only the setup logic matters.
    monkeypatch.setattr(sm120_mxfp8, "apply_sm120_mxfp8_patch", lambda: None)


@pytest.mark.parametrize("offload_train", [True, False])
def test_weight_cache_disabled_only_when_trainer_is_offloaded(offload_train):
    args = Namespace(fp8_param_gather=False, offload_train=offload_train)
    initialize._setup_sm120_mxfp8(args)
    assert getattr(args, "disable_parameter_transpose_cache", False) is offload_train


def test_megatron_only_args_without_offload_train():
    # Callers that parse only Megatron's arguments (like the parallel-consistency harness) lack offload_train.
    args = Namespace(fp8_param_gather=False)
    initialize._setup_sm120_mxfp8(args)
    assert not hasattr(args, "disable_parameter_transpose_cache")


def test_disable_flag_is_a_transformer_config_field():
    # core_transformer_config_from_args only copies args attributes that name a config field.
    assert "disable_parameter_transpose_cache" in {f.name for f in dataclasses.fields(TransformerConfig)}


def test_fp8_param_gather_is_rejected():
    with pytest.raises(ValueError, match="fp8-param-gather"):
        initialize._setup_sm120_mxfp8(Namespace(fp8_param_gather=True, offload_train=True))
