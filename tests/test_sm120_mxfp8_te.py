from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("transformer_engine")

from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch import LayerNormLinear, LayerNormMLP, Linear, autocast
from transformer_engine.pytorch.custom_recipes.quantization import GEMMType
from transformer_engine.pytorch.quantization import QuantizerRole

from slime.backends.megatron_utils.sm120_mxfp8 import (
    SM120MXFP8Quantizer,
    quantizer_factory,
)

NUM_GPUS = 1


def _require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 MXFP8 tests require a compute capability 12.0 GPU")


@pytest.fixture(autouse=True)
def require_sm120():
    _require_sm120()


@pytest.fixture
def gemm_calls(monkeypatch):
    calls = []
    original = SM120MXFP8Quantizer.qgemm

    def record(self, *args, **kwargs):
        calls.append(kwargs["gemm_type"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SM120MXFP8Quantizer, "qgemm", record)
    return calls


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _run_linear(m, n, k):
    torch.manual_seed(123)
    input_ref = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight_ref = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
    grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    input_mx = input_ref.detach().clone().requires_grad_()
    layer = Linear(k, n, bias=False, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        layer.weight.copy_(weight_ref)

    with autocast(enabled=True, recipe=CustomRecipe(qfactory=quantizer_factory)):
        output_mx = layer(input_mx)
    output_mx.backward(grad_output)

    output_ref = torch.nn.functional.linear(input_ref, weight_ref)
    output_ref.backward(grad_output)
    return output_mx, input_mx.grad, layer.weight.grad, output_ref, input_ref.grad, weight_ref.grad


@pytest.mark.parametrize("m", [128, 127])
def test_dense_forward_dgrad_wgrad_match_bf16_and_dispatch(m, gemm_calls):
    outputs = _run_linear(m=m, n=192, k=256)
    output_mx, dinput_mx, dweight_mx, output_ref, dinput_ref, dweight_ref = outputs

    assert output_mx.shape == output_ref.shape == (m, 192)
    assert dinput_mx.shape == dinput_ref.shape == (m, 256)
    assert dweight_mx.shape == dweight_ref.shape == (192, 256)
    assert output_mx.dtype == dinput_mx.dtype == dweight_mx.dtype == torch.bfloat16
    assert torch.isfinite(output_mx).all()
    assert torch.isfinite(dinput_mx).all()
    assert torch.isfinite(dweight_mx).all()
    assert _relative_error(output_mx, output_ref) < 0.06
    assert _relative_error(dinput_mx, dinput_ref) < 0.06
    assert _relative_error(dweight_mx, dweight_ref) < 0.06
    assert gemm_calls == [GEMMType.FPROP, GEMMType.DGRAD, GEMMType.WGRAD]


def test_qwen_mlp_shape_forward_backward(gemm_calls):
    outputs = _run_linear(m=32, n=9728, k=2560)
    output_mx, dinput_mx, dweight_mx, output_ref, dinput_ref, dweight_ref = outputs

    assert _relative_error(output_mx, output_ref) < 0.06
    assert _relative_error(dinput_mx, dinput_ref) < 0.06
    assert _relative_error(dweight_mx, dweight_ref) < 0.06
    assert gemm_calls == [GEMMType.FPROP, GEMMType.DGRAD, GEMMType.WGRAD]


def test_two_optimizer_steps_update_dense_weight(gemm_calls):
    torch.manual_seed(321)
    layer = Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="cuda")
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    initial_weight = layer.weight.detach().clone()

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        input_mx = torch.randn(63, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with autocast(enabled=True, recipe=CustomRecipe(qfactory=quantizer_factory)):
            output = layer(input_mx)
        output.float().square().mean().backward()
        optimizer.step()
        assert torch.isfinite(input_mx.grad).all()
        assert torch.isfinite(layer.weight).all()

    assert not torch.equal(layer.weight, initial_weight)
    assert gemm_calls == [
        GEMMType.FPROP,
        GEMMType.DGRAD,
        GEMMType.WGRAD,
        GEMMType.FPROP,
        GEMMType.DGRAD,
        GEMMType.WGRAD,
    ]


def test_cached_weight_supports_two_forwards_before_backward(gemm_calls):
    layer = Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="cuda")
    input_1 = torch.randn(63, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    input_2 = torch.randn(63, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with autocast(enabled=True, recipe=CustomRecipe(qfactory=quantizer_factory)):
        output_1 = layer(input_1, is_first_microbatch=True)
        output_2 = layer(input_2, is_first_microbatch=False)
    (output_1.float().square().mean() + output_2.float().square().mean()).backward()

    assert torch.isfinite(input_1.grad).all()
    assert torch.isfinite(input_2.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert gemm_calls == [
        GEMMType.FPROP,
        GEMMType.FPROP,
        GEMMType.DGRAD,
        GEMMType.WGRAD,
        GEMMType.DGRAD,
        GEMMType.WGRAD,
    ]


@pytest.mark.parametrize(
    ("module_cls", "module_args", "expected_shape", "expected_calls"),
    [
        (
            LayerNormLinear,
            (256, 192),
            (63, 192),
            [GEMMType.FPROP, GEMMType.DGRAD, GEMMType.WGRAD],
        ),
        (
            LayerNormMLP,
            (256, 512),
            (63, 256),
            [
                GEMMType.FPROP,
                GEMMType.FPROP,
                GEMMType.DGRAD,
                GEMMType.WGRAD,
                GEMMType.DGRAD,
                GEMMType.WGRAD,
            ],
        ),
    ],
)
def test_fused_dense_modules_forward_backward(module_cls, module_args, expected_shape, expected_calls, gemm_calls):
    module = module_cls(
        *module_args,
        bias=False,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    input_mx = torch.randn(63, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with autocast(enabled=True, recipe=CustomRecipe(qfactory=quantizer_factory)):
        output = module(input_mx)
    output.float().square().mean().backward()

    assert output.shape == expected_shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(input_mx.grad).all()
    assert gemm_calls == expected_calls


def test_dense_bias_backward(gemm_calls):
    layer = Linear(128, 128, bias=True, params_dtype=torch.bfloat16, device="cuda")
    input_mx = torch.randn(63, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with autocast(enabled=True, recipe=CustomRecipe(qfactory=quantizer_factory)):
        output = layer(input_mx)
    output.float().square().mean().backward()

    assert layer.bias.grad is not None
    assert torch.isfinite(layer.bias.grad).all()
    assert gemm_calls == [GEMMType.FPROP, GEMMType.DGRAD, GEMMType.WGRAD]


def test_grouped_linear_role_is_rejected():
    role = QuantizerRole(module_type="grouped_linear", tensor_type="weight", name="experts")
    with pytest.raises(NotImplementedError, match="grouped GEMM"):
        quantizer_factory(role)


def test_output_accumulation_is_rejected():
    quantizer = quantizer_factory(types.SimpleNamespace(module_type="linear"))
    with pytest.raises(NotImplementedError, match="gradient_accumulation_fusion"):
        quantizer.qgemm(
            torch.empty((32, 32), device="cuda", dtype=torch.float8_e4m3fn),
            torch.empty((32, 32), device="cuda", dtype=torch.float8_e4m3fn),
            types.SimpleNamespace(),
            torch.bfloat16,
            torch.empty((32, 1), device="cuda", dtype=torch.float8_e8m0fnu),
            torch.empty((32, 1), device="cuda", dtype=torch.float8_e8m0fnu),
            accumulate=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
