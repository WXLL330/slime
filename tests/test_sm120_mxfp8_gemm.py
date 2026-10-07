"""MXFP8 training GEMMs checked against a pure PyTorch emulation (``_mxfp8_emulation``).

SM120 runs TE's MXFP8 backward GEMMs through ``slime.backends.megatron_utils.sm120_mxfp8``
(NN/NT rewritten as TN on transposed column-wise copies). Without SM100 hardware
there is no native result to compare with, so the reference is the definition of
an MXFP8 GEMM: operands quantized along the reduction dimension, exact
accumulation. On SM100 the same tests check TE's native MXFP8 path.

Layer 1 - operands: the emulated quantizer and the transposed copies are
bit-identical to TE's quantization.
Layer 2 - single GEMMs (fprop TN, dgrad NN, wgrad NT): within FP32 accumulation
error of the reference, and far from a reference quantized along the wrong
dimension.
Layer 3 - te.Linear / te.LayerNormLinear / te.GroupedLinear forward and
backward against an emulated autograd of the same three GEMMs.

Error metric: the largest row-wise relative error. A Frobenius norm over the
whole output is dominated by outlier rows and hides errors on the small values
that a wrong block direction flushes to zero.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

NUM_GPUS = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _mxfp8_emulation as emu  # noqa: E402

CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
IS_SM120 = CAPABILITY is not None and CAPABILITY[0] == 12
IS_SM100 = CAPABILITY is not None and CAPABILITY[0] == 10
pytestmark = pytest.mark.skipif(
    not (IS_SM120 or IS_SM100), reason="MXFP8 GEMMs need SM100 (native TE) or SM120 (slime patch)"
)

if IS_SM120 or IS_SM100:
    te = pytest.importorskip("transformer_engine.pytorch")
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm
    import transformer_engine.pytorch.tensor.mxfp8_tensor as te_mxfp8
    from transformer_engine.common.recipe import MXFP8BlockScaling

    from slime.backends.megatron_utils import sm120_mxfp8

    if IS_SM120:
        sm120_mxfp8.apply_sm120_mxfp8_patch()
    general_gemm = sm120_mxfp8.general_gemm if IS_SM120 else te_gemm.general_gemm

# FP32 accumulation over K <= 4096 terms: random-walk error ~ sqrt(K) * 2^-24 ~ 4e-6.
# Observed <= 1.2e-6; a wrong block direction gives >= 7e-3 on outlier data.
GEMM_TOL = 2e-5
# BF16 outputs are FP32 results rounded once: at most half a BF16 ulp (2^-8 relative)
# away from the exact reference.
BF16_MODULE_TOL = 2.0**-8 + GEMM_TOL

QWEN35_GEMM_SHAPES = {
    # name: (tokens M, in_features K, out_features N); token counts are 32- but not 128-aligned
    "attn_qkv_tp2": (4096, 2048, 4608),
    "attn_proj_tp2": (4064, 2048, 2048),
    "shared_expert_fc1": (160, 2048, 1024),
    "expert_fc1": (96, 2048, 1024),
    "expert_fc2": (4064, 512, 2048),
    "minimal": (32, 32, 32),
}


def _quantizer(rowwise=True, columnwise=True, optimize_for_gemm=False):
    quantizer = te_mxfp8.MXFP8Quantizer(
        fp8_dtype=te_mxfp8.tex.DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise
    )
    quantizer.optimize_for_gemm = optimize_for_gemm
    return quantizer


def _data(rows, cols, scale, outlier_dim, seed, factor=2.0**12, dtype=torch.bfloat16):
    """Gaussian data where every 97th slice along ``outlier_dim`` is ``factor`` larger.

    A block running along ``outlier_dim`` mixes one outlier with 31 small values,
    pushing them into E4M3 subnormals; blocks along the other dim are unaffected.
    ``outlier_dim=None`` gives plain Gaussian data.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t = torch.randn(rows, cols, device="cuda", generator=gen) * scale
    if outlier_dim == 0:
        t[::97, :] *= factor
    elif outlier_dim == 1:
        t[:, ::97] *= factor
    return t.to(dtype)


def _row_rel_err(actual, expected):
    actual, expected = actual.double(), expected.double()
    norms = expected.norm(dim=-1)
    keep = norms > 0
    assert keep.any()
    assert torch.all((actual - expected).norm(dim=-1)[~keep] == 0)
    return ((actual - expected).norm(dim=-1)[keep] / norms[keep]).max().item()


def _fq(x, dim):
    return emu.fake_quantize(x, dim)


# ----------------------------------------------------------------------------- layer 1


def _edge_case_inputs():
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=gen).to(dtype)

    cases = {
        "gaussian": randn(256, 512),
        "fp32_input": randn(256, 512, dtype=torch.float32),
        "small_magnitude": (randn(256, 512, dtype=torch.float32) * 1e-3).bfloat16(),
        "outlier_channels": _data(256, 512, 1.0, 1, seed=1),
        "outlier_tokens": _data(256, 512, 1.0, 0, seed=2),
        "unaligned_to_128": randn(96, 160),
        "single_block": randn(32, 32),
        "large_tensor": randn(4064, 2048),
    }
    t = randn(256, 512)
    t[:32, :32] = 0
    cases["zero_block"] = t
    t = randn(256, 512)
    t[0, 0], t[1, 1], t[2, 64] = 448.0, 449.0, 2.0**126
    cases["range_limits"] = t
    t = randn(256, 512)
    t[5, 5] = 1e-38
    t[40:72, 100:132] *= 1e-36
    cases["denormals"] = t
    for name, value in (("pos_inf", float("inf")), ("neg_inf", float("-inf")), ("nan", float("nan"))):
        t = randn(256, 512)
        t[3, 3] = value
        cases[name] = t
    return cases


EDGE_CASES = _edge_case_inputs() if (IS_SM120 or IS_SM100) else {}


@pytest.mark.parametrize("case", sorted(EDGE_CASES))
def test_emulated_quantize_matches_te(case):
    x = EDGE_CASES[case]
    rows, cols = x.shape
    q = _quantizer()(x)
    row_data, row_scale = emu.quantize(x, dim=1)
    col_data, col_scale = emu.quantize(x, dim=0)
    assert torch.equal(row_data.view(torch.uint8), q._rowwise_data.view(torch.uint8))
    assert torch.equal(row_scale, q._rowwise_scale_inv[:rows, : cols // 32])
    assert torch.equal(col_data.view(torch.uint8), q._columnwise_data.view(torch.uint8))
    assert torch.equal(col_scale, q._columnwise_scale_inv[: rows // 32, :cols])


@pytest.mark.parametrize("optimize_for_gemm", [False, True], ids=["plain_scales", "swizzled_scales"])
@pytest.mark.parametrize("shape", [(4096, 2048), (4064, 512), (96, 160), (32, 32)])
def test_transposed_copy_is_rowwise_quantization_of_transpose(shape, optimize_for_gemm):
    x = _data(*shape, 1.0, 1, seed=3)
    rows, cols = shape
    transposed = sm120_mxfp8.transpose_columnwise_to_rowwise(_quantizer(optimize_for_gemm=optimize_for_gemm)(x))
    expected = _quantizer(columnwise=False, optimize_for_gemm=optimize_for_gemm)(x.t().contiguous())

    assert transposed._with_gemm_swizzled_scales == optimize_for_gemm
    assert torch.equal(transposed._rowwise_data.view(torch.uint8), expected._rowwise_data.view(torch.uint8))
    if optimize_for_gemm:
        # TE fills swizzled padding deterministically, so the whole buffer matches.
        assert torch.equal(transposed._rowwise_scale_inv, expected._rowwise_scale_inv)
    else:
        # Plain scale padding is uninitialized memory and never read by the GEMM.
        assert transposed._rowwise_scale_inv.shape == expected._rowwise_scale_inv.shape
        valid = (slice(0, cols), slice(0, rows // 32))
        assert torch.equal(transposed._rowwise_scale_inv[valid], expected._rowwise_scale_inv[valid])
        emu_data, emu_scale = emu.quantize(x, dim=0)
        assert torch.equal(transposed._rowwise_data.view(torch.uint8), emu_data.t().contiguous().view(torch.uint8))
        assert torch.equal(transposed._rowwise_scale_inv[valid], emu_scale.t())


def _rowwise_values(tensor, rows, cols):
    data = tensor._rowwise_data.view(torch.float8_e4m3fn)
    return emu.dequantize(data, tensor._rowwise_scale_inv[:rows, : cols // 32], dim=1, dtype=torch.float64)


def test_transpose_matches_dequantize_requantize_prototype():
    """The direct transpose reproduces the values of the first prototype (dequantize
    the column-wise copy, transpose, re-quantize row-wise) on a column-wise-only
    tensor, and the GEMM output bit for bit.

    The encodings differ in ~1% of blocks: when a block amax rounds down to
    224 * scale, re-quantization picks half the scale and doubles the E4M3 codes.
    The prototype was only value-exact here: ``dequantize()`` prefers row-wise data,
    so on a tensor carrying both copies it re-quantized values blocked along the
    wrong dimension.
    """
    x = _data(1024, 2048, 0.02, 1, seed=4)
    colwise_only = _quantizer(rowwise=False)(x)
    new = sm120_mxfp8.transpose_columnwise_to_rowwise(colwise_only)
    old = _quantizer(columnwise=False)(colwise_only.dequantize(dtype=torch.bfloat16).t().contiguous())

    assert torch.equal(_rowwise_values(new, 2048, 1024), _rowwise_values(old, 2048, 1024))
    other = _quantizer(columnwise=False)(_data(512, 1024, 1.0, None, seed=5))
    out_new = te_gemm.general_gemm(new, other, out_dtype=torch.float32, layout="TN")[0]
    out_old = te_gemm.general_gemm(old, other, out_dtype=torch.float32, layout="TN")[0]
    assert torch.equal(out_new, out_old)


# ----------------------------------------------------------------------------- layer 2


def _gemm_case(layout, shape, outliers, swizzled, seed):
    """Run one TE GEMM and build its emulated reference and a wrong-direction reference.

    TE computes D = op(B) @ op(A) with op(A) = A^T for "T" and op(B) = B^T for "T".
    Outliers sit along each operand's non-reduction dim, the case a wrong block
    direction gets visibly wrong.
    """
    m, k, n = shape
    act_q = _quantizer(optimize_for_gemm=swizzled)
    weight_q = _quantizer()

    def data(rows, cols, scale, reduction_dim, seed_offset):
        outlier_dim = (1 - reduction_dim) if outliers else None
        return _data(rows, cols, scale, outlier_dim, seed=seed + seed_offset)

    if layout == "TN":  # fprop: Y = X @ W^T
        x, w = data(m, k, 1.0, 1, 0), data(n, k, 0.02, 1, 1)
        out = general_gemm(weight_q(w), act_q(x), out_dtype=torch.float32, layout="TN")[0]
        return out, _fq(x, 1) @ _fq(w, 1).t(), _fq(x, 0) @ _fq(w, 0).t()
    if layout == "NN":  # dgrad: dX = dY @ W
        dy, w = data(m, n, 1e-3, 1, 2), data(n, k, 0.02, 0, 3)
        out = general_gemm(weight_q(w), act_q(dy), out_dtype=torch.float32, layout="NN")[0]
        return out, _fq(dy, 1) @ _fq(w, 0), _fq(dy, 0) @ _fq(w, 1)
    assert layout == "NT"  # wgrad: dW = dY^T @ X
    dy, x = data(m, n, 1e-3, 0, 4), data(m, k, 1.0, 0, 5)
    out = general_gemm(act_q(x), act_q(dy), out_dtype=torch.float32, layout="NT")[0]
    return out, _fq(dy, 0).t() @ _fq(x, 0), _fq(dy, 1).t() @ _fq(x, 1)


@pytest.mark.parametrize("swizzled", [False, True], ids=["plain_act_scales", "swizzled_act_scales"])
@pytest.mark.parametrize("outliers", [False, True], ids=["gaussian", "outliers"])
@pytest.mark.parametrize("shape_name", sorted(QWEN35_GEMM_SHAPES))
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"], ids=["fprop_TN", "dgrad_NN", "wgrad_NT"])
def test_gemm_matches_emulation(layout, shape_name, outliers, swizzled):
    out, reference, _ = _gemm_case(layout, QWEN35_GEMM_SHAPES[shape_name], outliers, swizzled, seed=10)
    assert _row_rel_err(out, reference) <= GEMM_TOL


@pytest.mark.parametrize("shape_name", ["attn_qkv_tp2", "expert_fc1", "expert_fc2"])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"], ids=["fprop_TN", "dgrad_NN", "wgrad_NT"])
def test_gemm_tolerance_rejects_wrong_block_direction(layout, shape_name):
    """Negative control: blocking an operand along its non-reduction dim (the classic
    mistake of using the row-wise copy where the column-wise one is needed) lands
    orders of magnitude outside ``GEMM_TOL``, so the check above would catch it."""
    out, reference, wrong_reference = _gemm_case(
        layout, QWEN35_GEMM_SHAPES[shape_name], outliers=True, swizzled=False, seed=20
    )
    assert _row_rel_err(out, reference) <= GEMM_TOL
    assert _row_rel_err(out, wrong_reference) >= 1e-3
    assert _row_rel_err(wrong_reference, reference) >= 100 * GEMM_TOL


# ----------------------------------------------------------------------------- layer 3


def _module_tol(dtype):
    return GEMM_TOL if dtype == torch.float32 else BF16_MODULE_TOL


def _linear_reference(x, w, dy):
    """Emulated MXFP8 autograd of y = x @ w^T with TE's quantization directions."""
    y = _fq(x, 1) @ _fq(w, 1).t()
    dx = _fq(dy, 1) @ _fq(w, 0)
    dw = _fq(dy, 0).t() @ _fq(x, 0)
    return y, dx, dw


def _assert_quantization_active(actual, x, w):
    """Sanity check against an unquantized product: close, but not identical, so the
    MXFP8 path really ran instead of a silent BF16 fallback."""
    unquantized = x.double() @ w.double().t()
    rel = ((actual.double() - unquantized).norm() / unquantized.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(actual.double().flatten(), unquantized.flatten(), dim=0).item()
    assert 1e-3 < rel < 0.1
    assert cosine > 0.999


def _run_te(module, inputs, dy):
    with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
        y = module(*inputs)
    y.backward(dy)
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_linear_matches_emulation(dtype):
    m, k, n = 4064, 2048, 1024
    linear = te.Linear(k, n, bias=False, params_dtype=dtype).cuda()
    with torch.no_grad():
        linear.weight.copy_(_data(n, k, 0.02, 1, seed=30, factor=2.0**6))
    x = _data(m, k, 1.0, 1, seed=31, factor=2.0**6, dtype=dtype).requires_grad_()
    dy = _data(m, n, 1e-3, None, seed=32, dtype=dtype)

    y = _run_te(linear, (x,), dy)

    w = linear.weight.detach()
    ref_y, ref_dx, ref_dw = _linear_reference(x.detach(), w, dy)
    tol = _module_tol(dtype)
    assert _row_rel_err(y, ref_y) <= tol
    assert _row_rel_err(x.grad, ref_dx) <= tol
    assert _row_rel_err(linear.weight.grad, ref_dw) <= tol
    _assert_quantization_active(y.detach(), x.detach(), w)


def _norm_reference(x, gamma, beta, normalization, zero_centered_gamma, eps):
    if zero_centered_gamma:
        gamma = gamma + 1
    if normalization == "LayerNorm":
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "normalization,zero_centered_gamma",
    [("LayerNorm", False), ("RMSNorm", True)],
    ids=["layernorm", "rmsnorm_zero_centered"],
)
def test_layernorm_linear_matches_emulation(normalization, zero_centered_gamma, dtype):
    m, k, n, eps = 4064, 2048, 1024, 1e-6
    # return_layernorm_output exposes the normalized activation TE quantizes; it leaves
    # every output and gradient bit-identical to the default path.
    module = te.LayerNormLinear(
        k,
        n,
        bias=False,
        normalization=normalization,
        zero_centered_gamma=zero_centered_gamma,
        eps=eps,
        params_dtype=dtype,
        return_layernorm_output=True,
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(40)
    with torch.no_grad():
        module.weight.copy_(_data(n, k, 0.02, 1, seed=41, factor=2.0**6))
        module.layer_norm_weight.copy_(
            0.1 * torch.randn(k, device="cuda", generator=gen) + (0 if zero_centered_gamma else 1)
        )
        if normalization == "LayerNorm":
            module.layer_norm_bias.copy_(0.1 * torch.randn(k, device="cuda", generator=gen))
    x = (torch.randn(m, k, device="cuda", generator=gen) * 3 + 1).to(dtype).requires_grad_()
    dy = _data(m, n, 1e-3, None, seed=42, dtype=dtype)

    with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
        y, ln_out = module(x)
    y.backward(dy)

    # The GEMM reference starts from TE's own normalized activation: rebuilding it in
    # FP64 moves FP32 values by an ulp, and a value on an E4M3 rounding boundary then
    # flips by a whole E4M3 step. The norm itself is checked against FP64 separately,
    # with the dgrad GEMM output rounded to the activation dtype as TE does.
    w = module.weight.detach()
    x64 = x.detach().double().requires_grad_()
    gamma64 = module.layer_norm_weight.detach().double().requires_grad_()
    beta64 = module.layer_norm_bias.detach().double().requires_grad_() if normalization == "LayerNorm" else None
    h64 = _norm_reference(x64, gamma64, beta64, normalization, zero_centered_gamma, eps)
    h = ln_out.detach()
    ref_y, ref_dh, ref_dw = _linear_reference(h, w, dy)
    h64.backward(ref_dh.to(dtype).double())

    tol = _module_tol(dtype)
    assert _row_rel_err(h, h64.detach()) <= tol
    assert _row_rel_err(y, ref_y) <= tol
    assert _row_rel_err(module.weight.grad, ref_dw) <= tol
    assert _row_rel_err(x.grad, x64.grad) <= tol
    assert _row_rel_err(module.layer_norm_weight.grad[None], gamma64.grad[None]) <= tol
    if beta64 is not None:
        assert _row_rel_err(module.layer_norm_bias.grad[None], beta64.grad[None]) <= tol
    _assert_quantization_active(y.detach(), h, w)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_grouped_linear_matches_emulation(dtype):
    k, n = 2048, 1024
    m_splits = [64, 0, 96, 32, 160, 0, 32, 128]  # uneven, 32-aligned, two experts without tokens
    num_experts = len(m_splits)
    module = te.GroupedLinear(num_experts, k, n, bias=False, params_dtype=dtype).cuda()
    weights = [getattr(module, f"weight{i}") for i in range(num_experts)]
    with torch.no_grad():
        for i, weight in enumerate(weights):
            weight.copy_(_data(n, k, 0.02, 1, seed=50 + i, factor=2.0**6))
    x = _data(sum(m_splits), k, 1.0, 1, seed=60, factor=2.0**6, dtype=dtype).requires_grad_()
    dy = _data(sum(m_splits), n, 1e-3, None, seed=61, dtype=dtype)

    y = _run_te(module, (x, m_splits), dy)

    tol = _module_tol(dtype)
    offsets = [0]
    for count in m_splits:
        offsets.append(offsets[-1] + count)
    for i, weight in enumerate(weights):
        start, end = offsets[i], offsets[i + 1]
        if start == end:
            assert weight.grad is None or torch.count_nonzero(weight.grad) == 0
            continue
        ref_y, ref_dx, ref_dw = _linear_reference(x.detach()[start:end], weight.detach(), dy[start:end])
        assert _row_rel_err(y[start:end], ref_y) <= tol
        assert _row_rel_err(x.grad[start:end], ref_dx) <= tol
        assert _row_rel_err(weight.grad, ref_dw) <= tol


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
