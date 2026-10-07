"""Pure PyTorch emulation of TransformerEngine's MXFP8 (E4M3 data, UE8M0 scales).

Used as the golden reference for the SM120 MXFP8 GEMM tests. It depends on
nothing but torch, and its quantization is bit-identical to TE's
``MXFP8Quantizer`` (verified in ``test_sm120_mxfp8_gemm.py``):

- Blocks are 32 consecutive elements along the quantized dimension (the GEMM
  reduction dimension). Both tensor dimensions must be multiples of 32.
- Scale: the block amax is multiplied by the FP32 constant ``1/448`` and rounded
  *up* to a power of two, so no element exceeds the E4M3 range after scaling.
  The biased exponent saturates at 254; an all-zero block gets exponent 0.
- Data: ``x * 2^(127 - e)`` cast to E4M3 with round-to-nearest-even and
  saturation to +-448 (``satfinite``), NaN stays NaN.
- NaN: TE's row-wise kernel takes the amax with ``fmaxf`` (NaN ignored) while its
  column-wise kernel propagates NaN into the scale (0xFF) and the whole block.

A GEMM reference is ``dequantize(quantize(a, dim_a)) @ dequantize(quantize(b, dim_b))``
evaluated in FP64, i.e. MXFP8 operands with exact accumulation.
"""

import torch

MXFP8_BLOCK = 32
E4M3_MAX = 448.0


def quantize(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor with 32-element blocks along ``dim``.

    Returns E4M3 data with the shape of ``x`` and uint8 biased exponents whose
    ``dim`` axis is 32x shorter.
    """
    assert x.dim() == 2 and dim in (0, 1)
    assert x.shape[0] % MXFP8_BLOCK == 0 and x.shape[1] % MXFP8_BLOCK == 0, "TE requires both dims % 32 == 0"
    xf = x.float() if dim == 1 else x.float().t()
    rows, cols = xf.shape
    blocks = xf.reshape(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK)

    magnitude = blocks.abs()
    if dim == 1:
        magnitude = torch.where(torch.isnan(magnitude), torch.zeros_like(magnitude), magnitude)
    amax = magnitude.amax(-1)

    target = amax * torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32)
    bits = target.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exponent = exponent + ((mantissa != 0) & (exponent < 0xFE)).int()
    exponent = exponent.clamp(max=0xFE)
    exponent = torch.where(torch.isnan(target), torch.full_like(exponent, 0xFF), exponent)

    multiplier = torch.exp2(127.0 - exponent.float())
    multiplier = torch.where(exponent == 0xFF, torch.full_like(multiplier, float("nan")), multiplier)
    scaled = (blocks * multiplier[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
    data = scaled.to(torch.float8_e4m3fn).reshape(rows, cols)
    scale = exponent.to(torch.uint8)
    if dim == 0:
        data, scale = data.t(), scale.t()
    return data.contiguous(), scale.contiguous()


def dequantize(data: torch.Tensor, scale: torch.Tensor, dim: int, dtype=torch.float32) -> torch.Tensor:
    """Inverse of :func:`quantize` (exact: every MXFP8 value is representable in FP32)."""
    values = data.float() if dim == 1 else data.float().t()
    exps = scale if dim == 1 else scale.t()
    rows, cols = values.shape
    factor = torch.exp2(exps.float() - 127.0)
    out = values.reshape(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK) * factor[..., None]
    out = out.reshape(rows, cols)
    return (out if dim == 1 else out.t()).to(dtype).contiguous()


def fake_quantize(x: torch.Tensor, dim: int, dtype=torch.float64) -> torch.Tensor:
    """MXFP8 quantize-dequantize of ``x`` with blocks along ``dim``."""
    return dequantize(*quantize(x, dim), dim, dtype=dtype)


def matmul(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
    """Reference for ``a @ b^T`` (or ``a @ b``) with both operands MXFP8-quantized
    along the reduction dimension, accumulated in FP64."""
    qa = fake_quantize(a, dim=1)
    qb = fake_quantize(b, dim=1 if transpose_b else 0)
    return qa @ (qb.t() if transpose_b else qb)
