"""Run TransformerEngine MXFP8 training GEMMs on SM120.

TE disables MXFP8 on SM12x because cuBLASLt there only has MXFP8 kernels for the
TN layout, while TE's backward pass issues NN (dgrad) and NT (wgrad) GEMMs on the
column-wise copies of its MXFP8 tensors. TE stores a column-wise copy with the
same shape as the source tensor (only the 32-element scaling blocks run along
dim 0), so transposing its FP8 data and E8M0 scales yields exactly the
row-wise MXFP8 quantization of the transposed tensor. Feeding those transposed
copies turns every NN/NT GEMM into a TN GEMM with bit-identical operands.

``apply_sm120_mxfp8_patch`` must be called explicitly before building TE
modules; importing this file has no side effects.
"""

import torch
import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
import transformer_engine.pytorch.module.linear as te_linear
import transformer_engine.pytorch.quantization as te_quantization
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

_original_general_gemm = te_gemm.general_gemm
_original_general_grouped_gemm = te_gemm.general_grouped_gemm


def transpose_columnwise_to_rowwise(tensor: MXFP8TensorStorage) -> MXFP8TensorStorage:
    """Return the row-wise MXFP8 tensor of ``tensor^T`` built from its column-wise copy.

    The column-wise scale buffer is padded to (roundup(M/32, 4), roundup(K, 128)),
    so its transpose already has the row-wise padded shape (roundup(K, 128),
    roundup(M/32, 4)) of the transposed tensor; the GEMM never reads the padding.
    TE swizzles column-wise scales for the GEMM operand they feed, which is
    byte-for-byte the swizzled row-wise layout of the transposed tensor, so
    swizzled scales only need their 2D view swapped.
    """
    assert tensor._columnwise_data is not None, "MXFP8 operand is missing its column-wise data"
    data = tensor._columnwise_data
    data = data.reshape(-1, data.shape[-1])
    scale = tensor._columnwise_scale_inv
    if tensor._with_gemm_swizzled_scales:
        scale = scale.reshape(scale.shape[1], scale.shape[0])
    else:
        scale = scale.t().contiguous()
    return MXFP8TensorStorage(
        rowwise_data=data.t().contiguous(),
        rowwise_scale_inv=scale,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=tensor._fp8_dtype,
        quantizer=None,
        with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
    )


def _as_tn(A, B, layout: str):
    # TE layouts name (transa, transb); TN is the only MXFP8 layout cuBLASLt supports on SM120.
    # A needs a transpose when it is not transposed, B when it is.
    if layout == "TN" or not (isinstance(A, MXFP8TensorStorage) and isinstance(B, MXFP8TensorStorage)):
        return A, B, layout
    if layout[0] == "N":
        A = transpose_columnwise_to_rowwise(A)
    if layout[1] == "T":
        B = transpose_columnwise_to_rowwise(B)
    return A, B, "TN"


def general_gemm(A, B, *args, layout: str = "TN", **kwargs):
    A, B, layout = _as_tn(A, B, layout)
    return _original_general_gemm(A, B, *args, layout=layout, **kwargs)


def general_grouped_gemm(A, B, out, *args, layout: str = "TN", **kwargs):
    if layout != "TN":
        converted = [_as_tn(a, b, layout) for a, b in zip(A, B, strict=True)]
        A = [a for a, _, _ in converted]
        B = [b for _, b, _ in converted]
        layout = converted[0][2] if converted else layout
    return _original_general_grouped_gemm(A, B, out, *args, layout=layout, **kwargs)


def apply_sm120_mxfp8_patch() -> None:
    """Enable TE's MXFP8 recipe on SM120 and route its NN/NT GEMMs through TN."""
    assert torch.cuda.get_device_capability() >= (12, 0), "the SM120 MXFP8 patch only targets SM12x GPUs"
    te_quantization._MXFP8_SUPPORT = (True, "")
    te_linear.general_gemm = general_gemm
    te_layernorm_linear.general_gemm = general_gemm
    te_grouped_linear.general_grouped_gemm = general_grouped_gemm
