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
import triton
import triton.language as tl
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

_original_general_gemm = te_gemm.general_gemm
_original_general_grouped_gemm = te_gemm.general_grouped_gemm

# Tile edge of the transpose kernels, in bytes. 128 was the fastest of 32/64/128 on
# the Qwen3.5-35B-A3B training shapes (RTX PRO 6000).
_TRANSPOSE_TILE = 128

# MXFP8 tensors have both dimensions divisible by 32, so every row of their data starts
# 32 bytes aligned. Telling Triton keeps the loads and stores vectorized.
_TRANSPOSE_ALIGN = 32

# Below this size a single transpose is launch bound, where PyTorch's copy is cheaper.
_TRANSPOSE_MIN_BYTES = 4 << 20


@triton.jit
def _transpose_tile(src, dst, rows, cols, TILE: tl.constexpr):
    """Transpose one TILE x TILE tile of a row-major byte matrix."""
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = (row[:, None] < rows) & (col[None, :] < cols)
    tile = tl.load(src + row[:, None] * cols + col[None, :], mask=mask, other=0)
    tl.store(dst + col[:, None] * rows + row[None, :], tl.trans(tile), mask=tl.trans(mask))


@triton.jit
def _transpose_kernel(src, dst, rows, cols, TILE: tl.constexpr):
    _transpose_tile(src, dst, tl.multiple_of(rows, 32), tl.multiple_of(cols, 32), TILE)


@triton.jit
def _transpose_batch_kernel(meta, count, TILE: tl.constexpr):
    """Transpose ``count`` byte matrices, one program per (tile, matrix).

    ``meta`` holds four ``count``-element int64 arrays: source pointers, destination
    pointers, rows and columns. Shapes may differ between matrices; the grid covers
    the largest one and the masks drop the tiles the smaller ones do not have.
    """
    index = tl.program_id(2)
    src = tl.multiple_of(tl.load(meta + index), 32).to(tl.pointer_type(tl.uint8))
    dst = tl.multiple_of(tl.load(meta + count + index), 32).to(tl.pointer_type(tl.uint8))
    rows = tl.multiple_of(tl.load(meta + 2 * count + index), 32)
    cols = tl.multiple_of(tl.load(meta + 3 * count + index), 32)
    _transpose_tile(src, dst, rows, cols, TILE)


def _check_transposable(tensors: list[torch.Tensor]) -> None:
    assert all(
        t.is_contiguous()
        and t.shape[0] % _TRANSPOSE_ALIGN == 0
        and t.shape[1] % _TRANSPOSE_ALIGN == 0
        and t.data_ptr() % _TRANSPOSE_ALIGN == 0
        for t in tensors
    ), "the transpose kernels assume row-major MXFP8 data: shape and address are multiples of 32"


def _transpose_one(src: torch.Tensor) -> torch.Tensor:
    _check_transposable([src])
    dst = torch.empty((src.shape[1], src.shape[0]), dtype=src.dtype, device=src.device)
    grid = (triton.cdiv(src.shape[0], _TRANSPOSE_TILE), triton.cdiv(src.shape[1], _TRANSPOSE_TILE))
    # Passing the tensors themselves lets Triton see their alignment, which is worth 2x
    # over the batch kernel reading the same pointers out of an argument array.
    _transpose_kernel[grid](src, dst, src.shape[0], src.shape[1], TILE=_TRANSPOSE_TILE, num_warps=8)
    return dst


def _device_meta(values: list[int], device: torch.device) -> torch.Tensor:
    """Put ``values`` on the device without making the host wait for the GPU.

    ``torch.tensor(..., device=...)`` copies from pageable memory, which blocks the host
    until the whole CUDA queue has drained (measured: 51 ms of a 54 ms queue). The backward
    pass issues one of these per grouped GEMM, so it would serialize the host against the
    GPU and cost far more than the transpose saves. A pinned staging buffer keeps the copy
    asynchronous, and PyTorch's caching host allocator only recycles the buffer once the
    copy it was used for has run.
    """
    staging = torch.empty(len(values), dtype=torch.int64, pin_memory=True)
    staging.numpy()[:] = values
    return staging.to(device, non_blocking=True)


def _transpose_batch(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    _check_transposable(tensors)
    out, pointers, rows, cols = [], [], [], []
    for src in tensors:
        dst = torch.empty((src.shape[1], src.shape[0]), dtype=src.dtype, device=src.device)
        out.append(dst)
        pointers.append(dst.data_ptr())
        rows.append(src.shape[0])
        cols.append(src.shape[1])
    meta = _device_meta([src.data_ptr() for src in tensors] + pointers + rows + cols, tensors[0].device)
    grid = (triton.cdiv(max(rows), _TRANSPOSE_TILE), triton.cdiv(max(cols), _TRANSPOSE_TILE), len(tensors))
    _transpose_batch_kernel[grid](meta, len(tensors), TILE=_TRANSPOSE_TILE, num_warps=8)
    return out


def _transpose_bytes(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Transpose contiguous 2D byte tensors in as few kernel launches as possible.

    ``Tensor.t().contiguous()`` falls back to PyTorch's element-wise copy for 1-byte
    dtypes, which reads the transposed operand without coalescing (6x to 14x slower
    than moving the same bytes contiguously) and costs one launch per tensor. The
    routed experts transpose one operand per expert, so there the launches dominate:
    one batched launch for all of them beats PyTorch by 4x. A single operand only wins
    once it is large enough for the bandwidth to outweigh the launch, which on an
    RTX PRO 6000 is around 4 MiB.
    """
    if len(tensors) <= 2:
        return [src.t().contiguous() if src.numel() < _TRANSPOSE_MIN_BYTES else _transpose_one(src) for src in tensors]
    return _transpose_batch(tensors)


def transpose_columnwise_to_rowwise_batch(tensors: list[MXFP8TensorStorage]) -> list[MXFP8TensorStorage]:
    """Return the row-wise MXFP8 tensors of ``tensor^T`` built from their column-wise copies.

    The column-wise scale buffer is padded to (roundup(M/32, 4), roundup(K, 128)),
    so its transpose already has the row-wise padded shape (roundup(K, 128),
    roundup(M/32, 4)) of the transposed tensor; the GEMM never reads the padding.
    TE swizzles column-wise scales for the GEMM operand they feed, which is
    byte-for-byte the swizzled row-wise layout of the transposed tensor, so
    swizzled scales only need their 2D view swapped.
    """
    datas = []
    for tensor in tensors:
        assert tensor._columnwise_data is not None, "MXFP8 operand is missing its column-wise data"
        data = tensor._columnwise_data
        datas.append(data.reshape(-1, data.shape[-1]))
    transposed = _transpose_bytes([data.view(torch.uint8) for data in datas])
    out = []
    for tensor, data, data_t in zip(tensors, datas, transposed, strict=True):
        scale = tensor._columnwise_scale_inv
        if tensor._with_gemm_swizzled_scales:
            scale = scale.reshape(scale.shape[1], scale.shape[0])
        else:
            scale = scale.t().contiguous()
        out.append(
            MXFP8TensorStorage(
                rowwise_data=data_t.view(data.dtype),
                rowwise_scale_inv=scale,
                columnwise_data=None,
                columnwise_scale_inv=None,
                fp8_dtype=tensor._fp8_dtype,
                quantizer=None,
                with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
            )
        )
    return out


def transpose_columnwise_to_rowwise(tensor: MXFP8TensorStorage) -> MXFP8TensorStorage:
    """Single-tensor form of ``transpose_columnwise_to_rowwise_batch``."""
    return transpose_columnwise_to_rowwise_batch([tensor])[0]


def _is_mxfp8_pair(A, B) -> bool:
    return isinstance(A, MXFP8TensorStorage) and isinstance(B, MXFP8TensorStorage)


def general_gemm(A, B, *args, layout: str = "TN", **kwargs):
    # TE layouts name (transa, transb); TN is the only MXFP8 layout cuBLASLt supports on SM120.
    # A needs a transpose when it is not transposed, B when it is.
    if layout != "TN" and _is_mxfp8_pair(A, B):
        transpose_a, transpose_b = layout[0] == "N", layout[1] == "T"
        operands = ([A] if transpose_a else []) + ([B] if transpose_b else [])
        transposed = iter(transpose_columnwise_to_rowwise_batch(operands))
        if transpose_a:
            A = next(transposed)
        if transpose_b:
            B = next(transposed)
        layout = "TN"
    return _original_general_gemm(A, B, *args, layout=layout, **kwargs)


def general_grouped_gemm(A, B, out, *args, layout: str = "TN", **kwargs):
    # One transpose launch for all experts, not one per expert.
    if layout != "TN" and A and all(_is_mxfp8_pair(a, b) for a, b in zip(A, B, strict=True)):
        if layout[0] == "N":
            A = transpose_columnwise_to_rowwise_batch(A)
        if layout[1] == "T":
            B = transpose_columnwise_to_rowwise_batch(B)
        layout = "TN"
    return _original_general_grouped_gemm(A, B, out, *args, layout=layout, **kwargs)


def apply_sm120_mxfp8_patch() -> None:
    """Enable TE's MXFP8 recipe on SM120 and route its NN/NT GEMMs through TN."""
    assert torch.cuda.get_device_capability() >= (12, 0), "the SM120 MXFP8 patch only targets SM12x GPUs"
    te_quantization._MXFP8_SUPPORT = (True, "")
    te_linear.general_gemm = general_gemm
    te_layernorm_linear.general_gemm = general_gemm
    te_grouped_linear.general_grouped_gemm = general_grouped_gemm
