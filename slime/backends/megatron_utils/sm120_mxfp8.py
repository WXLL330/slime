"""SM120 dense MXFP8 backend for Transformer Engine custom recipes."""

from __future__ import annotations

import dataclasses
import inspect
import math
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper, to_blocked
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from transformer_engine.pytorch.custom_recipes.quantization import GEMMType, MMParams
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer

BLOCK_SIZE = 32
QUANT_DTYPE = torch.float8_e4m3fn
SCALE_DTYPE = torch.float8_e8m0fnu
QUANTIZER_FACTORY = "slime.backends.megatron_utils.sm120_mxfp8.quantizer_factory"

_TE_PATCHES_INSTALLED = False


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _validate_runtime_compatibility(custom_gemm) -> None:
    torchao_version = _package_version("torchao")
    te_version = _package_version("transformer_engine")
    if not torchao_version.startswith("0.17."):
        raise RuntimeError(
            "SM120 MXFP8 is validated with TorchAO 0.17.x, got "
            f"{torchao_version}. The backend uses TorchAO prototype MXFP8 APIs."
        )

    expected_parameters = (
        "A",
        "B",
        "workspace",
        "out_dtype",
        "quantization_params",
        "gelu",
        "gelu_in",
        "accumulate",
        "layout",
        "out",
        "bias",
        "use_split_accumulator",
        "grad",
    )
    actual_parameters = tuple(inspect.signature(custom_gemm).parameters)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            "Unsupported Transformer Engine custom_gemm API for SM120 MXFP8 "
            f"(transformer_engine={te_version}): expected parameters {expected_parameters}, "
            f"got {actual_parameters}"
        )


def _validate_sm120_device(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise RuntimeError("SM120 MXFP8 requires CUDA tensors")
    if torch.cuda.get_device_capability(tensor.device) != (12, 0):
        raise RuntimeError(
            "SM120 MXFP8 requires a compute capability 12.0 GPU, got "
            f"{torch.cuda.get_device_capability(tensor.device)}"
        )


def _flatten_and_pad(tensor: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, ...], int]:
    _validate_sm120_device(tensor)
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"SM120 MXFP8 only supports BF16 or FP32 inputs, got {tensor.dtype}")
    if tensor.ndim < 2:
        raise ValueError(f"SM120 MXFP8 expects a tensor with at least 2 dimensions, got {tensor.ndim}")

    original_shape = tuple(tensor.shape)
    tensor = tensor.reshape(-1, tensor.shape[-1]).contiguous()
    if tensor.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(
            "SM120 MXFP8 requires the GEMM K dimension to be divisible by 32, got " f"shape {tuple(tensor.shape)}"
        )

    padded_rows = math.ceil(tensor.shape[0] / BLOCK_SIZE) * BLOCK_SIZE
    if padded_rows != tensor.shape[0]:
        tensor = F.pad(tensor, (0, 0, 0, padded_rows - tensor.shape[0]))
    return tensor, original_shape, padded_rows


def _install_te_patches() -> None:
    """Apply recipe-scoped fixes for TE's custom GEMM dispatch."""
    global _TE_PATCHES_INSTALLED
    if _TE_PATCHES_INSTALLED:
        return

    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import gemm as cpp_gemm
    from transformer_engine.pytorch.custom_recipes import gemm as custom_recipe_gemm
    from transformer_engine.pytorch.module import layernorm_linear, layernorm_mlp, linear
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.utils import assert_dim_for_fp8_exec as te_dim_check

    te_custom_gemm = custom_recipe_gemm.custom_gemm
    te_bgrad_quantize = tex.bgrad_quantize
    _validate_runtime_compatibility(te_custom_gemm)

    def recipe_is_active() -> bool:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        return getattr(recipe, "qfactory", None) is quantizer_factory

    def custom_recipe_dim_check(*tensors: torch.Tensor) -> None:
        if not recipe_is_active():
            te_dim_check(*tensors)
            return

        input_tensor, *weights = tensors
        if input_tensor.shape[-1] % BLOCK_SIZE != 0:
            raise ValueError(
                "SM120 MXFP8 requires the input feature dimension to be divisible by 32, got "
                f"{tuple(input_tensor.shape)}"
            )
        for weight in weights:
            if weight.shape[-2] % BLOCK_SIZE != 0 or weight.shape[-1] % BLOCK_SIZE != 0:
                raise ValueError(
                    "SM120 MXFP8 requires dense weight dimensions to be divisible by 32, got " f"{tuple(weight.shape)}"
                )

    def layout_aware_custom_gemm(*args, **kwargs):
        layout = args[8] if len(args) > 8 else kwargs.get("layout", "TN")
        uses_sm120_storage = (
            len(args) >= 2
            and isinstance(args[0], SM120MXFP8TensorStorage)
            and isinstance(args[1], SM120MXFP8TensorStorage)
        )
        accumulate = args[7] if len(args) > 7 else kwargs.get("accumulate", False)
        out = args[9] if len(args) > 9 else kwargs.get("out")
        if uses_sm120_storage and (accumulate or out is not None):
            raise NotImplementedError(
                "SM120 MXFP8 does not support GEMM output accumulation; disable " "gradient_accumulation_fusion"
            )
        if uses_sm120_storage and layout in ("NN", "NT"):
            if len(args) > 12:
                args = (*args[:12], True, *args[13:])
            else:
                kwargs["grad"] = True
        return te_custom_gemm(*args, **kwargs)

    def custom_bgrad_quantize(grad_output: torch.Tensor, quantizer: Quantizer):
        if isinstance(quantizer, SM120MXFP8Quantizer):
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
            return grad_bias, quantizer(grad_output)
        return te_bgrad_quantize(grad_output, quantizer)

    linear.assert_dim_for_fp8_exec = custom_recipe_dim_check
    layernorm_linear.assert_dim_for_fp8_exec = custom_recipe_dim_check
    layernorm_mlp.assert_dim_for_fp8_exec = custom_recipe_dim_check
    custom_recipe_gemm.custom_gemm = layout_aware_custom_gemm
    cpp_gemm.custom_gemm = layout_aware_custom_gemm
    tex.bgrad_quantize = custom_bgrad_quantize
    _TE_PATCHES_INSTALLED = True


def _quantize_rowwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mx_tensor = MXTensor.to_mx(
        tensor,
        QUANT_DTYPE,
        BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.FLOOR,
        kernel_preference=KernelPreference.AUTO,
        mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
    )
    return mx_tensor.qdata, mx_tensor.scale


def _quantize_columnwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mx_tensor = _to_mxfp8_dim1_kernel_wrapper(
        tensor,
        BLOCK_SIZE,
        QUANT_DTYPE,
        tensor.dtype,
        KernelPreference.AUTO,
        MXFP8Dim1CastKernelChoice.CUDA,
        ScaleCalculationMode.FLOOR,
    )
    return mx_tensor.qdata, mx_tensor.scale


@dataclasses.dataclass
class SM120MXFP8TensorStorage(QuantizedTensorStorage):
    """TE custom tensor storage backed by TorchAO MXFP8 tensors."""

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    quant_dtype: torch.dtype = QUANT_DTYPE
    original_shape: Optional[Tuple[int, ...]] = None
    padded_rows: Optional[int] = None
    _quantizer: Optional[Quantizer] = None
    _source: Optional[torch.Tensor] = None

    @property
    def custom(self) -> bool:
        return True

    @property
    def _data(self):
        return self.data

    @_data.setter
    def _data(self, value):
        self.data = value

    @property
    def _scale_inv(self):
        return self.scale

    @_scale_inv.setter
    def _scale_inv(self, value):
        self.scale = value

    @property
    def flat_rows(self) -> int:
        if self.original_shape is None:
            raise RuntimeError("MXFP8 storage has no original shape")
        return math.prod(self.original_shape[:-1])

    @property
    def columns(self) -> int:
        if self.original_shape is None:
            raise RuntimeError("MXFP8 storage has no original shape")
        return self.original_shape[-1]

    def prepare_for_saving(
        self,
    ) -> tuple[list[Optional[torch.Tensor]], QuantizedTensorStorage]:
        tensors = [self.data, self.data_t, self.scale, self.scale_t, self._source]
        metadata = dataclasses.replace(
            self,
            data=None,
            data_t=None,
            scale=None,
            scale_t=None,
            _source=None,
        )
        return tensors, metadata

    def restore_from_saved(self, tensors: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        self.data, self.data_t, self.scale, self.scale_t, self._source = tensors[:5]
        return tensors[5:]

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ) -> None:
        if rowwise_usage and self.data is None:
            if self._source is None:
                raise RuntimeError("Cannot recreate discarded rowwise MXFP8 data")
            self.data, self.scale = _quantize_rowwise(self._source)
        if columnwise_usage and self.data_t is None:
            if self._source is None:
                raise RuntimeError("Cannot recreate discarded columnwise MXFP8 data")
            self.data_t, self.scale_t = _quantize_columnwise(self._source)
        if rowwise_usage is False:
            self.data = None
            self.scale = None
        if columnwise_usage is False:
            self.data_t = None
            self.scale_t = None
        if self.data is not None and self.data_t is not None:
            self._source = None

    def size(self, *args):
        if self.original_shape is None:
            raise RuntimeError("MXFP8 storage has no original shape")
        shape = torch.Size(self.original_shape)
        return shape if not args else shape[args[0]]

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if self.original_shape is None or self.padded_rows is None:
            raise RuntimeError("MXFP8 storage metadata is incomplete")

        if self.data is not None and self.scale is not None:
            mx_tensor = MXTensor.from_qdata_and_scales(
                self.data,
                self.scale,
                self.dtype,
                block_size=BLOCK_SIZE,
                kernel_preference=KernelPreference.AUTO,
            )
            tensor = mx_tensor.dequantize(dtype)[: self.flat_rows, : self.columns]
        elif self.data_t is not None and self.scale_t is not None:
            mx_tensor = MXTensor.from_qdata_and_scales(
                self.data_t,
                self.scale_t,
                self.dtype,
                block_size=BLOCK_SIZE,
                kernel_preference=KernelPreference.AUTO,
            )
            tensor = mx_tensor.dequantize(dtype).t()[: self.flat_rows, : self.columns]
        else:
            raise RuntimeError("MXFP8 storage has no data to dequantize")
        return tensor.reshape(self.original_shape)


class SM120MXFP8Quantizer(Quantizer):
    """TorchAO MXFP8 quantization exposed through TE's custom Quantizer API."""

    def __init__(self) -> None:
        super().__init__(rowwise=True, columnwise=True)
        self.internal = True
        self.optimize_for_gemm = True

    @property
    def custom(self) -> bool:
        return True

    @property
    def supports_allgather_fp8(self) -> bool:
        return False

    def _quantize(self, tensor: torch.Tensor) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Tuple[int, ...],
        int,
        Optional[torch.Tensor],
    ]:
        tensor, original_shape, padded_rows = _flatten_and_pad(tensor)

        data = scale = None
        if self.rowwise_usage:
            data, scale = _quantize_rowwise(tensor)

        data_t = scale_t = None
        if self.columnwise_usage:
            data_t, scale_t = _quantize_columnwise(tensor)

        source = None if self.rowwise_usage and self.columnwise_usage else tensor
        return data, scale, data_t, scale_t, original_shape, padded_rows, source

    def quantize(self, tensor: torch.Tensor, **_kwargs) -> SM120MXFP8TensorStorage:
        data, scale, data_t, scale_t, original_shape, padded_rows, source = self._quantize(tensor)
        return SM120MXFP8TensorStorage(
            data=data,
            scale=scale,
            data_t=data_t,
            scale_t=scale_t,
            dtype=tensor.dtype,
            device=tensor.device,
            original_shape=original_shape,
            padded_rows=padded_rows,
            _quantizer=self,
            _source=source,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensorStorage,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensorStorage:
        if not isinstance(dst, SM120MXFP8TensorStorage):
            raise TypeError(f"Expected SM120MXFP8TensorStorage, got {type(dst).__name__}")
        if noop_flag is not None and noop_flag.item() != 0:
            return dst

        data, scale, data_t, scale_t, original_shape, padded_rows, source = self._quantize(src)
        dst.data = data
        dst.scale = scale
        dst.data_t = data_t
        dst.scale_t = scale_t
        dst.dtype = src.dtype
        dst.device = src.device
        dst.original_shape = original_shape
        dst.padded_rows = padded_rows
        dst._quantizer = self
        dst._source = source
        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        **_kwargs,
    ) -> SM120MXFP8TensorStorage:
        del requires_grad
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"SM120 MXFP8 empty storage expects a 2D shape, got {shape}")
        if shape[-1] % BLOCK_SIZE != 0:
            raise ValueError(f"SM120 MXFP8 K dimension must be divisible by 32, got {shape}")
        if device is None:
            device = torch.device("cuda")
        device = torch.device(device)
        padded_rows = math.ceil(shape[0] / BLOCK_SIZE) * BLOCK_SIZE

        data = scale = None
        if self.rowwise_usage:
            data = torch.empty((padded_rows, shape[1]), dtype=QUANT_DTYPE, device=device)
            scale = torch.empty((padded_rows, shape[1] // BLOCK_SIZE), dtype=SCALE_DTYPE, device=device)
        data_t = scale_t = None
        if self.columnwise_usage:
            data_t = torch.empty((shape[1], padded_rows), dtype=QUANT_DTYPE, device=device)
            scale_t = torch.empty((shape[1], padded_rows // BLOCK_SIZE), dtype=SCALE_DTYPE, device=device)
        return SM120MXFP8TensorStorage(
            data=data,
            scale=scale,
            data_t=data_t,
            scale_t=scale_t,
            dtype=dtype,
            device=device,
            original_shape=shape,
            padded_rows=padded_rows,
            _quantizer=self,
        )

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        accumulate: bool = False,
        gemm_type: GEMMType = GEMMType.FPROP,
        qresult_x: Optional[QuantizedTensorStorage] = None,
        qresult_w: Optional[QuantizedTensorStorage] = None,
    ) -> torch.Tensor:
        del m_params
        if out is not None or accumulate:
            raise NotImplementedError(
                "SM120 MXFP8 does not support GEMM output accumulation; disable " "gradient_accumulation_fusion"
            )
        if out_dtype != torch.bfloat16:
            raise TypeError(f"SM120 MXFP8 GEMM only supports BF16 output, got {out_dtype}")
        if not isinstance(qresult_x, SM120MXFP8TensorStorage) or not isinstance(qresult_w, SM120MXFP8TensorStorage):
            raise TypeError("SM120 MXFP8 qgemm requires SM120MXFP8TensorStorage operands")
        _validate_sm120_device(qx)

        m, k = qx.shape
        n, weight_k = qw.shape
        if k != weight_k or k % BLOCK_SIZE != 0:
            raise ValueError(f"Invalid MXFP8 GEMM shapes: {tuple(qx.shape)} and {tuple(qw.shape)}")

        if m == 0 or n == 0 or k == 0:
            result = torch.zeros((m, n), dtype=out_dtype, device=qx.device)
            if bias is not None:
                result = result + bias
        else:
            scale_x = to_blocked(sx.reshape(m, k // BLOCK_SIZE))
            scale_w = to_blocked(sw.reshape(n, k // BLOCK_SIZE))
            result = torch._scaled_mm(
                qx,
                qw.t(),
                scale_a=scale_x.view(SCALE_DTYPE),
                scale_b=scale_w.view(SCALE_DTYPE),
                bias=bias,
                out_dtype=out_dtype,
            )
            if isinstance(result, tuple):
                result = result[0]

        if gemm_type == GEMMType.FPROP:
            rows, columns = qresult_x.flat_rows, qresult_w.flat_rows
        elif gemm_type == GEMMType.DGRAD:
            rows, columns = qresult_x.flat_rows, qresult_w.columns
        elif gemm_type == GEMMType.WGRAD:
            rows, columns = qresult_x.columns, qresult_w.columns
        else:
            raise ValueError(f"Unsupported GEMM type: {gemm_type}")
        return result[:rows, :columns]


def quantizer_factory(role) -> SM120MXFP8Quantizer:
    """Create the dense SM120 quantizer requested by a TE CustomRecipe."""
    _install_te_patches()
    if role is not None and getattr(role, "module_type", None) == "grouped_linear":
        raise NotImplementedError("SM120 MXFP8 does not support grouped GEMM")
    return SM120MXFP8Quantizer()
