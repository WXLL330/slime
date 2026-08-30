"""Runtime support for updating SGLang online-MXFP8 weights from BF16."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from fastapi import Request
from fastapi.responses import ORJSONResponse

from slime.utils.distributed_utils import get_gloo_group

_STATE_ATTR = "_slime_mxfp8_update_state"
_PATCH_ATTR = "_slime_online_mxfp8_weight_sync_installed"
_MODEL_RUNNER_PATCH_ATTR = "_slime_online_mxfp8_transaction_installed"
_DEFER_COMMIT_ATTR = "_slime_defer_mxfp8_commit"
_ROUTE_ATTR = "_slime_post_process_weights_route_installed"
SLIME_POST_PROCESS_WEIGHTS_ENDPOINT = "slime/post_process_weights"


class Rank0UpdatePhaseError(RuntimeError):
    """A rank-0 rollout lifecycle failure broadcast to every training rank."""


def uses_online_mxfp8(args: Any) -> bool:
    return getattr(args, "sglang_quantization", None) == "mxfp8"


def requires_weight_requantization(args: Any, quantization_config: dict[str, Any] | None) -> bool:
    if uses_online_mxfp8(args):
        return True
    return bool(quantization_config and quantization_config.get("quant_method") == "compressed-tensors")


def abort_online_mxfp8_weight_update(rollout_engines: list[Any], *, resume: bool = True) -> None:
    """Roll back pending MXFP8 storage swaps and resume paused engines."""
    import ray

    try:
        ray.get(
            [
                engine.post_process_weights.remote(
                    restore_weights_before_load=True,
                    post_process_quantization=True,
                )
                for engine in rollout_engines
            ]
        )
    finally:
        if resume:
            ray.get([engine.continue_generation.remote() for engine in rollout_engines])


def run_rank_0_update_phase(
    phase: str,
    action: Callable[[], None],
    *,
    rollback_engines: Sequence[Any] | None = None,
) -> None:
    """Run a rollout lifecycle phase on rank 0 and broadcast its result."""
    payload: list[str | None] = [None]
    cause = None
    if dist.get_rank() == 0:
        try:
            action()
        except Exception as exc:
            cause = exc
            payload[0] = f"{phase} failed on rank 0: {type(exc).__name__}: {exc}"
            if rollback_engines is not None:
                try:
                    abort_online_mxfp8_weight_update(list(rollback_engines))
                except Exception as rollback_exc:
                    payload[0] += f"; rollback failed: {type(rollback_exc).__name__}: {rollback_exc}"

    dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
    if payload[0] is not None:
        raise Rank0UpdatePhaseError(payload[0]) from cause


def install_online_mxfp8_weight_sync_patch() -> None:
    """Add BF16 restore/requantize hooks to SGLang's online MXFP8 linear."""
    install_online_mxfp8_linear_patch()
    _install_post_process_weights_route()


def install_online_mxfp8_linear_patch() -> None:
    """Install the linear patch in the current SGLang process."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.model_executor.model_runner import ModelRunner

    if not getattr(Fp8LinearMethod, _PATCH_ATTR, False):
        _patch_mxfp8_linear_method(Fp8LinearMethod)
    if not getattr(ModelRunner, _MODEL_RUNNER_PATCH_ATTR, False):
        _patch_mxfp8_model_runner(ModelRunner)


def run_scheduler_process_with_slime_patches(*args, **kwargs) -> None:
    """SGLang scheduler target that installs Slime patches after spawn."""
    install_online_mxfp8_linear_patch()

    from sglang.srt.managers.scheduler import run_scheduler_process

    run_scheduler_process(*args, **kwargs)


def _patch_mxfp8_linear_method(Fp8LinearMethod: type) -> None:
    """Patch SGLang's linear method while preserving its parameter objects."""

    original_process = Fp8LinearMethod.process_weights_after_loading

    def is_online_mxfp8(method: Fp8LinearMethod) -> bool:
        return bool(method.use_mxfp8 and not method.is_checkpoint_fp8_serialized)

    def restore_weights_before_loading(method: Fp8LinearMethod, layer: torch.nn.Module) -> None:
        if not is_online_mxfp8(method):
            return
        if hasattr(layer, _STATE_ATTR):
            _rollback_layer(layer)
            return

        weight = layer.weight
        source_dtype = getattr(layer, "orig_dtype", None)
        if source_dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"Online MXFP8 weight update requires BF16 or FP16 source weights, got {source_dtype}")
        if weight.dtype == source_dtype:
            return
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"Expected an MXFP8 E4M3 weight before update, got {weight.dtype}")

        state = {
            "weight": weight.data,
            "weight_scale_inv": layer.weight_scale_inv.data,
        }
        setattr(layer, _STATE_ATTR, state)
        weight.data = torch.empty(weight.shape, dtype=source_dtype, device=weight.device)

    def process_weights_after_loading(method: Fp8LinearMethod, layer: torch.nn.Module) -> None:
        state = getattr(layer, _STATE_ATTR, None)
        try:
            original_process(method, layer)
            if state is None:
                return

            if not getattr(method, _DEFER_COMMIT_ATTR, False):
                _commit_layer(layer)
        except Exception:
            if state is not None:
                _rollback_layer(layer)
            raise

    Fp8LinearMethod.restore_weights_before_loading = restore_weights_before_loading
    Fp8LinearMethod.process_weights_after_loading = process_weights_after_loading
    setattr(Fp8LinearMethod, _PATCH_ATTR, True)


def _validate_pending_layer(layer: torch.nn.Module) -> None:
    for name, storage in getattr(layer, _STATE_ATTR).items():
        parameter = getattr(layer, name)
        if parameter.shape != storage.shape or parameter.dtype != storage.dtype:
            raise RuntimeError(
                f"Online MXFP8 update changed {name} layout from "
                f"{tuple(storage.shape)}/{storage.dtype} to {tuple(parameter.shape)}/{parameter.dtype}"
            )


def _commit_layer(layer: torch.nn.Module) -> None:
    state = getattr(layer, _STATE_ATTR)
    _validate_pending_layer(layer)
    for name, storage in state.items():
        parameter = getattr(layer, name)
        storage.copy_(parameter.data)
        parameter.data = storage
    delattr(layer, _STATE_ATTR)


def _rollback_layer(layer: torch.nn.Module) -> None:
    state = getattr(layer, _STATE_ATTR, None)
    if state is None:
        return
    for name, storage in state.items():
        getattr(layer, name).data = storage
    delattr(layer, _STATE_ATTR)


def _pending_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in model.modules() if hasattr(module, _STATE_ATTR)]


def _set_deferred_commit(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and getattr(quant_method, "use_mxfp8", False):
            setattr(quant_method, _DEFER_COMMIT_ATTR, enabled)


def _commit_model(model: torch.nn.Module) -> None:
    pending = _pending_layers(model)
    for layer in pending:
        _validate_pending_layer(layer)
    for layer in pending:
        _commit_layer(layer)


def _rollback_model(model: torch.nn.Module) -> None:
    for layer in _pending_layers(model):
        _rollback_layer(layer)


def _patch_mxfp8_model_runner(ModelRunner: type) -> None:
    """Make post-load quantization atomic across all online-MXFP8 linears."""

    original_post_process = ModelRunner.post_process_weights
    original_disk_update = ModelRunner.update_weights_from_disk

    def post_process_weights(self, *args, **kwargs):
        restore_weights_before_load = kwargs.get("restore_weights_before_load", False)
        post_process_quantization = kwargs.get("post_process_quantization", False)
        if len(args) >= 1:
            restore_weights_before_load = args[0]
        if len(args) >= 2:
            post_process_quantization = args[1]
        if restore_weights_before_load and post_process_quantization:
            _rollback_model(self.model)
            return True, "Online MXFP8 weight update rolled back"
        if not post_process_quantization:
            try:
                return original_post_process(self, *args, **kwargs)
            except Exception:
                _rollback_model(self.model)
                raise

        _set_deferred_commit(self.model, True)
        try:
            result = original_post_process(self, *args, **kwargs)
            _commit_model(self.model)
            return result
        except Exception:
            _rollback_model(self.model)
            raise
        finally:
            _set_deferred_commit(self.model, False)

    def update_weights_from_disk(self, *args, **kwargs):
        if not _pending_layers(self.model):
            return original_disk_update(self, *args, **kwargs)

        _set_deferred_commit(self.model, True)
        try:
            result = original_disk_update(self, *args, **kwargs)
            if result[0]:
                _commit_model(self.model)
            else:
                _rollback_model(self.model)
            return result
        except Exception:
            _rollback_model(self.model)
            raise
        finally:
            _set_deferred_commit(self.model, False)

    ModelRunner.post_process_weights = post_process_weights
    ModelRunner.update_weights_from_disk = update_weights_from_disk
    setattr(ModelRunner, _MODEL_RUNNER_PATCH_ATTR, True)


def _install_post_process_weights_route() -> None:
    """Register a JSON-body route compatible with SGLang's msgspec request type."""
    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers.io_struct import PostProcessWeightsReqInput

    if getattr(http_server.app, _ROUTE_ATTR, False):
        return

    from sglang.srt.utils.auth import AuthLevel, auth_level

    @http_server.app.post(f"/{SLIME_POST_PROCESS_WEIGHTS_ENDPOINT}")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def slime_post_process_weights(request: Request):
        req = PostProcessWeightsReqInput(**(await request.json()))
        success, message = await http_server._global_state.tokenizer_manager.post_process_weights(req, request)
        return ORJSONResponse(
            {"success": success, "message": message},
            status_code=200 if success else 400,
        )

    setattr(http_server.app, _ROUTE_ATTR, True)
