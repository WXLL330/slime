from __future__ import annotations

import asyncio
import socket
import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

from slime.backends.sglang_utils.mxfp8_weight_sync import (
    SLIME_POST_PROCESS_WEIGHTS_ENDPOINT,
    _patch_mxfp8_linear_method,
    _patch_mxfp8_model_runner,
    install_online_mxfp8_weight_sync_patch,
    requires_weight_requantization,
    run_scheduler_process_with_slime_patches,
)

NUM_GPUS = 1


def _require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SGLang MXFP8 weight sync tests require a compute capability 12.0 GPU")


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _run_rank_0_failure_worker(rank, world_size, port, output_dir, phase):
    import torch.distributed as dist

    from slime.backends.sglang_utils import mxfp8_weight_sync
    from slime.utils import distributed_utils

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    distributed_utils.set_gloo_group(dist.group.WORLD)
    events = []

    def action():
        events.append("action")
        raise ValueError("injected rank-0 failure")

    def rollback():
        events.append("rollback")

    mxfp8_weight_sync.abort_online_mxfp8_weight_update = lambda engines: rollback()
    try:
        mxfp8_weight_sync.run_rank_0_update_phase(phase, action, rollback_engines=[])
    except mxfp8_weight_sync.Rank0UpdatePhaseError as exc:
        error = str(exc)
    else:
        error = "no error"
    finally:
        Path(output_dir, f"rank-{rank}.txt").write_text(f"{','.join(events)}\n{error}\n")
        distributed_utils.set_gloo_group(None)
        dist.destroy_process_group()


def _run_transfer_failure_worker(rank, world_size, port, output_dir):
    import torch.distributed as dist

    from slime.backends.megatron_utils.update_weight import update_weight_from_tensor as tensor_module
    from slime.backends.megatron_utils.update_weight.common import WeightTransferError
    from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
        UpdateWeightFromDistributed,
    )
    from slime.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor
    from slime.utils import distributed_utils

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    distributed_utils.set_gloo_group(dist.group.WORLD)
    results = []

    distributed_updater = UpdateWeightFromDistributed.__new__(UpdateWeightFromDistributed)
    distributed_updater._iter_non_expert_chunks = lambda: iter([[("weight", torch.empty(0))]] if rank == 0 else [])
    distributed_updater._iter_expert_chunks = lambda: iter([])
    distributed_updater._on_chunk = lambda chunk: None

    def fail_distributed_transfer(chunk, pbar):
        raise RuntimeError("injected distributed Ray failure")

    distributed_updater._update_bucket_weights_from_distributed = fail_distributed_transfer
    try:
        distributed_updater._send_weights(None)
    except WeightTransferError as exc:
        results.append(str(exc))
    else:
        results.append("no distributed error")

    tensor_module.ray.get = lambda refs: refs
    tensor_module.accelerator.ipc_collect = lambda: None
    tensor_module.accelerator.empty_cache = lambda: None

    class FakeIterator:
        def get_hf_weight_chunks(self, weights, param_info_buckets):
            yield [("weight", torch.empty(0))]

    tensor_updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    tensor_updater.args = types.SimpleNamespace(sglang_quantization=None)
    tensor_updater.rollout_engines = []
    tensor_updater.quantization_config = None
    tensor_updater.weights_getter = lambda: {}
    tensor_updater._expert_transfer_plan = []
    tensor_updater._full_param_info_buckets = []
    tensor_updater._hf_weight_iterator = FakeIterator()

    def send_tensor_weights(chunk):
        if rank == 0:
            raise RuntimeError("injected tensor Ray failure")
        return [], None

    tensor_updater._send_hf_params = send_tensor_weights
    try:
        tensor_updater._update_weights()
    except WeightTransferError as exc:
        results.append(str(exc))
    else:
        results.append("no tensor error")

    Path(output_dir, f"transfer-rank-{rank}.txt").write_text("\n".join(results) + "\n")
    distributed_utils.set_gloo_group(None)
    dist.destroy_process_group()


def test_requantization_selection():
    mxfp8_args = types.SimpleNamespace(sglang_quantization="mxfp8")
    bf16_args = types.SimpleNamespace(sglang_quantization=None)

    assert requires_weight_requantization(mxfp8_args, None)
    assert requires_weight_requantization(bf16_args, {"quant_method": "compressed-tensors"})
    assert not requires_weight_requantization(bf16_args, None)
    assert not requires_weight_requantization(bf16_args, {"quant_method": "fp8"})


@pytest.mark.parametrize("phase", ["rollout weight-update prepare", "rollout weight-update finalize"])
def test_rank_0_phase_failure_is_rolled_back_and_broadcast_to_all_ranks(tmp_path, phase):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_rank_0_failure_worker,
        args=(world_size, _free_port(), str(tmp_path), phase),
        nprocs=world_size,
        join=True,
    )

    expected_error = f"{phase} failed on rank 0: ValueError: injected rank-0 failure"
    assert (tmp_path / "rank-0.txt").read_text().splitlines() == ["action,rollback", expected_error]
    assert (tmp_path / "rank-1.txt").read_text().splitlines() == ["", expected_error]


def test_source_rank_transfer_failures_are_broadcast_to_all_ranks(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_transfer_failure_worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    expected = [
        "non-expert rollout weight transfer failed: rank 0: RuntimeError: injected distributed Ray failure",
        "rollout tensor weight transfer failed: rank 0: RuntimeError: injected tensor Ray failure",
    ]
    assert (tmp_path / "transfer-rank-0.txt").read_text().splitlines() == expected
    assert (tmp_path / "transfer-rank-1.txt").read_text().splitlines() == expected


@pytest.mark.parametrize(
    ("failure", "expected_error"), [("pause", "injected pause failure"), ("version", "Weight version mismatch")]
)
def test_full_disk_reload_rolls_back_and_resumes_on_failure(monkeypatch, tmp_path, failure, expected_error):
    from slime.ray import actor_group

    events = []

    class Ref:
        def __init__(self, method):
            self.method = method

    class RemoteMethod:
        def __init__(self, method):
            self.method = method

        def remote(self, *args, **kwargs):
            events.append(self.method)
            return Ref(self.method)

    class Engine:
        pause_generation = RemoteMethod("pause")
        flush_cache = RemoteMethod("flush")
        post_process_weights = RemoteMethod("restore")
        update_weights_from_disk = RemoteMethod("update")
        get_weight_version = RemoteMethod("version")
        continue_generation = RemoteMethod("continue")

    engines = [Engine(), Engine()]

    class RolloutManager:
        get_updatable_engines_and_lock = RemoteMethod("get-engines")

    def fake_ray_get(refs):
        if isinstance(refs, Ref):
            if refs.method == "get-engines":
                return engines, None
            return refs
        methods = [ref.method for ref in refs]
        if failure == "pause" and methods and methods[0] == "pause":
            raise RuntimeError("injected pause failure")
        if methods and methods[0] == "version":
            return ["stale"] * len(refs)
        return refs

    monkeypatch.setattr(actor_group.ray, "get", fake_ray_get)
    monkeypatch.setattr(
        actor_group,
        "abort_online_mxfp8_weight_update",
        lambda engines, **kwargs: events.append("rollback"),
    )

    train_group = actor_group.RayTrainGroup.__new__(actor_group.RayTrainGroup)
    train_group.args = types.SimpleNamespace(
        offload_rollout=False,
        update_weight_local_checkpoint_dir=None,
        update_weight_disk_keep_files=True,
        sglang_quantization="mxfp8",
        ci_test=True,
    )
    train_group._rollout_manager = RolloutManager()

    with pytest.raises(RuntimeError, match=expected_error):
        train_group._reload_rollout_weights_from_disk(tmp_path, weight_version=3)

    assert events.count("rollback") == 1
    assert events.count("continue") == len(engines)
    assert events.index("rollback") < events.index("continue")


def test_serialized_mxfp8_restore_is_a_noop():
    install_online_mxfp8_weight_sync_patch()
    method = Fp8LinearMethod(Fp8Config(use_mxfp8=True, is_checkpoint_fp8_serialized=True))
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty((32, 32), dtype=torch.float8_e4m3fn), requires_grad=False),
    )

    weight = layer.weight
    method.restore_weights_before_loading(layer)

    assert layer.weight is weight
    assert layer.weight.dtype == torch.float8_e4m3fn


def test_online_mxfp8_second_restore_rolls_back():
    install_online_mxfp8_weight_sync_patch()
    method = Fp8LinearMethod(Fp8Config(use_mxfp8=True, is_checkpoint_fp8_serialized=False))
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty((32, 32), dtype=torch.float8_e4m3fn), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(torch.empty((32, 1), dtype=torch.uint8), requires_grad=False),
    )

    weight_pointer = layer.weight.data_ptr()
    scale_pointer = layer.weight_scale_inv.data_ptr()

    method.restore_weights_before_loading(layer)
    method.restore_weights_before_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.data_ptr() == weight_pointer
    assert layer.weight_scale_inv.data_ptr() == scale_pointer


def test_slime_post_process_route_accepts_json_body(monkeypatch):
    from sglang.srt.entrypoints import http_server

    install_online_mxfp8_weight_sync_patch()
    route = next(
        route
        for route in http_server.app.routes
        if getattr(route, "path", None) == f"/{SLIME_POST_PROCESS_WEIGHTS_ENDPOINT}"
    )
    assert route.dependant.request_param_name == "request"
    assert route.dependant.query_params == []
    received = []

    class TokenizerManager:
        async def post_process_weights(self, req, request):
            received.append(req)
            return True, "Success"

    class Request:
        async def json(self):
            return {
                "restore_weights_before_load": True,
                "post_process_quantization": False,
            }

    monkeypatch.setattr(
        http_server,
        "_global_state",
        types.SimpleNamespace(tokenizer_manager=TokenizerManager()),
    )
    response = asyncio.run(route.endpoint(Request()))

    assert response.status_code == 200
    assert received[0].restore_weights_before_load is True
    assert received[0].post_process_quantization is False

    from sglang.srt.utils.auth import AuthLevel, decide_request_auth

    assert route.endpoint._auth_level == AuthLevel.ADMIN_OPTIONAL
    assert not decide_request_auth(
        method="POST",
        path=route.path,
        authorization_header=None,
        api_key="secret",
        admin_api_key=None,
        auth_level=route.endpoint._auth_level,
    ).allowed
    assert decide_request_auth(
        method="POST",
        path=route.path,
        authorization_header="Bearer secret",
        api_key="secret",
        admin_api_key=None,
        auth_level=route.endpoint._auth_level,
    ).allowed


@pytest.mark.parametrize(
    ("quantization", "expected_endpoint"),
    (("mxfp8", SLIME_POST_PROCESS_WEIGHTS_ENDPOINT), (None, "post_process_weights")),
)
def test_sglang_engine_selects_compatible_post_process_endpoint(quantization, expected_endpoint):
    from slime.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = types.SimpleNamespace(sglang_quantization=quantization)
    calls = []
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))

    engine.post_process_weights(restore_weights_before_load=True)

    assert calls == [
        (
            expected_endpoint,
            {
                "restore_weights_before_load": True,
                "post_process_quantization": False,
            },
        )
    ]


def test_scheduler_process_installs_linear_patch_after_spawn(monkeypatch):
    from sglang.srt.managers import scheduler
    from slime.backends.sglang_utils import mxfp8_weight_sync

    events = []
    monkeypatch.setattr(
        mxfp8_weight_sync,
        "install_online_mxfp8_linear_patch",
        lambda: events.append(("patch",)),
    )
    monkeypatch.setattr(
        scheduler,
        "run_scheduler_process",
        lambda *args, **kwargs: events.append(("scheduler", args, kwargs)),
    )

    run_scheduler_process_with_slime_patches("server", tp_rank=1)

    assert events == [
        ("patch",),
        ("scheduler", ("server",), {"tp_rank": 1}),
    ]


def test_launch_server_passes_spawn_safe_scheduler_target(monkeypatch):
    from slime.backends.sglang_utils import sglang_engine

    calls = []
    fake_patch_module = types.ModuleType("slime.backends.sglang_utils.mxfp8_weight_sync")
    fake_patch_module.install_online_mxfp8_weight_sync_patch = lambda: calls.append(("patch",))
    fake_patch_module.run_scheduler_process_with_slime_patches = object()
    fake_http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
    fake_http_server.launch_server = lambda server_args, **kwargs: calls.append(("launch", server_args, kwargs))
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.mxfp8_weight_sync", fake_patch_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", fake_http_server)

    server_args = object()
    sglang_engine._launch_server_with_slime_patches(server_args)

    assert calls == [
        ("patch",),
        (
            "launch",
            server_args,
            {"run_scheduler_process_func": fake_patch_module.run_scheduler_process_with_slime_patches},
        ),
    ]


def test_model_transaction_rolls_back_all_layers_and_can_retry():
    class FakeLinearMethod:
        use_mxfp8 = True
        is_checkpoint_fp8_serialized = False

        def __init__(self):
            self.fail = False

        def process_weights_after_loading(self, layer):
            if self.fail:
                raise RuntimeError("injected quantization failure")
            layer.weight.data = layer.weight.detach().to(torch.float8_e4m3fn)
            layer.weight_scale_inv.data = torch.full_like(layer.weight_scale_inv, 129)

    class FakeModelRunner:
        def __init__(self, model):
            self.model = model

        def post_process_weights(self, restore_weights_before_load=False, post_process_quantization=False):
            for module in self.model.modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None:
                    continue
                if restore_weights_before_load:
                    quant_method.restore_weights_before_loading(module)
                if post_process_quantization:
                    quant_method.process_weights_after_loading(module)
            return True, "Success"

        def update_weights_from_disk(self, *args, **kwargs):
            return True, "Success"

    _patch_mxfp8_linear_method(FakeLinearMethod)
    _patch_mxfp8_model_runner(FakeModelRunner)
    model = torch.nn.ModuleList()
    methods = []
    for value in (1, 2):
        layer = torch.nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(torch.full((32, 32), value, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        layer.register_parameter(
            "weight_scale_inv",
            torch.nn.Parameter(torch.full((32, 1), 127 + value, dtype=torch.uint8), requires_grad=False),
        )
        layer.quant_method = FakeLinearMethod()
        methods.append(layer.quant_method)
        model.append(layer)

    runner = FakeModelRunner(model)
    old_weights = [layer.weight.detach().clone() for layer in model]
    old_scales = [layer.weight_scale_inv.detach().clone() for layer in model]
    old_pointers = [(layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr()) for layer in model]

    runner.post_process_weights(True, False)
    for layer in model:
        layer.weight.fill_(7)
    methods[1].fail = True
    with pytest.raises(RuntimeError, match="injected quantization failure"):
        runner.post_process_weights(False, True)

    for layer, weight, scale, pointers in zip(model, old_weights, old_scales, old_pointers, strict=True):
        assert torch.equal(layer.weight, weight)
        assert torch.equal(layer.weight_scale_inv, scale)
        assert (layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr()) == pointers

    methods[1].fail = False
    runner.post_process_weights(True, False)
    for layer in model:
        layer.weight.fill_(9)
    runner.post_process_weights(False, True)

    for layer, pointers in zip(model, old_pointers, strict=True):
        assert (layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr()) == pointers
        assert torch.equal(layer.weight, torch.full_like(layer.weight, 9))


def test_abort_rolls_back_and_resumes_every_engine(monkeypatch):
    import ray

    from slime.backends.sglang_utils.mxfp8_weight_sync import abort_online_mxfp8_weight_update

    events = []

    class RemoteMethod:
        def __init__(self, engine, name):
            self.engine = engine
            self.name = name

        def remote(self, **kwargs):
            events.append((self.engine, self.name, kwargs))
            return object()

    class Engine:
        def __init__(self, name):
            self.post_process_weights = RemoteMethod(name, "rollback")
            self.continue_generation = RemoteMethod(name, "continue")

    monkeypatch.setattr(ray, "get", lambda refs: refs)
    abort_online_mxfp8_weight_update([Engine("a"), Engine("b")])

    assert events == [
        (
            "a",
            "rollback",
            {"restore_weights_before_load": True, "post_process_quantization": True},
        ),
        (
            "b",
            "rollback",
            {"restore_weights_before_load": True, "post_process_quantization": True},
        ),
        ("a", "continue", {}),
        ("b", "continue", {}),
    ]


def test_two_bf16_updates_requantize_mxfp8_and_preserve_runtime_storage(monkeypatch):
    _require_sm120()
    monkeypatch.setattr(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", fp8_utils.Fp8GemmRunnerBackend.TRITON)
    install_online_mxfp8_weight_sync_patch()

    method = Fp8LinearMethod(Fp8Config(use_mxfp8=True, is_checkpoint_fp8_serialized=False))
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.empty((256, 256), device="cuda", dtype=torch.bfloat16), requires_grad=False),
    )

    torch.manual_seed(123)
    initial_weight = torch.randn_like(layer.weight)
    layer.weight.copy_(initial_weight)
    method.process_weights_after_loading(layer)

    weight_pointer = layer.weight.data_ptr()
    scale_pointer = layer.weight_scale_inv.data_ptr()
    previous_qweight = layer.weight.detach().clone()
    input_tensor = torch.randn((63, 256), device="cuda", dtype=torch.bfloat16)

    for seed in (456, 789):
        torch.manual_seed(seed)
        source_weight = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)

        method.restore_weights_before_loading(layer)
        assert layer.weight.dtype == torch.bfloat16
        layer.weight.copy_(source_weight)
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale_inv.dtype == torch.uint8
        assert layer.weight.data_ptr() == weight_pointer
        assert layer.weight_scale_inv.data_ptr() == scale_pointer
        assert not torch.equal(layer.weight, previous_qweight)

        dequantized_weight = upcast_from_mxfp(
            layer.weight,
            layer.weight_scale_inv,
            torch.float32,
            axis=1,
        ).to(torch.bfloat16)
        assert _relative_error(dequantized_weight, source_weight) < 0.06

        output = method.apply(layer, input_tensor)
        quantized_input, input_scale = mxfp8_group_quantize(input_tensor)
        dequantized_input = upcast_from_mxfp(quantized_input, input_scale, torch.float32, axis=1).to(torch.bfloat16)
        reference = torch.nn.functional.linear(dequantized_input, dequantized_weight)
        assert output.dtype == torch.bfloat16
        assert _relative_error(output, reference) < 0.02
        previous_qweight = layer.weight.detach().clone()
