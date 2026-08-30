from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("transformer_engine")


NUM_GPUS = 2


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _all_gather(tensor, world_size):
    import torch.distributed as dist

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _run_tp_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.pytorch.custom_recipes.quantization import GEMMType

    from slime.backends.megatron_utils.sm120_mxfp8 import QUANTIZER_FACTORY, SM120MXFP8Quantizer

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    model_parallel_cuda_manual_seed(1234)

    calls = []
    original_qgemm = SM120MXFP8Quantizer.qgemm

    def record_qgemm(self, *args, **kwargs):
        calls.append(kwargs["gemm_type"])
        return original_qgemm(self, *args, **kwargs)

    SM120MXFP8Quantizer.qgemm = record_qgemm
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            tensor_model_parallel_size=world_size,
            sequence_parallel=True,
            gradient_accumulation_fusion=False,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            fp8="e4m3",
            fp8_recipe="custom",
            fp8_quantizer_factory=QUANTIZER_FACTORY,
        )
        column = TEColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        row = TERowParallelLinear(
            input_size=512,
            output_size=256,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )

        input_mx = torch.randn(31, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        global_input = torch.cat(_all_gather(input_mx.detach(), world_size), dim=0).requires_grad_()
        global_column_weight = torch.cat(_all_gather(column.weight.detach(), world_size), dim=0).requires_grad_()
        global_row_weight = torch.cat(_all_gather(row.weight.detach(), world_size), dim=1).requires_grad_()

        hidden_ref = torch.nn.functional.linear(global_input, global_column_weight)
        output_ref_global = torch.nn.functional.linear(torch.nn.functional.gelu(hidden_ref), global_row_weight)
        output_ref = output_ref_global.chunk(world_size, dim=0)[rank]
        (output_ref_global.float().square().mean() * world_size).backward()

        with get_fp8_context(config):
            hidden, _ = column(input_mx)
            output, _ = row(torch.nn.functional.gelu(hidden))
        output.float().square().mean().backward()
        torch.cuda.synchronize()

        assert output.shape == input_mx.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(input_mx.grad).all()
        assert torch.isfinite(column.weight.grad).all()
        assert torch.isfinite(row.weight.grad).all()
        comparisons = {
            "output": (output, output_ref),
            "dinput": (input_mx.grad, global_input.grad.chunk(world_size, dim=0)[rank]),
            "column dweight": (column.weight.grad, global_column_weight.grad.chunk(world_size, dim=0)[rank]),
            "row dweight": (row.weight.grad, global_row_weight.grad.chunk(world_size, dim=1)[rank]),
        }
        for name, (actual, expected) in comparisons.items():
            error = _relative_error(actual, expected)
            assert error < 0.10, f"{name} relative error: {error}"
        assert calls == [
            GEMMType.FPROP,
            GEMMType.FPROP,
            GEMMType.DGRAD,
            GEMMType.WGRAD,
            GEMMType.DGRAD,
            GEMMType.WGRAD,
        ]
        dist.barrier()
    finally:
        SM120MXFP8Quantizer.qgemm = original_qgemm
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_megatron_tp2_sequence_parallel_forward_backward():
    if torch.cuda.device_count() < NUM_GPUS:
        pytest.skip("Megatron TP2 MXFP8 test requires two GPUs")
    if any(torch.cuda.get_device_capability(index) != (12, 0) for index in range(NUM_GPUS)):
        pytest.skip("Megatron TP2 MXFP8 test requires SM120 GPUs")

    torch.multiprocessing.spawn(_run_tp_worker, args=(NUM_GPUS, _free_port()), nprocs=NUM_GPUS, join=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
