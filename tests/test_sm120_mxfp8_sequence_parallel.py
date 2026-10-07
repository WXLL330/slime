"""MXFP8 tensor-parallel linears with sequence parallelism, checked against the emulation.

With sequence parallelism TE all-gathers MXFP8 tensors across the TP group: the
column-parallel input in forward and backward (wgrad needs its column-wise copy),
and the row-parallel grad output in backward. On SM120 the gathered column-wise
copies then go through the slime patch that rewrites NN/NT GEMMs as TN. Each rank
compares its shard with a full-size FP64 emulation; local 32-row (or 32-column)
blocks line up with the global ones because every shard is a multiple of 32.
"""

import socket
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

NUM_GPUS = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _mxfp8_emulation as emu  # noqa: E402

CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
IS_SM120 = CAPABILITY is not None and CAPABILITY[0] == 12
IS_SM100 = CAPABILITY is not None and CAPABILITY[0] == 10
pytestmark = pytest.mark.skipif(
    not (IS_SM120 or IS_SM100) or torch.cuda.device_count() < NUM_GPUS,
    reason=f"needs {NUM_GPUS} SM100 (native TE) or SM120 (slime patch) GPUs",
)

# FP32 accumulation plus one FP32 reduce-scatter, see GEMM_TOL in test_sm120_mxfp8_gemm.py.
TOL = 2e-5
M, K, N = 4096, 2048, 1024  # tokens, in_features, out_features; every shard is 32-aligned


def _full(rows, cols, scale, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, cols, device="cuda", generator=gen) * scale


def _row_rel_err(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm(dim=-1) / expected.norm(dim=-1)).max().item()


def _fq(x, dim):
    return emu.fake_quantize(x, dim)


def _worker(rank, world_size, port, parallel_mode, results):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import MXFP8BlockScaling

        if IS_SM120:
            from slime.backends.megatron_utils.sm120_mxfp8 import apply_sm120_mxfp8_patch

            apply_sm120_mxfp8_patch()

        group = dist.new_group(list(range(world_size)))
        x_full = _full(M, K, 1.0, seed=1)
        w_full = _full(N, K, 0.02, seed=2)
        dy_full = _full(M, N, 1e-3, seed=3)
        rows = slice(rank * M // world_size, (rank + 1) * M // world_size)

        linear = te.Linear(
            K,
            N,
            bias=False,
            params_dtype=torch.float32,
            parallel_mode=parallel_mode,
            sequence_parallel=True,
            tp_group=group,
            tp_size=world_size,
        ).cuda()
        if parallel_mode == "column":
            cols = slice(rank * N // world_size, (rank + 1) * N // world_size)
            w_local, x_local, dy_local = w_full[cols], x_full[rows], dy_full[:, cols]
            # forward gathers the sequence; backward reduce-scatters dgrad over the output shards
            ref_y = _fq(x_full, 1) @ _fq(w_local, 1).t()
            ref_dx = (_fq(dy_full, 1) @ _fq(w_full, 0))[rows]
            ref_dw = _fq(dy_local, 0).t() @ _fq(x_full, 0)
        else:
            ks = slice(rank * K // world_size, (rank + 1) * K // world_size)
            w_local, x_local, dy_local = w_full[:, ks], x_full[:, ks], dy_full[rows]
            # forward reduce-scatters over the input shards; backward gathers the grad output
            ref_y = (_fq(x_full, 1) @ _fq(w_full, 1).t())[rows]
            ref_dx = _fq(dy_full, 1) @ _fq(w_local, 0)
            ref_dw = _fq(dy_full, 0).t() @ _fq(x_local, 0)
        with torch.no_grad():
            linear.weight.copy_(w_local)
        x = x_local.clone().requires_grad_()

        with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
            y = linear(x)
        y.backward(dy_local)

        results[rank] = {
            "y": _row_rel_err(y, ref_y),
            "dx": _row_rel_err(x.grad, ref_dx),
            "dw": _row_rel_err(linear.weight.grad, ref_dw),
        }
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("parallel_mode", ["column", "row"])
def test_sequence_parallel_linear_matches_emulation(parallel_mode):
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_worker, args=(NUM_GPUS, _free_port(), parallel_mode, results), nprocs=NUM_GPUS, join=True)

    assert sorted(results.keys()) == list(range(NUM_GPUS))
    for rank, errors in results.items():
        for name, error in errors.items():
            assert error <= TOL, f"rank {rank} {parallel_mode} {name}: {error:.3e} > {TOL:.0e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
