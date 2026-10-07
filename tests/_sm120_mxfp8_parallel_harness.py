"""torchrun harness for test_sm120_mxfp8_parallel_consistency.py.

Builds a tiny Qwen3.5 MoE with the slime plugin under any Megatron parallel config and
runs one step twice in the same process, on the same weights and data:

1. real MXFP8 (TE MXFP8BlockScaling + the SM120 patch), recording routing and, for every
   TE module, its input and grad_output;
2. emulation: ``config.fp8`` stays set (Megatron's expert Fp8Padding etc. are unchanged)
   but TE's FP8 autocast is dropped, and both operands of every TE GEMM are MXFP8
   fake-quantized along their reduction dim (``_mxfp8_emulation``) and multiplied in FP32.
   The run is teacher-forced: slime's R3 routing replay reproduces the routing, and every
   TE module gets the real run's input (``rec + (x - x.detach())`` is exactly ``rec``) and
   grad_output.

Each TE module's output, input grad and weight grad then differ only by accumulation
order. Without forcing, the comparison is dominated by quantization amplification: a tiny
upstream difference pushes some elements across E4M3 rounding boundaries at every
re-quantization. A free run (routing replay only) is reported for context.

Rank 0 writes a JSON report to ``--out``.
"""

import contextlib
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model

from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.cp_utils import slice_with_cp
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _mxfp8_emulation as emu  # noqa: E402

TE_GEMM_WEIGHT = re.compile(
    r"(linear_qkv|linear_proj|in_proj_qkv|in_proj_z|out_proj|linear_fc1|linear_fc2)\.weight\d*$"
)


def extra_args(parser):
    parser.add_argument("--hf-checkpoint", type=str, required=True)
    parser.add_argument("--custom-model-provider-path", type=str, default=None)
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    parser.add_argument("--data-pad-size-multiplier", type=int, default=128)
    parser.add_argument("--seqlens", type=str, default="352,416,256")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--negative-control",
        choices=["none", "scale_roll", "swizzle"],
        default="none",
        help="Break the SM120 transpose on purpose: shift the E8M0 scales by one row, or "
        "plain-transpose GEMM-swizzled scales. The comparison must flag both.",
    )
    return parser


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    norm = b.norm()
    return ((a - b).norm() / norm).item() if norm > 0 else (a - b).norm().item()


def fake_quant(x, dim):
    """FP32 MXFP8 quantize-dequantize along ``dim`` of the 2D view of ``x``.

    Zero padding to multiples of 32 never changes another block's scale. FP32 operands make
    the emulated GEMM accumulate exactly: cuBLAS BF16 GEMMs of some shapes (e.g. M=N=1024,
    K=2048, or small per-expert M) add an extra BF16 rounding.
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, cols = x2.shape
    if rows == 0 or cols == 0:
        return x.float()
    pad_rows, pad_cols = (-rows) % 32, (-cols) % 32
    if pad_rows or pad_cols:
        x2 = F.pad(x2, (0, pad_cols, 0, pad_rows))
    return emu.fake_quantize(x2, dim, dtype=torch.float32)[:rows, :cols].reshape(shape)


def emulate_operands(A, B, layout):
    # TE layouts name (transa, transb): A reduces over dim 1 when transposed, B when not.
    dim_a = 1 if layout[0] == "T" else 0
    dim_b = 1 if layout[1] == "N" else 0
    if isinstance(A, list):
        return [fake_quant(a, dim_a) for a in A], [fake_quant(b, dim_b) for b in B]
    return fake_quant(A, dim_a), fake_quant(B, dim_b)


def fp32_buffer(out, accumulate):
    if out.dtype == torch.float32:
        return out
    return out.float() if accumulate else torch.empty(out.shape, dtype=torch.float32, device=out.device)


def emulated_gemm(gemm, A, B, layout, kwargs):
    assert kwargs.get("bias") is None and not kwargs.get("gelu") and kwargs.get("quantization_params") is None
    fa, fb = emulate_operands(A, B, layout)
    out = kwargs.get("out")
    kwargs = dict(kwargs, out_dtype=torch.float32)
    if out is None:
        result = gemm(fa, fb, layout=layout, **kwargs)
        return (result[0].to(A.dtype),) + tuple(result[1:])
    tmp = fp32_buffer(out, kwargs.get("accumulate", False))
    result = gemm(fa, fb, layout=layout, **dict(kwargs, out=tmp))
    if tmp is not out:
        out.copy_(tmp)
    return (out,) + tuple(result[1:])


def emulated_grouped_gemm(gemm, A, B, args, layout, kwargs):
    args = list(args)
    out = args[0]
    quantizers = args[1] if len(args) > 1 else kwargs.get("quantization_params")
    assert not kwargs.get("use_bias") and not kwargs.get("gelu") and all(q is None for q in quantizers or [])
    fa, fb = emulate_operands(A, B, layout)
    args[0] = [fp32_buffer(o, kwargs.get("accumulate", False)) for o in out]
    result = gemm(fa, fb, *args, layout=layout, **kwargs)
    for o, tmp in zip(out, args[0], strict=True):
        if tmp is not o:
            o.copy_(tmp)
    return (out,) + tuple(result[1:])


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    args = set_default_megatron_args(parse_args(extra_args))
    args.micro_batch_size = 1
    validate_args(args)
    assert args.fp8 and args.fp8_recipe == "mxfp8"
    assert os.environ.get("ENABLE_ROUTING_REPLAY") == "1", "routers only record/replay when this is set at build time"
    init(args)

    import megatron.core.transformer.transformer_block as mcore_block
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
    import transformer_engine.pytorch.module.linear as te_linear
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

    import slime.backends.megatron_utils.sm120_mxfp8 as sm120
    from slime.utils.routing_replay import RoutingReplay

    state = {"emulate": False, "force": None}  # force: None | "record" | "apply"
    gemm_calls = {}

    def instrument(gemm, kind):
        def wrapper(A, B, *args, layout="TN", **kwargs):
            is_mxfp8 = isinstance(A[0] if isinstance(A, list) else A, MXFP8TensorStorage)
            key = f"{kind}:{layout}:{'mxfp8' if is_mxfp8 else 'bf16'}"
            gemm_calls[key] = gemm_calls.get(key, 0) + 1
            if not state["emulate"]:
                return gemm(A, B, *args, layout=layout, **kwargs)
            assert not is_mxfp8, "the emulated run must not see MXFP8 operands"
            if kind == "grouped":
                return emulated_grouped_gemm(gemm, A, B, args, layout, kwargs)
            assert not args, "TE passes general_gemm options as keywords"
            return emulated_gemm(gemm, A, B, layout, kwargs)

        return wrapper

    te_linear.general_gemm = instrument(te_linear.general_gemm, "linear")
    te_layernorm_linear.general_gemm = instrument(te_layernorm_linear.general_gemm, "layernorm_linear")
    te_grouped_linear.general_grouped_gemm = instrument(te_grouped_linear.general_grouped_gemm, "grouped")

    if args.negative_control != "none":
        transpose = sm120.transpose_columnwise_to_rowwise

        def broken_transpose(tensor):
            out = transpose(tensor)
            if args.negative_control == "scale_roll":
                out._rowwise_scale_inv = out._rowwise_scale_inv.roll(1, dims=0).contiguous()
            elif tensor._with_gemm_swizzled_scales:
                out._rowwise_scale_inv = tensor._columnwise_scale_inv.t().contiguous()
            return out

        sm120.transpose_columnwise_to_rowwise = broken_transpose

    get_fp8_context = mcore_block.get_fp8_context
    mcore_block.get_fp8_context = lambda *a, **kw: (
        contextlib.nullcontext() if state["emulate"] else get_fp8_context(*a, **kw)
    )

    torch.manual_seed(1234)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=True)[0]
    gpt = model.module.module if hasattr(model.module, "module") else model.module
    te_modules = {
        name: m for name, m in gpt.named_modules() if isinstance(m, (te.Linear, te.LayerNormLinear, te.GroupedLinear))
    }

    # Sparse weight outliers and per-channel activation outliers, as in real checkpoints.
    gen = torch.Generator(device="cuda").manual_seed(4321 + dist.get_rank())
    with torch.no_grad():
        for m in te_modules.values():
            for p in m.parameters(recurse=False):
                if p.dim() == 2:
                    outliers = torch.rand(p.shape, device=p.device, generator=gen) < 0.005
                    p.mul_(torch.where(outliers, 32.0, 1.0).to(p.dtype))
        embedding = gpt.embedding.word_embeddings.weight
        embedding[:, torch.randperm(embedding.shape[1], device=embedding.device, generator=gen)[:8]] *= 20.0

    recorded_input, recorded_grad_output = {}, {}
    module_output, module_input_grad = {}, {}

    def add_hooks(name, module):
        def forward_pre(module, inputs):
            if state["force"] == "record":
                recorded_input[name] = inputs[0].detach().clone()
            elif state["force"] == "apply":
                x = inputs[0]
                return (recorded_input[name] + (x - x.detach()),) + tuple(inputs[1:])

        def forward(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            module_output[name] = out.detach().float().clone()

        def backward_pre(module, grad_output):
            if state["force"] == "record":
                recorded_grad_output[name] = tuple(g if g is None else g.detach().clone() for g in grad_output)
            elif state["force"] == "apply":
                return recorded_grad_output[name]

        def backward(module, grad_input, grad_output):
            if grad_input[0] is not None:
                module_input_grad[name] = grad_input[0].detach().float().clone()

        module.register_forward_pre_hook(forward_pre)
        module.register_forward_hook(forward)
        module.register_full_backward_pre_hook(backward_pre)
        module.register_full_backward_hook(backward)

    for name, m in te_modules.items():
        add_hooks(name, m)

    layer_outputs, routing = {}, {}
    for i, layer in enumerate(gpt.decoder.layers):
        layer.register_forward_hook(
            lambda m, inp, out, i=i: layer_outputs.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach().float().clone()
            )
        )
        layer.mlp.router.register_forward_hook(
            lambda m, inp, out, i=i: routing.__setitem__(i, out[1].detach().clone())
        )

    # slime's THD packing with CP zigzag slicing; every rank's token count is 32-aligned.
    gen = torch.Generator(device="cuda").manual_seed(7 + mpu.get_data_parallel_rank())
    seqlens = [int(s) for s in args.seqlens.split(",")]
    tokens = [slice_with_cp(torch.randint(0, 150000, (n,), device="cuda", generator=gen), 0) for n in seqlens]
    labels = [slice_with_cp(torch.randint(0, 150000, (n,), device="cuda", generator=gen), 0) for n in seqlens]
    cu_seqlens = [0]
    for t in tokens:
        cu_seqlens.append(cu_seqlens[-1] + t.size(0))
    tokens, labels = torch.cat(tokens), torch.cat(labels)
    pad_size = mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
    pad = (pad_size - tokens.size(0) % pad_size) % pad_size
    if pad:
        tokens, labels = F.pad(tokens, (0, pad)), F.pad(labels, (0, pad))
        cu_seqlens.append(cu_seqlens[-1] + pad)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device="cuda") * mpu.get_context_parallel_world_size()
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
    tokens, labels = tokens[None], labels[None]

    def step(emulate, force, replay):
        state["emulate"], state["force"] = emulate, force
        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward" if replay else "record"
        if replay:
            RoutingReplay.clear_all_forward()
        else:
            RoutingReplay.clear_all()
        model.zero_grad_buffer()
        for d in (layer_outputs, routing, module_output, module_input_grad, gemm_calls):
            d.clear()
        logits = model(input_ids=tokens, position_ids=None, attention_mask=None, labels=None, packed_seq_params=packed)
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.transpose(0, 1).float(), labels.t()).mean()
        loss.backward()
        grads = {}
        for name, p in model.named_parameters():
            g = getattr(p, "main_grad", None)
            g = p.grad if g is None else g
            if g is not None:
                grads[name] = g.detach().float().clone()
        return {
            "loss": loss.item(),
            "logits": logits.detach().float(),
            "layers": dict(layer_outputs),
            "routing": dict(routing),
            "outputs": dict(module_output),
            "input_grads": dict(module_input_grad),
            "grads": grads,
            "calls": dict(gemm_calls),
        }

    real = step(emulate=False, force="record", replay=False)
    forced = step(emulate=True, force="apply", replay=True)
    free = step(emulate=True, force=None, replay=True)

    weight_errors = {n: rel(forced["grads"][n], g) for n, g in real["grads"].items()}
    local = {
        "local_tokens": tokens.size(1),
        "finite": bool(torch.isfinite(real["logits"]).all())
        and all(bool(torch.isfinite(g).all()) for g in real["grads"].values()),
        "calls_real": real["calls"],
        "calls_emulated": forced["calls"],
        "loss_abs_diff": abs(forced["loss"] - real["loss"]),
        "logits_rel": rel(forced["logits"], real["logits"]),
        "routing_agreement": min(
            1.0 - (forced["routing"][i] != real["routing"][i]).any(-1).float().mean().item() for i in real["routing"]
        ),
        "te_output_rel": max(rel(forced["outputs"][n], real["outputs"][n]) for n in real["outputs"]),
        "te_dgrad_rel": max(rel(forced["input_grads"][n], real["input_grads"][n]) for n in real["input_grads"]),
        "te_wgrad_rel": max(e for n, e in weight_errors.items() if TE_GEMM_WEIGHT.search(n)),
        "other_grad_rel": max(e for n, e in weight_errors.items() if not TE_GEMM_WEIGHT.search(n)),
        "free_logits_rel": rel(free["logits"], real["logits"]),
        "free_layer_rel": max(rel(free["layers"][i], real["layers"][i]) for i in real["layers"]),
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() == 0:
        report = {
            "tp": mpu.get_tensor_model_parallel_world_size(),
            "sp": bool(args.sequence_parallel),
            "ep": mpu.get_expert_model_parallel_world_size(),
            "cp": mpu.get_context_parallel_world_size(),
            "dp": mpu.get_data_parallel_world_size(),
            "ranks": gathered,
        }
        Path(args.out).write_text(json.dumps(report, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
