import re

from ...sglang import mxfp8_group_quantize

# Same scope as the official Qwen3.5 FP8 checkpoints: linear-attention in_proj_a/in_proj_b,
# conv1d, routers and shared_expert_gate stay BF16 and are listed in `ignored_layers`.
_QUANTIZED_DENSE_PARAMS = {
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
    "self_attention.linear_attn.in_proj_qkv.weight",
    "self_attention.linear_attn.in_proj_z.weight",
    "self_attention.linear_attn.out_proj.weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
}


def quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "mxfp8"
    assert quantization_config["activation_scheme"] == "dynamic"

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", megatron_name)
    if match:
        rest = match.group(2)
    else:
        match = re.match(r"module\.module\.mtp\.layers\.(\d+)\.(.+)", megatron_name)
        if not match:
            return converted_named_params
        rest = match.group(2).replace("transformer_layer.", "")

    expert_match = re.match(r"mlp\.experts\.(linear_fc[12])\.weight\d+", rest)
    shared_expert_match = re.match(r"mlp\.shared_experts\.(linear_fc[12])\.weight", rest)
    if not (expert_match or shared_expert_match or rest in _QUANTIZED_DENSE_PARAMS):
        return converted_named_params

    quantized_named_params = []
    for converted_name, param in converted_named_params:
        if not converted_name.endswith(".weight"):
            quantized_named_params.append((converted_name, param))
            continue
        qweight, scale = mxfp8_group_quantize(param.contiguous())
        quantized_named_params.append((converted_name, qweight))
        quantized_named_params.append((converted_name.replace(".weight", ".weight_scale_inv"), scale))
    return quantized_named_params
