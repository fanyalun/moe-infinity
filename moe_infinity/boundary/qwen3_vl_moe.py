import torch
from moe_store.parsing.hf_config import moe_text_config


def normalize_expert_tensor(
    tensor: torch.Tensor, config: object, projection: str
) -> torch.Tensor:
    text = moe_text_config(config)
    experts = int(text.num_experts)
    hidden = int(text.hidden_size)
    intermediate = int(text.moe_intermediate_size)
    if projection == "gate_up_proj":
        expected = (experts, 2 * intermediate, hidden)
    elif projection == "down_proj":
        expected = (experts, hidden, intermediate)
    else:
        raise ValueError(f"Unknown Qwen3-VL expert projection {projection!r}")
    actual = tuple(tensor.shape)
    if actual == expected:
        return tensor
    transposed = (expected[0], expected[2], expected[1])
    if actual == transposed:
        # Published checkpoints and newer HF exports use opposite matrix axes.
        return tensor.transpose(1, 2)
    raise ValueError(
        f"Qwen3-VL {projection} shape {actual} does not match "
        f"{expected} or {transposed}"
    )
