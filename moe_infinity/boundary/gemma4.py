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
        raise ValueError(f"Unknown Gemma4 expert projection {projection!r}")
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Gemma4 {projection} shape {tuple(tensor.shape)} does not match {expected}"
        )
    return tensor
