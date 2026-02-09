# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_infinity.utils import ArcherConfig


class Qwen3VLMoeExpertMLP(nn.Module):
    """Single expert MLP, split from fused gate_up_proj."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SyncQwen3VLMoeSparseMoeBlock(nn.Module):
    archer_config: ArcherConfig = None
    layer_id: int = None

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size

        # router
        self.gate = nn.Linear(
            self.hidden_dim, self.num_experts, bias=False
        )

        # per-expert modules (replacing fused 3D tensors)
        self.experts = nn.ModuleList(
            [
                Qwen3VLMoeExpertMLP(
                    config.hidden_size, config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(
            dim=-1, keepdim=True
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        router_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        )
        routing_weights_mask = (
            routing_weights[:, :, None] * router_mask
        ).permute(0, 2, 1)
        routing_weights_mask = torch.sum(
            routing_weights_mask, dim=-1
        )
        router_mask = router_mask.permute(0, 2, 1)

        for i in range(self.top_k):
            router_mask[:, :, 0] = torch.logical_or(
                router_mask[:, :, 0], router_mask[:, :, i]
            )
        router_mask = router_mask[:, :, 0]

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        results = self.expert_executor.dispatch_local(
            hidden_states, router_mask, self.layer_id
        )
        for output, _, idx, _ in results:
            token_indices = router_mask[:, idx].bool()
            final_hidden_states[token_indices, :] += (
                output.to(routing_weights_mask.device)
                * routing_weights_mask[token_indices, idx][:, None]
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states
