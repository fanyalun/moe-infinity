import torch
import torch.nn as nn
from typing import Dict
from transformers import Qwen3VLMoeTextConfig
import os
import json
from typing import Optional, Tuple
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

def load_linear_weight(weight_path):
    """Load linear weights from a specific file"""
    if not os.path.exists(weight_path):
        print(f"⚠ Linear weight file not found: {weight_path}")
        return {}

    try:
        with open(weight_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        lw: Dict[str, float] = {}
        for item in data:
            layer = item.get("layer")
            expert_idx = item.get("expert_idx")
            weight = item.get("linear_weight", 0.0)
            
            if layer is None or expert_idx is None:
                continue
            
            key = f"{layer}_{expert_idx}"
            lw[key] = float(weight)
            
        print(f"✓ Loaded {len(lw)} weights from {weight_path}")
        return lw
    except Exception as e:
        print(f"⚠ Error loading linear weights: {e}")
        return {}

def load_expert_modality(modality_path):
    """Load expert modality from a specific file"""
    if not os.path.exists(modality_path):
        print(f"⚠ Expert modality file not found: {modality_path}")
        return {}

    try:
        with open(modality_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        em: Dict[str, int] = {}
        # 遍历二维数组结构
        for layer_idx, layer_modalities in enumerate(data):
            # 第一维是 layer 索引
            for expert_idx, modality_value in enumerate(layer_modalities):
                # 第二维是 expert 索引，值是 modality
                key = f"{layer_idx}_{expert_idx}"
                em[key] = int(modality_value)
            
        print(f"✓ Loaded {len(em)} expert modalities from {modality_path}")
        return em
    except Exception as e:
        print(f"⚠ Error loading expert modalities: {e}")
        return {}

class QwenMoeWrapperSkip(nn.Module):
    """
    Qwen MoE Wrapper (Skip Version)
    """

    def __init__(
        self,
        text_config: Qwen3VLMoeTextConfig,
        original_module: nn.Module,
        layer_id: int,
        global_config=None,
    ):
        super().__init__()

        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = original_module.gate  # nn.Linear
        self.experts = original_module.experts  # Qwen3VLMoeTextExperts

        if global_config is None:
            global_config = text_config

        self.image_token_id = getattr(global_config, "image_token_id", 151655)
        self.video_token_id = getattr(global_config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(global_config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(global_config, "vision_end_token_id", 151653)
        self.text_vocab_size = getattr(text_config, "vocab_size", 151936)

        raw_pad_id = getattr(global_config, "pad_token_id", None)
        if raw_pad_id is None and hasattr(global_config, "text_config"):
            raw_pad_id = getattr(global_config.text_config, "pad_token_id", None)

        self.pad_token_id = -1 if raw_pad_id is None else int(raw_pad_id)

        self.register_buffer(
            "linear_weight",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_modality",
            torch.zeros(self.num_experts, dtype=torch.int8),
            persistent=False,
        )

        self.register_buffer(
            "decode_keep_mask",
            torch.ones(self.num_experts, dtype=torch.bool),
            persistent=False,
        )

        self.register_buffer(
            "token_modality_flat",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )

        self._has_modality: bool = False

    @torch.no_grad()
    def set_linear_weight_from_dict(self, weight_dict: Dict[str, float]):
        device = self.linear_weight.device
        lw = torch.zeros(self.num_experts, dtype=self.linear_weight.dtype, device=device)
        em = torch.zeros(self.num_experts, dtype=self.expert_modality.dtype, device=device)

        prefix = f"{self.layer_id}_"
        for e in range(self.num_experts):
            key = prefix + str(e)
            if key in weight_dict:
                val = float(weight_dict[key])
                lw[e] = val
                if val > 0:
                    em[e] = 1   # vision expert
                elif val < 0:
                    em[e] = -1  # text expert
                else:
                    em[e] = 0   # balanced

        self.linear_weight.copy_(lw)
        self.expert_modality.copy_(em)
        self.decode_keep_mask = (em != 1)
        self._has_modality = bool((em != 0).any().item())

    @torch.no_grad()
    def set_token_modality_from_input_ids(self, input_ids: torch.Tensor):
        ids = input_ids
        mod = torch.zeros_like(ids, dtype=torch.int8)
        text_mask = (ids < self.text_vocab_size)
        mod[text_mask] = -1
        vision_mask = (ids == self.image_token_id) | (ids == self.video_token_id)
        mod[vision_mask] = 1
        special_mask = (ids == self.vision_start_token_id) | (ids == self.vision_end_token_id)
        if self.pad_token_id >= 0:
            special_mask = special_mask | (ids == self.pad_token_id)
        mod[special_mask] = 0
        self.token_modality_flat = mod.reshape(-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        device = hidden_states.device
        dtype = hidden_states.dtype

        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )

        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        if self._has_modality and seq_length > 1 and self.token_modality_flat.numel() == bs:
            K = self.top_k
            tok_mod_flat = self.token_modality_flat
            exp_mod = self.expert_modality

            tok_mod_slots = tok_mod_flat.unsqueeze(-1).expand(bs, K)
            exp_mod_slots = exp_mod[selected_experts]

            valid = (tok_mod_slots != 0) & (exp_mod_slots != 0)
            prod = (tok_mod_slots * exp_mod_slots).to(torch.int8)
            mismatch = valid & (prod == -1)

            sel_flat = selected_experts.reshape(-1)
            valid_flat = valid.reshape(-1)
            mismatch_flat = mismatch.reshape(-1)

            valid_int = valid_flat.to(torch.int32)
            mismatch_int = mismatch_flat.to(torch.int32)

            valid_count = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
            valid_count.scatter_add_(0, sel_flat, valid_int)

            mismatch_count = torch.zeros_like(valid_count)
            mismatch_count.scatter_add_(0, sel_flat, mismatch_int)

            skip_expert = (
                (valid_count > 0)
                & (mismatch_count == valid_count)
                & (exp_mod != 0)
            )
            zero_mask = skip_expert[selected_experts]
            routing_weights = routing_weights.masked_fill(zero_mask, 0.0)

            sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
            norm_mask = sum_w2 > 0
            sum_w2 = sum_w2.clamp_(min=1e-9)
            routing_weights = torch.where(
                norm_mask,
                routing_weights / sum_w2,
                routing_weights,
            )

        elif self._has_modality and seq_length == 1:
            zero_mask = ~self.decode_keep_mask[selected_experts]
            routing_weights = routing_weights.masked_fill(zero_mask, 0.0)

            sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
            norm_mask = sum_w2 > 0
            sum_w2 = sum_w2.clamp_(min=1e-9)
            routing_weights = torch.where(
                norm_mask,
                routing_weights / sum_w2,
                routing_weights,
            )

        routing_weights = routing_weights.to(dtype)
        router_logits = torch.zeros_like(router_logits).scatter_(1, selected_experts, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_dim)
        routed_out = self.experts(hidden_states, router_logits, selected_experts)
        return routed_out

class QwenMoeWrapperSkipAll(nn.Module):
    """
    Qwen MoE Wrapper (Skip Version)
    """

    def __init__(
        self,
        text_config: Qwen3VLMoeTextConfig,
        original_module: nn.Module,
        layer_id: int,
        global_config=None,
    ):
        super().__init__()

        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = original_module.gate  # nn.Linear
        self.experts = original_module.experts  # Qwen3VLMoeTextExperts

        if global_config is None:
            global_config = text_config

        self.image_token_id = getattr(global_config, "image_token_id", 151655)
        self.video_token_id = getattr(global_config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(global_config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(global_config, "vision_end_token_id", 151653)
        self.text_vocab_size = getattr(text_config, "vocab_size", 151936)

        raw_pad_id = getattr(global_config, "pad_token_id", None)
        if raw_pad_id is None and hasattr(global_config, "text_config"):
            raw_pad_id = getattr(global_config.text_config, "pad_token_id", None)

        self.pad_token_id = -1 if raw_pad_id is None else int(raw_pad_id)

        self.register_buffer(
            "linear_weight",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_modality",
            torch.zeros(self.num_experts, dtype=torch.int8),
            persistent=False,
        )

        self.register_buffer(
            "decode_keep_mask",
            torch.ones(self.num_experts, dtype=torch.bool),
            persistent=False,
        )

        self.register_buffer(
            "token_modality_flat",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )

        self._has_modality: bool = False

    @torch.no_grad()
    def set_linear_weight_from_dict(self, weight_dict: Dict[str, float]):
        device = self.linear_weight.device
        lw = torch.zeros(self.num_experts, dtype=self.linear_weight.dtype, device=device)
        em = torch.zeros(self.num_experts, dtype=self.expert_modality.dtype, device=device)

        prefix = f"{self.layer_id}_"
        for e in range(self.num_experts):
            key = prefix + str(e)
            if key in weight_dict:
                val = float(weight_dict[key])
                lw[e] = val
                if val > 0:
                    em[e] = 1   # vision expert
                elif val < 0:
                    em[e] = -1  # text expert
                else:
                    em[e] = 0   # balanced

        self.linear_weight.copy_(lw)
        self.expert_modality.copy_(em)
        self.decode_keep_mask = (em != 1)
        self._has_modality = bool((em != 0).any().item())

    @torch.no_grad()
    def set_token_modality_from_input_ids(self, input_ids: torch.Tensor):
        ids = input_ids
        mod = torch.zeros_like(ids, dtype=torch.int8)
        text_mask = (ids < self.text_vocab_size)
        mod[text_mask] = -1
        vision_mask = (ids == self.image_token_id) | (ids == self.video_token_id)
        mod[vision_mask] = 1
        special_mask = (ids == self.vision_start_token_id) | (ids == self.vision_end_token_id)
        if self.pad_token_id >= 0:
            special_mask = special_mask | (ids == self.pad_token_id)
        mod[special_mask] = 0
        self.token_modality_flat = mod.reshape(-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        device = hidden_states.device
        dtype = hidden_states.dtype

        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )

        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        if self._has_modality and seq_length > 1 and self.token_modality_flat.numel() == bs:
            K = self.top_k
            tok_mod_flat = self.token_modality_flat  # [bs]
            exp_mod = self.expert_modality           # [num_experts]
            
            # 扩展维度以便广播
            # tok_mod_flat: [bs] -> [bs, K]
            tok_mod_slots = tok_mod_flat.unsqueeze(-1).expand(bs, K)
            # exp_mod: [num_experts] -> 通过selected_experts索引得到 [bs, K]
            exp_mod_slots = exp_mod[selected_experts]
            
            # 计算不匹配掩码：token模态和专家模态都不为0且符号相反
            # 注意：我们只关心有明确模态的token和专家
            # 当token模态为1(视觉)且专家模态为-1(文本)时，不匹配
            # 当token模态为-1(文本)且专家模态为1(视觉)时，不匹配
            # 其他情况（至少一方为0或同号）视为匹配
            valid = (tok_mod_slots != 0) & (exp_mod_slots != 0)
            prod = (tok_mod_slots * exp_mod_slots).to(torch.int8)
            mismatch = valid & (prod == -1)  # 符号相反时为-1
            
            # 将不匹配的专家权重置为0
            routing_weights = routing_weights.masked_fill(mismatch, 0.0)
            
            # 重新归一化权重
            sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
            norm_mask = sum_w2 > 0
            sum_w2 = sum_w2.clamp_(min=1e-9)
            routing_weights = torch.where(
                norm_mask,
                routing_weights / sum_w2,
                routing_weights,  # 如果全部为0，保持原样（实际上不应该发生）
            )

        elif self._has_modality and seq_length == 1:
            # decode阶段保持原来的逻辑
            zero_mask = ~self.decode_keep_mask[selected_experts]
            routing_weights = routing_weights.masked_fill(zero_mask, 0.0)

            sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
            norm_mask = sum_w2 > 0
            sum_w2 = sum_w2.clamp_(min=1e-9)
            routing_weights = torch.where(
                norm_mask,
                routing_weights / sum_w2,
                routing_weights,
            )

        routing_weights = routing_weights.to(dtype)
        router_logits = torch.zeros_like(router_logits).scatter_(1, selected_experts, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_dim)
        routed_out = self.experts(hidden_states, router_logits, selected_experts)
        return routed_out

class QwenMoeWrapperSkipLogits(nn.Module):
    """
    Qwen MoE Wrapper (Skip Version)
    """

    def __init__(
        self,
        text_config: Qwen3VLMoeTextConfig,
        original_module: nn.Module,
        layer_id: int,
        global_config=None,
    ):
        super().__init__()

        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = original_module.gate  # nn.Linear
        self.experts = original_module.experts  # Qwen3VLMoeTextExperts

        if global_config is None:
            global_config = text_config

        self.image_token_id = getattr(global_config, "image_token_id", 151655)
        self.video_token_id = getattr(global_config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(global_config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(global_config, "vision_end_token_id", 151653)
        self.text_vocab_size = getattr(text_config, "vocab_size", 151936)

        raw_pad_id = getattr(global_config, "pad_token_id", None)
        if raw_pad_id is None and hasattr(global_config, "text_config"):
            raw_pad_id = getattr(global_config.text_config, "pad_token_id", None)

        self.pad_token_id = -1 if raw_pad_id is None else int(raw_pad_id)

        self.register_buffer(
            "linear_weight",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_modality",
            torch.zeros(self.num_experts, dtype=torch.int8),
            persistent=False,
        )

        self.register_buffer(
            "decode_keep_mask",
            torch.ones(self.num_experts, dtype=torch.bool),
            persistent=False,
        )

        self.register_buffer(
            "token_modality_flat",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )

        self._has_modality: bool = False

    @torch.no_grad()
    def set_linear_weight_from_dict(self, weight_dict: Dict[str, float]):
        device = self.linear_weight.device
        lw = torch.zeros(self.num_experts, dtype=self.linear_weight.dtype, device=device)
        em = torch.zeros(self.num_experts, dtype=self.expert_modality.dtype, device=device)

        prefix = f"{self.layer_id}_"
        for e in range(self.num_experts):
            key = prefix + str(e)
            if key in weight_dict:
                val = float(weight_dict[key])
                lw[e] = val
                if val > 0:
                    em[e] = 1   # vision expert
                elif val < 0:
                    em[e] = -1  # text expert
                else:
                    em[e] = 0   # balanced

        self.linear_weight.copy_(lw)
        self.expert_modality.copy_(em)
        self.decode_keep_mask = (em != 1)
        self._has_modality = bool((em != 0).any().item())

    @torch.no_grad()
    def set_token_modality_from_input_ids(self, input_ids: torch.Tensor):
        ids = input_ids
        mod = torch.zeros_like(ids, dtype=torch.int8)
        text_mask = (ids < self.text_vocab_size)
        mod[text_mask] = -1
        vision_mask = (ids == self.image_token_id) | (ids == self.video_token_id)
        mod[vision_mask] = 1
        special_mask = (ids == self.vision_start_token_id) | (ids == self.vision_end_token_id)
        if self.pad_token_id >= 0:
            special_mask = special_mask | (ids == self.pad_token_id)
        mod[special_mask] = 0
        self.token_modality_flat = mod.reshape(-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, seq_length, hidden_dim = hidden_states.shape
            bs = batch_size * seq_length
            device = hidden_states.device
            dtype = hidden_states.dtype

            hidden_states_flat = hidden_states.reshape(bs, hidden_dim)
            router_logits = self.gate(hidden_states_flat)

            routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
            # torch.topk 默认是有序的(sorted=True)，所以 routing_weights[:, 0] 是最大值，[:, -1] 是最小值
            routing_weights, selected_experts = torch.topk(
                routing_weights,
                self.top_k,
                dim=-1,
            )

            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(router_logits.dtype)

            if self._has_modality and seq_length > 1 and self.token_modality_flat.numel() == bs:
                K = self.top_k
                tok_mod_flat = self.token_modality_flat  # [bs]
                exp_mod = self.expert_modality           # [num_experts]
                
                # 扩展维度以便广播
                tok_mod_slots = tok_mod_flat.unsqueeze(-1).expand(bs, K)
                exp_mod_slots = exp_mod[selected_experts]
                
                # 1. 计算本来应该Skip的专家（Mismatch Mask）
                valid = (tok_mod_slots != 0) & (exp_mod_slots != 0)
                prod = (tok_mod_slots * exp_mod_slots).to(torch.int8)
                mismatch = valid & (prod == -1)  # [bs, K]
                
                # 2. 统计每个 token 应该丢弃多少个专家
                num_to_drop = mismatch.sum(dim=-1, keepdim=True) # [bs, 1]

                # 3. 生成基于 Rank 的 Mask
                # 我们要保留前 (K - num_to_drop) 个专家
                # 创建索引 [0, 1, ..., K-1]
                range_tensor = torch.arange(K, device=device).unsqueeze(0) # [1, K]
                # 如果 index < (K - num_to_drop)，则保留 (True)
                keep_threshold = K - num_to_drop
                rank_mask = range_tensor < keep_threshold # [bs, K]

                # 4. 应用 Mask：跳过等量的 Low Rank 专家
                routing_weights = routing_weights.masked_fill(~rank_mask, 0.0)

                # 重新归一化权重
                sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
                norm_mask = sum_w2 > 0
                sum_w2 = sum_w2.clamp_(min=1e-9)
                routing_weights = torch.where(
                    norm_mask,
                    routing_weights / sum_w2,
                    routing_weights, 
                )

            elif self._has_modality and seq_length == 1:
                # 1. 计算本来应该 Skip 的专家 (Decode阶段基于 decode_keep_mask)
                # 原始逻辑：zero_mask = ~self.decode_keep_mask[selected_experts]
                # 这里我们要算出有多少个专家属于"不应该保留"的类型
                mismatch_mask = ~self.decode_keep_mask[selected_experts] # [bs, K]
                
                # 2. 统计应该丢弃的数量
                num_to_drop = mismatch_mask.sum(dim=-1, keepdim=True) # [bs, 1]

                # 3. 生成基于 Rank 的 Mask
                K = self.top_k
                range_tensor = torch.arange(K, device=device).unsqueeze(0)
                keep_threshold = K - num_to_drop
                rank_mask = range_tensor < keep_threshold

                # 4. 应用 Mask：跳过等量的 Low Rank 专家
                routing_weights = routing_weights.masked_fill(~rank_mask, 0.0)

                sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
                norm_mask = sum_w2 > 0
                sum_w2 = sum_w2.clamp_(min=1e-9)
                routing_weights = torch.where(
                    norm_mask,
                    routing_weights / sum_w2,
                    routing_weights,
                )

            routing_weights = routing_weights.to(dtype)
            router_logits = torch.zeros_like(router_logits).scatter_(1, selected_experts, routing_weights)
            hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_dim)
            routed_out = self.experts(hidden_states, router_logits, selected_experts)
            return routed_out
            
class QwenMoeWrapperSkipAttn(nn.Module):
    """
    Qwen MoE Wrapper (Skip Version)
    """

    def __init__(
        self,
        text_config: Qwen3VLMoeTextConfig,
        original_module: nn.Module,
        layer_id: int,
        global_config=None,
    ):
        super().__init__()

        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = original_module.gate  # nn.Linear
        self.experts = original_module.experts  # Qwen3VLMoeTextExperts

        if global_config is None:
            global_config = text_config

        self.image_token_id = getattr(global_config, "image_token_id", 151655)
        self.video_token_id = getattr(global_config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(global_config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(global_config, "vision_end_token_id", 151653)
        self.text_vocab_size = getattr(text_config, "vocab_size", 151936)

        raw_pad_id = getattr(global_config, "pad_token_id", None)
        if raw_pad_id is None and hasattr(global_config, "text_config"):
            raw_pad_id = getattr(global_config.text_config, "pad_token_id", None)

        self.pad_token_id = -1 if raw_pad_id is None else int(raw_pad_id)

        self.register_buffer(
            "linear_weight",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_modality",
            torch.zeros(self.num_experts, dtype=torch.int8),
            persistent=False,
        )

        self.register_buffer(
            "decode_keep_mask",
            torch.ones(self.num_experts, dtype=torch.bool),
            persistent=False,
        )

        self.register_buffer(
            "token_modality_flat",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )
        self._has_modality: bool = False

    @torch.no_grad()
    def set_linear_weight_from_dict(self, weight_dict: Dict[str, float]):
        device = self.linear_weight.device
        lw = torch.zeros(self.num_experts, dtype=self.linear_weight.dtype, device=device)
        em = torch.zeros(self.num_experts, dtype=self.expert_modality.dtype, device=device)

        prefix = f"{self.layer_id}_"
        for e in range(self.num_experts):
            key = prefix + str(e)
            if key in weight_dict:
                val = float(weight_dict[key])
                lw[e] = val
                if val > 0:
                    em[e] = 1   # vision expert
                elif val < 0:
                    em[e] = -1  # text expert
                else:
                    em[e] = 0   # balanced

        self.linear_weight.copy_(lw)
        self.expert_modality.copy_(em)
        self.decode_keep_mask = (em != 1)
        self._has_modality = bool((em != 0).any().item())

    @torch.no_grad()
    def set_token_modality_from_input_ids(self, input_ids: torch.Tensor):
        ids = input_ids
        mod = torch.zeros_like(ids, dtype=torch.int8)
        text_mask = (ids < self.text_vocab_size)
        mod[text_mask] = -1
        vision_mask = (ids == self.image_token_id) | (ids == self.video_token_id)
        mod[vision_mask] = 1
        special_mask = (ids == self.vision_start_token_id) | (ids == self.vision_end_token_id)
        if self.pad_token_id >= 0:
            special_mask = special_mask | (ids == self.pad_token_id)
        mod[special_mask] = 0
        self.token_modality_flat = mod.reshape(-1)

    @torch.no_grad()
    def set_attn_weights(self, attn_weights: torch.Tensor):
        """根据注意力权重计算 token 重要性分数并筛选 top-2% 的 token"""
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        
        # sum over query dimension (dim=2) to get attention received by each key position
        token_attention_scores = attn_weights.sum(dim=2)  # [batch_size, num_heads, seq_len]

        # 在 head 维度求和
        token_attention_scores = token_attention_scores.sum(dim=1)  # [batch_size, seq_len]
                
        # 为每个 batch 筛选 top-2% 的 token
        batch_size, seq_len = token_attention_scores.shape
        top_k = max(1, int(seq_len * 0.02))  # 至少选择 1 个 token
        
        # 获取 top-k 的索引
        _, top_indices = token_attention_scores.topk(top_k, dim=1)  # [batch_size, top_k]
        
        # 创建重要 token 掩码
        self.important_token_mask = torch.zeros_like(token_attention_scores, dtype=torch.bool)
        self.important_token_mask.scatter_(1, top_indices, True)
        
        # # 可选：打印一些统计信息用于调试
        # if torch.cuda.current_device() == 0 or not torch.cuda.is_available():  # 只在第一个 GPU 或 CPU 上打印
        #     avg_importance = token_attention_scores.mean(dim=0)
        #     print(f"Layer {self.layer_id}: Top-{top_k} tokens selected out of {seq_len}")
        #     print(f"Average token importance: {avg_importance.mean().item():.4f}")

        # 将重要 token 的模态设置为 0（中性），使其不参与模态 skip
        if self.token_modality_flat.numel() > 0:
            modality_reshaped = self.token_modality_flat.view(batch_size, seq_len)
            modality_reshaped.masked_fill_(self.important_token_mask, 0)
            self.token_modality_flat = modality_reshaped.view(-1)
        del token_attention_scores
        torch.cuda.empty_cache()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        dtype = hidden_states.dtype

        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )

        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        if self.layer_id in [0, 1, 2, 3 ,46, 47]:
            # 第一二层不进行modality skip
            pass

        elif self._has_modality and seq_length > 1 and self.token_modality_flat.numel() == bs:
            K = self.top_k
            tok_mod_flat = self.token_modality_flat  # [bs]
            exp_mod = self.expert_modality           # [num_experts]
            
            # 扩展维度以便广播
            # tok_mod_flat: [bs] -> [bs, K]
            tok_mod_slots = tok_mod_flat.unsqueeze(-1).expand(bs, K)
            # exp_mod: [num_experts] -> 通过selected_experts索引得到 [bs, K]
            exp_mod_slots = exp_mod[selected_experts]
            
            # 计算不匹配掩码：token模态和专家模态都不为0且符号相反
            # 注意：我们只关心有明确模态的token和专家
            # 当token模态为1(视觉)且专家模态为-1(文本)时，不匹配
            # 当token模态为-1(文本)且专家模态为1(视觉)时，不匹配
            # 其他情况（至少一方为0或同号）视为匹配
            valid = (tok_mod_slots != 0) & (exp_mod_slots != 0)
            prod = (tok_mod_slots * exp_mod_slots).to(torch.int8)
            mismatch = valid & (prod == -1)  # 符号相反时为-1
            
            # 将不匹配的专家权重置为0
            routing_weights = routing_weights.masked_fill(mismatch, 0.0)
            
            # 重新归一化权重
            sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
            norm_mask = sum_w2 > 0
            sum_w2 = sum_w2.clamp_(min=1e-9)
            routing_weights = torch.where(
                norm_mask,
                routing_weights / sum_w2,
                routing_weights,  # 如果全部为0，保持原样（实际上不应该发生）
            )

        elif self._has_modality and seq_length == 1:
            # decode阶段保持原来的逻辑
            zero_mask = ~self.decode_keep_mask[selected_experts]
            routing_weights = routing_weights.masked_fill(zero_mask, 0.0)

            sum_w2 = routing_weights.sum(dim=-1, keepdim=True)
            norm_mask = sum_w2 > 0
            sum_w2 = sum_w2.clamp_(min=1e-9)
            routing_weights = torch.where(
                norm_mask,
                routing_weights / sum_w2,
                routing_weights,
            )

        routing_weights = routing_weights.to(dtype)
        router_logits = torch.zeros_like(router_logits).scatter_(1, selected_experts, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_dim)
        routed_out = self.experts(hidden_states, router_logits, selected_experts)
        return routed_out

# ==============================================================================
# 1. 专家计算 Wrapper (无需修改，保持原样)
# ==============================================================================
class Qwen3VLMoeTextExpertsWrapper(nn.Module):
    def __init__(self, original_module: nn.Module):
        super().__init__()
        self.num_experts = original_module.num_experts
        if hasattr(original_module, "moe_intermediate_size"):
            self.intermediate_size = original_module.moe_intermediate_size
        else:
            self.intermediate_size = original_module.intermediate_size
        self.hidden_size = original_module.hidden_size
        
        # 引用原始权重 (建议外部手动将这些权重移到 CPU 以测试模拟效果)
        self.gate_up_proj = original_module.gate_up_proj  
        self.down_proj = original_module.down_proj 
        self.act_fn = original_module.act_fn

        self.last_moe_compute_time = 0.0

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        hidden_states = hidden_states.reshape(-1, hidden_size)
        router_indices = router_indices.reshape(-1, router_indices.shape[-1])
        routing_weights = routing_weights.reshape(-1, self.num_experts)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_size), 
            dtype=dtype, 
            device=device
        )

        active_experts = torch.unique(router_indices)

        # --- Start Timing ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for expert_idx in active_experts:
            expert_idx = expert_idx.item()
            
            token_mask = (router_indices == expert_idx).any(dim=-1)
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            
            if token_indices.numel() == 0:
                continue

            current_states = hidden_states[token_indices]
            # 直接取对应专家的权重列
            current_weights = routing_weights[token_indices, expert_idx].unsqueeze(-1)

            # 模拟 Load
            w_gate_up = self.gate_up_proj[expert_idx].to(device, non_blocking=True)
            w_down = self.down_proj[expert_idx].to(device, non_blocking=True)

            # Compute
            gate_up = torch.matmul(current_states, w_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            gate = self.act_fn(gate)
            current_output = torch.matmul(gate * up, w_down)

            # Aggregate
            final_hidden_states.index_add_(0, token_indices, current_output * current_weights.to(dtype))
            
            del w_gate_up
            del w_down

        # --- End Timing ---
        end_event.record()
        self._start_event = start_event
        self._end_event = end_event

        return final_hidden_states.reshape(batch_size, seq_len, hidden_size)
    
    @property
    def get_last_compute_time(self):
        if hasattr(self, '_start_event') and hasattr(self, '_end_event'):
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        return 0.0
# ==============================================================================
# 2. Layer 计时 Wrapper (修复 MLP 和 Attention 的 Tuple 返回问题)
# ==============================================================================
class QwenLayerTimeStats(nn.Module):
    def __init__(self,original_module: nn.Module, layer_id: int):
        super().__init__()
        self.self_attn = original_module.self_attn
        self.mlp = original_module.mlp
        
        # 包装 experts
        if not isinstance(self.mlp.experts, Qwen3VLMoeTextExpertsWrapper):
            self.mlp.experts = Qwen3VLMoeTextExpertsWrapper(original_module=self.mlp.experts)   
        
        self.input_layernorm = original_module.input_layernorm
        self.post_attention_layernorm = original_module.post_attention_layernorm
        self.hidden_size = original_module.hidden_size
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        layer_start_ev = torch.cuda.Event(enable_timing=True)
        layer_end_ev = torch.cuda.Event(enable_timing=True)
        attn_start_ev = torch.cuda.Event(enable_timing=True)
        attn_end_ev = torch.cuda.Event(enable_timing=True)

        layer_start_ev.record()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # --- Attention ---
        attn_start_ev.record()
        
        attn_outputs, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # [Fix 1] 解包 Attention 输出 (output, weights)
        if isinstance(attn_outputs, tuple):
            print(f"[Layer {self.layer_id}] Attention returned a tuple, unpacking.")
            attn_output = attn_outputs[0]
        else:
            attn_output = attn_outputs
            
        attn_end_ev.record()
        
        hidden_states = residual + attn_output

        # --- MLP + MoE ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 调用 MLP
        mlp_outputs = self.mlp(hidden_states)
        
        # [Fix 2] 解包 MLP 输出
        # Qwen MoE MLP 通常返回 (hidden_states, router_logits)
        if isinstance(mlp_outputs, tuple):
            hidden_states = mlp_outputs[0]
            # 我们这里忽略 router_logits，因为只关心前向计算和计时
        else:
            hidden_states = mlp_outputs

        # 现在的 hidden_states 肯定是 Tensor，可以相加了
        hidden_states = residual + hidden_states
        
        layer_end_ev.record()

        # --- Print Stats (Decode only) ---
        if hidden_states.shape[1] == 1:
            torch.cuda.synchronize()
            
            attn_time = attn_start_ev.elapsed_time(attn_end_ev)
            layer_total_time = layer_start_ev.elapsed_time(layer_end_ev)
            moe_time = self.mlp.experts.get_last_compute_time
            
            other_time = layer_total_time - attn_time - moe_time
            
            print(f"[Layer {self.layer_id}] "
                  f"Total: {layer_total_time:.3f}ms | "
                  f"Attn: {attn_time:.3f}ms | "
                  f"MoE-Compute: {moe_time:.3f}ms | "
                  f"Other: {other_time:.3f}ms")

        # 这里的返回值取决于原始模型的期望。
        # 如果是中间层，最好返回 Tensor。如果是最后一层，或者有特殊需求，可能要调整。
        # 通常 Block 的 forward 只返回 hidden_states (Tensor) 或者 (hidden_states, cache)
        # 鉴于这是一个 TimeStats Wrapper，我们尽量模仿标准行为，返回 Tensor。
        return hidden_states

class QwenMoeDecoderLayerWrapper(nn.Module):
    def __init__(self, text_config: Qwen3VLMoeTextConfig, original_module: nn.Module, layer_id: int, global_config=None,):
        super().__init__()
        self.self_attn = original_module.self_attn
        self.mlp = QwenMoeWrapperSkipAttn(text_config, original_module=original_module.mlp, layer_id=layer_id, global_config=global_config)
        self.input_layernorm = original_module.input_layernorm
        self.post_attention_layernorm = original_module.post_attention_layernorm
        self.hidden_size = original_module.hidden_size
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
        # **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # # Prefill阶段，利用注意力权重信息更新moe模块的token重要性
        # # if hidden_states.shape[1] > 1 and self.layer_id not in [0, 1, 47]:
        # if hidden_states.shape[1] > 1 :
        #     self.mlp.set_attn_weights(attn_weights)
        # del attn_weights
        # torch.cuda.empty_cache()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

def custom_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = modeling_qwen3_vl_moe.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_qwen3_vl_moe.repeat_kv(value, module.num_key_value_groups)
            
    # 1. 计算原始logits
    attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # 2. 先对原始logits做softmax（用于返回）
    attn_weights_original = torch.nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query.dtype)

    # 3. 直接inplace修改attn_logits
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_logits.add_(causal_mask)  # ✓ inplace修改，节省显存

    # 4. 对修改后的logits做softmax（用于计算输出）
    attn_weights_masked = torch.nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights_masked = torch.nn.functional.dropout(attn_weights_masked, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights_masked, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights_original

class QwenMoeWrapperReplace(nn.Module):
    """
    Qwen MoE Wrapper (Pre-filtering Version)
    逻辑修改：在 Softmax/TopK 之前，直接屏蔽模态不匹配的专家 (Logits -> -inf)
    """

    def __init__(
        self,
        text_config: Qwen3VLMoeTextConfig,
        original_module: nn.Module,
        layer_id: int,
        global_config=None,
    ):
        super().__init__()

        self.hidden_dim = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = original_module.gate  # nn.Linear
        self.experts = original_module.experts  # Qwen3VLMoeTextExperts

        if global_config is None:
            global_config = text_config

        self.image_token_id = getattr(global_config, "image_token_id", 151655)
        self.video_token_id = getattr(global_config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(global_config, "vision_start_token_id", 151652)
        self.vision_end_token_id = getattr(global_config, "vision_end_token_id", 151653)
        self.text_vocab_size = getattr(text_config, "vocab_size", 151936)

        raw_pad_id = getattr(global_config, "pad_token_id", None)
        if raw_pad_id is None and hasattr(global_config, "text_config"):
            raw_pad_id = getattr(global_config.text_config, "pad_token_id", None)

        self.pad_token_id = -1 if raw_pad_id is None else int(raw_pad_id)

        self.register_buffer(
            "linear_weight",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_modality",
            torch.zeros(self.num_experts, dtype=torch.int8),
            persistent=False,
        )

        # decode阶段通常认为是文本生成，因此需要屏蔽视觉专家(expert_modality == 1)
        self.register_buffer(
            "decode_vision_mask", 
            torch.zeros(self.num_experts, dtype=torch.bool),
            persistent=False,
        )

        self.register_buffer(
            "token_modality_flat",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )

        self._has_modality: bool = False

    @torch.no_grad()
    def set_linear_weight_from_dict(self, weight_dict: Dict[str, float]):
        device = self.linear_weight.device
        lw = torch.zeros(self.num_experts, dtype=self.linear_weight.dtype, device=device)
        em = torch.zeros(self.num_experts, dtype=self.expert_modality.dtype, device=device)

        prefix = f"{self.layer_id}_"
        for e in range(self.num_experts):
            key = prefix + str(e)
            if key in weight_dict:
                val = float(weight_dict[key])
                lw[e] = val
                if val > 0:
                    em[e] = 1   # vision expert
                elif val < 0:
                    em[e] = -1  # text expert
                else:
                    em[e] = 0   # balanced

        self.linear_weight.copy_(lw)
        self.expert_modality.copy_(em)
        
        # 预计算：哪些是视觉专家（用于decode阶段直接屏蔽）
        self.decode_vision_mask = (em == 1)
        self._has_modality = bool((em != 0).any().item())

    @torch.no_grad()
    def set_token_modality_from_input_ids(self, input_ids: torch.Tensor):
        ids = input_ids
        mod = torch.zeros_like(ids, dtype=torch.int8)
        text_mask = (ids < self.text_vocab_size)
        mod[text_mask] = -1
        vision_mask = (ids == self.image_token_id) | (ids == self.video_token_id)
        mod[vision_mask] = 1
        special_mask = (ids == self.vision_start_token_id) | (ids == self.vision_end_token_id)
        if self.pad_token_id >= 0:
            special_mask = special_mask | (ids == self.pad_token_id)
        mod[special_mask] = 0
        self.token_modality_flat = mod.reshape(-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        bs = batch_size * seq_length
        device = hidden_states.device
        dtype = hidden_states.dtype

        hidden_states_flat = hidden_states.reshape(bs, hidden_dim)
        
        # 1. 计算原始 Logits [bs, num_experts]
        router_logits = self.gate(hidden_states_flat)

        # =================================================================
        # NEW LOGIC: 在 Softmax 之前直接屏蔽不匹配的专家
        # =================================================================
        if self._has_modality:
            # Case 1: Prefill 阶段 (seq_len > 1)，利用 token_modality_flat 进行精细化屏蔽
            if seq_length > 1 and self.token_modality_flat.numel() == bs:
                tok_mod = self.token_modality_flat.unsqueeze(1)  # [bs, 1]
                exp_mod = self.expert_modality.unsqueeze(0)      # [1, num_experts]
                
                # 计算不匹配逻辑：
                # 只有当 (tok=1, exp=-1) 或 (tok=-1, exp=1) 时乘积为 -1
                # 0 (Balanced/Neutral) 乘任何数都是 0，不会被判为 -1
                mismatch_mask = (tok_mod * exp_mod) == -1
                
                # 将不匹配专家的 logits 设为 -inf
                router_logits = router_logits.masked_fill(mismatch_mask, -float('inf'))

            # Case 2: Decode 阶段 (seq_len == 1)，默认假设为文本生成
            # 只有当 seq_length == 1 且没有详细的 modality info 时进入此分支
            # 此时我们假设当前 token 是 Text (-1)，所以要屏蔽 Vision Experts (1)
            elif seq_length == 1:
                # 屏蔽所有视觉专家
                router_logits = router_logits.masked_fill(self.decode_vision_mask, -float('inf'))

        # 2. 正常的 Softmax 和 Top-K 流程
        # 此时 router_logits 中不匹配的专家已经是 -inf，Softmax 后概率为 0，不会被选中
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )

        # 3. 归一化 (以防万一，虽然 Softmax 后选出的 Top-K 和应该接近 1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(dtype)

        # 4. 执行专家计算
        # 构建稀疏后的 logits 用于后续可能的辅助 loss 计算（虽然这里只传了 weights）
        final_logits_sparse = torch.zeros_like(router_logits).scatter_(1, selected_experts, routing_weights)
        
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_dim)
        routed_out = self.experts(hidden_states, final_logits_sparse, selected_experts)
        
        return routed_out