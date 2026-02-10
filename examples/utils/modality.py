# qwen3_vl_moe_test/utils/modality.py

import threading
import torch

class ModalityContext:
    """
    用于在推理过程中透传 input_ids 和 attention weights，
    以便在 MoE Layer 内部判断 Vision/Text 模态以及 Token 重要性。
    使用 ThreadLocal 确保并发安全。
    
    [Phase 4 Optimized] 增加了 Mask 缓存机制，避免每层重复计算。
    [Phase 5] 增加了 Attention Weights 传递。
    """
    _local = threading.local()

    @classmethod
    def set_input_ids(cls, input_ids: torch.Tensor):
        cls._local.input_ids = input_ids
        # [Phase 4] 输入变更时，强制清除缓存
        cls._clear_cache()

    @classmethod
    def get_input_ids(cls):
        return getattr(cls._local, 'input_ids', None)

    @classmethod
    def set_attn_weights(cls, weights: torch.Tensor):
        """
        [Phase 5] 设置当前层的 Attention Weights。
        weights: [Batch_Size * Seq_Len] (已展平) 或 [Batch_Size, Seq_Len]
        含义: 当前层最后一个 Token 对序列中每个 Token 的平均注意力分数。
        """
        cls._local.attn_weights = weights

    @classmethod
    def get_attn_weights(cls):
        return getattr(cls._local, 'attn_weights', None)

    @classmethod
    def clear(cls):
        if hasattr(cls._local, 'input_ids'):
            del cls._local.input_ids
        # [Phase 5] 清理 Attention 权重
        if hasattr(cls._local, 'attn_weights'):
            del cls._local.attn_weights
        # [Phase 4] 清理上下文时，同时清理缓存
        cls._clear_cache()

    @classmethod
    def _clear_cache(cls):
        """[Phase 4] 辅助函数：清理当前线程的缓存"""
        if hasattr(cls._local, 'cached_result'):
            del cls._local.cached_result

    @classmethod
    def get_modality_mask(cls, current_device, seq_len=None):
        """
        返回 is_vision mask。
        True = Vision Token, False = Text Token
        """
        # =========================================================
        # [Phase 4] Cache Hit Check
        # =========================================================
        # 检查 ThreadLocal 中是否有缓存，且参数（device, seq_len）是否匹配
        cached = getattr(cls._local, 'cached_result', None)
        if cached is not None:
            c_device, c_seq_len, c_mask = cached
            # 注意：这里我们假设只要 input_ids 没变(没调用set_input_ids)，
            # 且请求的 seq_len 和 device 一致，Mask 就无需重算。
            if c_device == current_device and c_seq_len == seq_len:
                return c_mask

        # =========================================================
        # Cache Miss: Execute Original Logic
        # =========================================================
        input_ids = cls.get_input_ids()
        
        # 临时变量存储计算结果
        mask = None

        if input_ids is None:
            # Fallback: 默认为 False (Text)
            # 注意：保持原有逻辑对 seq_len 的处理
            fallback_len = seq_len if seq_len is not None else 1
            mask = torch.zeros((1, fallback_len), dtype=torch.bool, device=current_device)
        
        else:
            # IMAGE_TOKEN_ID = 151655, VIDEO_TOKEN_ID = 151656
            is_vision = (input_ids == 151655) | (input_ids == 151656)
            
            # 处理 Sequence Length 对齐 (针对 Decode 阶段)
            if seq_len is not None and is_vision.shape[1] != seq_len:
                if seq_len == 1:
                    # Decode step: 取最后一个 token
                    mask = is_vision[:, -1:].to(current_device)
                else:
                    # Prefill step or chunked: 切片
                    mask = is_vision[:, -seq_len:].to(current_device)
            else:
                mask = is_vision.to(current_device)

        # =========================================================
        # [Phase 4] Update Cache
        # =========================================================
        # 将结果存入 ThreadLocal，Key 为 (device, seq_len)
        cls._local.cached_result = (current_device, seq_len, mask)
        
        return mask