import threading
import torch

class ModalityContext:
    """
    用于在推理过程中透传 input_ids，以便在 MoE Layer 内部判断 Vision/Text 模态。
    使用 ThreadLocal 确保并发安全。
    """
    _local = threading.local()

    @classmethod
    def set_input_ids(cls, input_ids: torch.Tensor):
        cls._local.input_ids = input_ids

    @classmethod
    def get_input_ids(cls):
        return getattr(cls._local, 'input_ids', None)

    @classmethod
    def clear(cls):
        if hasattr(cls._local, 'input_ids'):
            del cls._local.input_ids

    @classmethod
    def get_modality_mask(cls, current_device, seq_len=None):
        """
        返回 is_vision mask。
        True = Vision Token, False = Text Token
        """
        input_ids = cls.get_input_ids()
        if input_ids is None:
            # Fallback: 默认为 False (Text)
            if seq_len is None: seq_len = 1
            return torch.zeros((1, seq_len), dtype=torch.bool, device=current_device)
            
        # IMAGE_TOKEN_ID = 151655, VIDEO_TOKEN_ID = 151656
        #
        is_vision = (input_ids == 151655) | (input_ids == 151656)
        
        # 处理 Sequence Length 对齐 (针对 Decode 阶段)
        # 如果当前 forward 的 seq_len 小于 input_ids 的长度（说明是 generating），
        # 我们通常只取最后 seq_len 个 token 的模态信息。
        if seq_len is not None and is_vision.shape[1] != seq_len:
            if seq_len == 1:
                # Decode step: 取最后一个 token
                return is_vision[:, -1:].to(current_device)
            else:
                # Prefill step or chunked: 尝试取对应的切片，这里简化处理，直接截取
                # 实际生产中可能需要更复杂的 offset 逻辑
                return is_vision[:, -seq_len:].to(current_device)
            
        return is_vision.to(current_device)