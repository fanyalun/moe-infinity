from transformers.generation.streamers import BaseStreamer
import time
import torch

class TimingStreamer(BaseStreamer):
    """端到端计时工具（适用于batch_size==1）"""
    def __init__(self):
        super().__init__()
        self.start_time = None       # 整个流程开始时间（由外部设置）
        self.first_token_time = None # 第一个 新生成 token 的时间
        self.last_token_time = None  # 最后一个 token 的时间
        self.num_new_tokens = 0      # 仅统计新生成的 token 数量

    def put(self, value):
        # [关键修改] 过滤掉 Prompt 输入
        # generate 传来的 prompt 通常 shape 为 (1, seq_len > 1)
        # 或者是 list 形式。这里假设 batch_size=1，value 是 tensor
        if isinstance(value, torch.Tensor):
            if value.dim() > 1 and value.shape[-1] > 1:
                return  # 这是 Prompt，跳过
            
        # [可选] 为了获得更精准的 token 时间，建议在这里同步 CUDA
        # 否则 CPU 可能会跑在 GPU 前面，导致计时不准
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t = time.perf_counter()
        
        # 只有遇到第一个“新”token时才记录 first_token_time
        if self.first_token_time is None:
            self.first_token_time = t
        
        self.last_token_time = t
        self.num_new_tokens += 1

    def end(self):
        pass

    def get_metrics(self):
        if self.start_time is None or self.first_token_time is None or self.num_new_tokens == 0:
            return 0.0, 0.0, 0
        
        # TTFT: 从开始处理到吐出第一个新 token 的时间
        # 这包含了：图像处理 + 数据搬运 + 模型 Prefill + 生成第1个token
        ttft = self.first_token_time - self.start_time
        
        # TPOT: (最后一个token时间 - 第一个token时间) / (新token数 - 1)
        # 这才是纯粹的 Decode 阶段速度
        if self.num_new_tokens > 1:
            tpot = (self.last_token_time - self.first_token_time) / (self.num_new_tokens - 1)
        else:
            tpot = 0.0
            
        return ttft, tpot, self.num_new_tokens