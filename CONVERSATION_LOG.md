# MoE-Infinity 开发对话记录

## 日期: 2026-02-10 ~ 2026-02-11

## 一、Qwen3-VL-MoE 模型支持（核心功能）

### 新增文件
- `moe_infinity/models/qwen3_vl_moe.py` — 自定义 MoE 层实现（SyncQwen3VLMoeSparseMoeBlock）
- `moe_infinity/utils/hf_config.py` — 新增 Qwen3-VL-MoE 的 config 解析（MoE 层数计算、expert ID 映射）
- `moe_infinity/common/constants.py` — 注册模型映射（Qwen3VLMoeForConditionalGeneration, type=5）
- `examples/qwen3_vl_moe_example.py` — 示例推理脚本

### model_offload.py 核心改动
- 导入并注册 `SyncQwen3VLMoeSparseMoeBlock`，monkey-patch transformers 的原始类
- 处理 composite config（`text_config`）的 `torch_dtype` 为 None 的情况
- 在 expert topology key 命名中添加 `qwen3_vl_moe` 支持
- `_offload_state_dict()` 中拆分 Qwen3 的 fused 3D expert 权重：
  - `gate_up_proj [num_experts, hidden, 2*moe_inter]` → 逐 expert 拆为 `gate_proj` + `up_proj`，转置为 `[out, in]`
  - `down_proj [num_experts, moe_inter, hidden]` → 逐 expert 转置为 `[hidden, moe_inter]`
- `clean_up()` 中恢复 Qwen3 VL MoE 的原始类

## 二、Bug 修复

### C++ 端
- `core/prefetch/archer_prefetch_handle.cpp`:
  - 修复 `ReleaseTensor` 中 mutex 双重解锁导致 segfault（AcquireTensor 后保持 mutex locked，由 ReleaseTensor 解锁）
  - 修复 `last_node_` 空指针解引用（添加 null check）
- `core/model/model_topology.cpp`:
  - `InitializeTopology` 中 `dense_nodes` 为空时的守卫
  - `std::max(1, ...)` 防止除零

### Python 端
- `model_offload.py`:
  - NLLB `lm_head.weight` 添加 `model_type == "nllb_moe"` 守卫（原代码对所有模型生效）
  - 参数名正则匹配：处理数字不在 `.digit.` 格式中的情况（如 `norm1`）
  - composite config 的 `torch_dtype` 为 None 时 fallback 到 `text_config.torch_dtype` 或 `bfloat16`
- `moe_infinity/utils/hf_config.py`:
  - `parse_expert_dtype` 同样处理 None dtype fallback

## 三、Benchmark 脚本

### 新增文件
- `examples/benchmark_qwen3_vl_moe.py` — 对标 hybrid 方案的性能测试
- `examples/utils/` — 工具模块（dataset, eval, modality, prompt, stats, timestreamer）
- `results/moe_infinity_benchmark/` — 测试结果

### 初步性能对比（19 个 MMStar 样本，图像参数修复前）
```
Avg TTFT: MoE-Infinity=2.7909s, Keep8=2.3236s, Speedup=1.2x (Keep8 更快)
Avg TPOT: MoE-Infinity=0.2680s, Keep8=0.2078s, Speedup=1.3x (Keep8 更快)
```

## 四、代码清理
- 移除 model_offload.py hooks 和 qwen3_vl_moe.py 中的 debug prints
- 移除 hooks 中的注释代码块

## 五、已回退的改动（Dense Bypass 方案）

### 实施后回退的原因
曾实现 dense 参数绕过 OffloadEngine 直接常驻 GPU 的方案，包括：
- `_is_expert_param()` 判断参数类型
- `_offload_state_dict()` 中分离 dense 参数，保存为 `dense_shard_*.pt`
- `_load_dense_params()` 从 shard 文件加载 dense 参数到 GPU
- hooks 中 `offload_exemption` 跳过 dense 参数的 begin/end

### 回退原因
经过深入分析 C++ 引擎的驱逐策略后发现：
1. **原始设计已正确处理 dense 参数**：`InitializeTopology` 在初始化时将所有 dense nodes 移到 GPU
2. **Dense 和 Sparse 使用独立的驱逐池**，互不干扰
3. Dense 驱逐的 cache 上限是 `total_memory × 0.7`（硬编码），只要 dense 参数总量不超过 GPU 显存的 70%，就永远不会被驱逐
4. 绕过 OffloadEngine 反而导致 C++ 引擎不知道 dense 参数占用的显存，引发 OOM

## 六、MoE-Infinity 驱逐策略总结

### 节点分类
- **Dense nodes**: topology 中只有 1 个 tensor group 的 stage（attention、norm、embedding 等）
- **Sparse nodes**: topology 中有 >1 个 tensor group 的 stage（expert 层）

### 显存分配
- `device_memory_ratio` → `memory_capacity = total_GPU_memory × ratio`
- `sparse_budget = memory_capacity - dense_nodes_on_GPU`
- Dense 优先占 GPU，Sparse 用剩余空间

### 驱逐策略对比
| | Sparse 驱逐 | Dense 驱逐 |
|---|---|---|
| 函数 | `RemoveCachedSparseNode` | `RemoveCachedDenseNode` |
| cache 上限 | `memory_capacity - dense_on_GPU` | `total_memory × 0.7`（硬编码） |
| 策略 | LFU（按 `incache_visit_count`） | 按 `corr_id` 层序 |
| 保护机制 | `candidates_` + `exec_queue` | 无 |

### begin/end 快速路径
当 node 已在 GPU 时，`begin()` 走快速路径（`StartExec` 直接 return），只有 mutex + 指针操作开销。

## 七、待解决问题

### 1. 为 Qwen3-VL-MoE 添加预取支持
需要在 `moe_infinity/models/qwen3_vl_moe.py` 的 `forward()` 中添加：
- `expert_predictor.predict()` 调用
- `expert_prefetcher.prefetch_experts()` 调用
参考 `moe_infinity/models/grok.py:61-68`。

### 2. 重新跑 benchmark
回退 dense bypass 后，需要重新跑 benchmark 获取正确的性能数据。

## 八、关键文件索引
- `moe_infinity/runtime/model_offload.py` — 核心调度引擎
- `moe_infinity/models/qwen3_vl_moe.py` — Qwen3 VL MoE 自定义层
- `moe_infinity/utils/hf_config.py` — 模型配置解析
- `moe_infinity/common/constants.py` — 模型注册
- `core/model/model_topology.cpp` — C++ topology 初始化
- `core/prefetch/task_scheduler.cpp` — C++ 驱逐策略实现
- `core/prefetch/archer_prefetch_handle.cpp` — C++ begin/end 实现
- `examples/benchmark_qwen3_vl_moe.py` — 性能测试脚本
- `moe_infinity/models/grok.py` — 预取实现参考



## 新的对话记录（待总结）
 ▐▛███▜▌   Claude Code v2.1.37
▝▜█████▛▘  Opus 4.6 · API Usage Billing
  ▘▘ ▝▝    ~/moe-infinity

  Welcome to Opus 4.6

✻ Conversation compacted (ctrl+o for history)

  ⎿  Read examples/benchmark_qwen3_vl_moe.py (409 lines)
  ⎿  Read moe_infinity/runtime/model_offload.py (1313 lines)
  ⎿  Read ../model/config.json (69 lines)
  ⎿  Read moe_infinity/memory/expert_tracer.py (140 lines)
  ⎿  Read core/memory/device_caching_allocator.cpp (130 lines)

● 主人，我继续之前的调查。需要先看一下当前的代码状态，然后在推理循环中添加更精确的诊断来定位那40GB的显存跳变。

● Read 2 files (ctrl+o to expand)

● Let me read more of the key code sections to understand the flow.

● Searched for 2 patterns, read 5 files (ctrl+o to expand)

● Let me look at the MoE.generate() method and the _pre_forward_module_hook more carefully.

● Searched for 5 patterns, read 9 files (ctrl+o to expand)

● Now I have a clear picture of the code. The cache_sizes_ bug (line 264 in expert_dispatcher.cpp) causes the counter to go negative but doesn't directly
   cause unbounded GPU memory growth because eviction still works (one-for-one swap). The DeviceCachingAllocator reuses freed memory.

  The 40GB jump must come from somewhere else. I need to add precise diagnostics inside the inference loop to pinpoint exactly when it happens. Let me
  also check if there's something in the _pre_forward_module_hook that could be copying sparse params to GPU.

  Let me add targeted diagnostics to the benchmark script and the generate method.

● Read 1 file (ctrl+o to expand)

● Now let me add diagnostics inside the MoE.generate() method and at key points in the inference loop.

● Searched for 3 patterns, read 4 files (ctrl+o to expand)

● OK, I've thoroughly analyzed the code. The cache_sizes_ bug in expert_dispatcher.cpp:264 causes the counter to go negative but doesn't directly cause
  unbounded memory growth (eviction still works one-for-one). The DeviceCachingAllocator reuses freed memory.

  I need to add precise diagnostics to pinpoint the 40GB jump. Let me add tracking at key points in the first inference iteration.

● Update(examples/benchmark_qwen3_vl_moe.py)
  ⎿  Added 23 lines, removed 1 line
      220          desc="Inference",
      221      ):
      222          try:
      223 +            if idx == 0:                                                                                                               
      224 +                _bm_diag("inf[0] before build_prompt")                                                                                 
      225              messages = build_prompt(row, img_root)
      226              text = processor.apply_chat_template(
      227                  messages,
     ...
      239                  videos, video_metadatas = zip(*videos)
      240                  videos = list(videos)
      241                  video_metadatas = list(video_metadatas)
      242 +            if idx == 0:                                                                                                               
      243 +                _bm_diag("inf[0] before processor")                                                                                    
      244              inputs = processor(
      245                  text=text,
      246                  images=images,
     ...
      249                  min_pixels=768 * 28 * 28,
      250                  return_tensors="pt",
      251                  **(video_kwargs or {}),
      252 -            ).to("cuda:0")                                                                                                             
      252 +            )                                                                                                                          
      253 +            if idx == 0:                                                                                                               
      254 +                _bm_diag("inf[0] before .to(cuda)")                                                                                    
      255 +                for k, v in inputs.items():                                                                                            
      256 +                    if hasattr(v, 'shape'):                                                                                            
      257 +                        print(                                                                                                         
      258 +                            f"  input[{k}]: "                                                                                          
      259 +                            f"shape={list(v.shape)}, "                                                                                 
      260 +                            f"dtype={v.dtype}, "                                                                                       
      261 +                            f"dev={v.device}",                                                                                         
      262 +                            flush=True,                                                                                                
      263 +                        )                                                                                                              
      264 +            inputs = inputs.to("cuda:0")                                                                                               
      265 +            if idx == 0:                                                                                                               
      266 +                _bm_diag("inf[0] after .to(cuda)")                                                                                     
      267
      268              if torch.cuda.is_available():
      269                  torch.cuda.synchronize()
      270              streamer = StopWatch(moe.engine)
      271              streamer.start_time = time.perf_counter()
      272
      273 +            if idx == 0:                                                                                                               
      274 +                _bm_diag("inf[0] before moe.generate")                                                                                 
      275              generated_ids = moe.generate(
      276                  **inputs,
      277                  max_new_tokens=args.max_new_tokens,
      278                  do_sample=False,
      279                  streamer=streamer,
      280              )
      281 +            if idx == 0:                                                                                                               
      282 +                _bm_diag("inf[0] after moe.generate")                                                                                  
      283
      284              if torch.cuda.is_available():
      285                  torch.cuda.synchronize()

● Now let me add diagnostics inside MoE.generate() to track memory before/after model.generate().

● Update(moe_infinity/entrypoints/big_modeling.py)
  ⎿  Added 8 lines
      191          self._configure_hook(input_ids)
      192
      193          self.model.eval()
      194 +        if not hasattr(self, '_gen_call_count'):                                                                                       
      195 +            self._gen_call_count = 0                                                                                                   
      196 +        self._gen_call_count += 1                                                                                                      
      197 +        _diag = self._gen_call_count <= 6                                                                                              
      198 +        if _diag:                                                                                                                      
      199 +            _diag_gen(f"generate#{self._gen_call_count} before model.generate")                                                        
      200          with torch.no_grad():
      201              output = self.model.generate(input_ids, **kwargs)
      202 +        if _diag:                                                                                                                      
      203 +            _diag_gen(f"generate#{self._gen_call_count} after model.generate")                                                         
      204
      205          # finish trace entries so they accumulate in trace_collection
      206          for seq_id in self.seq_id_list:

● Now I need to add the _diag_gen helper function at the top of big_modeling.py.

● Update(moe_infinity/entrypoints/big_modeling.py)
  ⎿  Added 15 lines
       5  import torch
       6  import transformers
       7  from accelerate.utils.versions import is_torch_version
       8 +                                                                                                                                        
       9 +                                                                                                                                        
      10 +def _diag_gen(label):                                                                                                                   
      11 +    if not torch.cuda.is_available():                                                                                                   
      12 +        return                                                                                                                          
      13 +    torch.cuda.synchronize()                                                                                                            
      14 +    free, total = torch.cuda.mem_get_info(0)                                                                                            
      15 +    used = (total - free) / (1024 ** 2)                                                                                                 
      16 +    pt_r = torch.cuda.memory_reserved(0) / (1024 ** 2)                                                                                  
      17 +    print(                                                                                                                              
      18 +        f"[GEN-DIAG] {label}: "                                                                                                         
      19 +        f"nvidia≈{used:.0f}MB, "                                                                                                        
      20 +        f"non-pt={used - pt_r:.0f}MB",                                                                                                  
      21 +        flush=True,                                                                                                                     
      22 +    )                                                                                                                                   
      23  from huggingface_hub import snapshot_download
      24  from transformers import AutoConfig
      25

● Now let me also add a diagnostic inside _pre_forward_module_hook to track when sparse params are being moved to GPU. This is the most likely culprit —
  I want to see if any large tensors are being .to("cuda") during inference.

● Read 1 file (ctrl+o to expand)

● Update(moe_infinity/runtime/model_offload.py)
  ⎿  Added 14 lines
      1075              count[0] = count[0] + 1
      1076              self._register_hooks_recursively(child, count=count)
      1077
      1078 +        _hook_diag_budget = [20]  # mutable counter                                                                                   
      1079 +                                                                                                                                      
      1080          @torch.no_grad()
      1081          def _pre_forward_module_hook(module, args, kwargs):
      1082              device_list = []
      1083
      1084              for name, param in module.named_parameters(recurse=False):
      1085                  if param.data.data_ptr() not in self.offload_set:
      1086 +                    sz = param.data.nelement() * param.data.element_size()                                                            
      1087 +                    if sz > 1024 and _hook_diag_budget[0] > 0:                                                                        
      1088 +                        _hook_diag_budget[0] -= 1                                                                                     
      1089 +                        print(                                                                                                        
      1090 +                            f"[HOOK-DIAG] .to(cuda) "                                                                                 
      1091 +                            f"mod={module.__class__.__name__} "                                                                       
      1092 +                            f"param={name} "                                                                                          
      1093 +                            f"shape={list(param.data.shape)} "                                                                        
      1094 +                            f"dev={param.data.device} "                                                                               
      1095 +                            f"bytes={sz}",                                                                                            
      1096 +                            flush=True,                                                                                               
      1097 +                        )                                                                                                             
      1098                      num_devices = torch.cuda.device_count()
      1099                      param.data = param.data.to(
      1100                          f"cuda:{num_devices-1}"

● Now let me commit these diagnostic changes.

● Bash(git diff --stat)
  ⎿   examples/benchmark_qwen3_vl_moe.py       | 24 +++++++++++++++++++++++-
      moe_infinity/entrypoints/big_modeling.py | 23 +++++++++++++++++++++++
      moe_infinity/runtime/model_offload.py    | 14 ++++++++++++++
      3 files changed, 60 insertions(+), 1 deletion(-)

● Bash(git add examples/benchmark_qwen3_vl_moe.py moe_infinity/entrypoints/big_modeling.py moe_infinity/runtime/model_offload.py && git commit -m "diag:
      add inference-…)
  ⎿  [dev 67dd7b6] diag: add inference-loop GPU memory tracking to pinpoint 40GB jump
      3 files changed, 60 insertions(+), 1 deletion(-)

● Bash(git push origin dev)
  ⎿  To github.com:fanyalun/moe-infinity.git
        de231b3..67dd7b6  dev -> dev

● 主人，我已经推送了新的诊断代码（commit 67dd7b6）。这次添加了三层追踪：

  1. benchmark 脚本：在第一个推理样本的每个关键步骤（build_prompt → processor → .to(cuda) → moe.generate）前后都打印显存，还会打印每个 input tensor 的
  shape/dtype/device
  2. MoE.generate() 内部：在 model.generate() 前后打印显存（前6次调用）
  3. _pre_forward_module_hook：当有 >1KB 的参数被 .to(cuda) 时，打印模块名、参数名、shape、设备（限20条）

  这样我们就能精确定位那 40GB 的跳变发生在哪个阶段。

  关于目前的分析结论：
  - expert_dispatcher.cpp:264 的 cache_sizes_ bug 确实存在（cache hit 也会递减），但它不会直接导致显存无限增长，因为 eviction 仍然是一进一出
  - offload_set 在 InitializeTopology 后确实是 stale 的，但 dense 参数已经在 GPU 上（.to(cuda) 是 no-op），expert 模块的 hook 不会触发（dispatch_local
   C++）
  - 所以 40GB 跳变的根因还需要这次诊断数据来确认

  请在服务器上 git pull 后运行 benchmark，把 [BM-DIAG]、[GEN-DIAG]、[HOOK-DIAG] 开头的输出发给我。

✻ Sautéed for 13m 53s

❯ [GPU-DIAG] before setup_archer_hooks: nvidia-smi≈547MB, pt_alloc=24MB, pt_reserved=24MB, non-pt(cudaMalloc)=523MB                                  
  [diag] setup_archer_hooks: registered=19218, skipped=2, offload_set size=19218                                                                     
    skipped: buf:model.visual.rotary_pos_emb.inv_freq                                                                                                
    skipped: buf:model.language_model.rotary_emb.inv_freq                                                                                            
  [GPU-DIAG] before set_topology: nvidia-smi≈547MB, pt_alloc=24MB, pt_reserved=24MB, non-pt(cudaMalloc)=523MB                                        
  [20260211 19:46:49.257183Z ][1712554 ][INFO  ]Dense nodes: count={}, total_size={}MB 86 3967 - model_topology.cpp:523                              
  [20260211 19:46:49.257251Z ][1712554 ][INFO  ]GPU free before dense load: {}MB 80605 - model_topology.cpp:525                                      
  [20260211 19:46:49.257255Z ][1712554 ][INFO  ]Moving dense parameters to GPU - model_topology.cpp:531                                              
  1 leftt                                                                                                                                            
  finished: 86/86                                                                                                                                    
  [20260211 19:46:56.414974Z ][1712554 ][INFO  ]GPU free after dense load: {}MB 76521 - model_topology.cpp:543                                       
  [20260211 19:46:56.414990Z ][1712554 ][INFO  ]DeviceMemoryPool free_memory: {}MB 3336 - model_topology.cpp:545                                     
  [20260211 19:46:56.414993Z ][1712554 ][INFO  ]Moving sparse parameters to CPU - model_topology.cpp:550                                             
  [20260211 19:46:56.415159Z ][1712554 ][INFO  ]Sparse nodes: count={}, total_size={}MB 6144 55296 - model_topology.cpp:555                          
  1 leftttt                                                                                                                                          
  finished: 6144/6144                                                                                                                                
  [20260211 19:48:50.114981Z ][1712554 ][INFO  ]Sparse cache limit: {}MB 3336 - model_topology.cpp:562                                               
  [GPU-DIAG] after set_topology (InitializeTopology): nvidia-smi≈4751MB, pt_alloc=24MB, pt_reserved=24MB, non-pt(cudaMalloc)=4727MB                  
  [GPU-DIAG] after gc+empty_cache: nvidia-smi≈4751MB, pt_alloc=24MB, pt_reserved=24MB, non-pt(cudaMalloc)=4727MB                                     
  [GPU-DIAG] after setup_archer_hooks: nvidia-smi≈4751MB, pt_alloc=24MB, pt_reserved=24MB, non-pt(cudaMalloc)=4727MB                                 
  [GPU-DIAG] param device=cpu: 0.4MB                                                                                                                 
  Model loaded                                                                                                                                       
  [BM-DIAG] after model load: nvidia≈4751MB, pt_alloc=24MB, pt_resv=24MB, non-pt=4727MB                                                              
  Warming up...                                                                                                                                      
  [BM-DIAG] before warmup: nvidia≈4751MB, pt_alloc=24MB, pt_resv=24MB, non-pt=4727MB                                                                 
  [GEN-DIAG] generate#1 before model.generate: nvidia≈4767MB, non-pt=4727MB                                                                          
  [GEN-DIAG] generate#1 after model.generate: nvidia≈14195MB, non-pt=12919MB                                                                         
  [BM-DIAG] after warmup sample 0: nvidia≈14195MB, pt_alloc=111MB, pt_resv=1276MB, non-pt=12919MB                                                    
  [GEN-DIAG] generate#2 before model.generate: nvidia≈14195MB, non-pt=12919MB                                                                        
  [GEN-DIAG] generate#2 after model.generate: nvidia≈15007MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 1: nvidia≈15007MB, pt_alloc=112MB, pt_resv=2086MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#3 before model.generate: nvidia≈15007MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#3 after model.generate: nvidia≈15007MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 2: nvidia≈15007MB, pt_alloc=112MB, pt_resv=2086MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#4 before model.generate: nvidia≈15007MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#4 after model.generate: nvidia≈15007MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 3: nvidia≈15007MB, pt_alloc=111MB, pt_resv=2086MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#5 before model.generate: nvidia≈15007MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#5 after model.generate: nvidia≈15007MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 4: nvidia≈15007MB, pt_alloc=111MB, pt_resv=2086MB, non-pt=12921MB                                                    
  [BM-DIAG] after warmup (all): nvidia≈15007MB, pt_alloc=111MB, pt_resv=2086MB, non-pt=12921MB                                                       
  Warmup completed                                                                                                                                   
  Inference:   0%|                                                                                                                                   
                            | 0/20 [00:00<?, ?it/s][BM-DIAG] inf[0] before build_prompt: nvidia≈15007MB, pt_alloc=111MB, pt_resv=2086MB,             
  non-pt=12921MB                                                                                                                                     
  [BM-DIAG] inf[0] before processor: nvidia≈15007MB, pt_alloc=111MB, pt_resv=2086MB, non-pt=12921MB                                                  
  [BM-DIAG] inf[0] before .to(cuda): nvidia≈15007MB, pt_alloc=97MB, pt_resv=2086MB, non-pt=12921MB                                                   
    input[input_ids]: shape=[1, 3808], dtype=torch.int64, dev=cpu                                                                                    
    input[attention_mask]: shape=[1, 3808], dtype=torch.int64, dev=cpu                                                                               
    input[pixel_values]: shape=[14840, 1536], dtype=torch.float32, dev=cpu                                                                           
    input[image_grid_thw]: shape=[1, 3], dtype=torch.int64, dev=cpu                                                                                  
  [BM-DIAG] inf[0] after .to(cuda): nvidia≈15007MB, pt_alloc=184MB, pt_resv=2086MB, non-pt=12921MB                                                   
  [BM-DIAG] inf[0] before moe.generate: nvidia≈15007MB, pt_alloc=184MB, pt_resv=2086MB, non-pt=12921MB                                               
  [GEN-DIAG] generate#6 before model.generate: nvidia≈15007MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#6 after model.generate: nvidia≈55335MB, non-pt=12921MB                                                                         
  [BM-DIAG] inf[0] after moe.generate: nvidia≈55335MB, pt_alloc=184MB, pt_resv=42414MB, non-pt=12921MB                                               

● 关键发现！数据非常清晰：

  generate#6 before model.generate: nvidia≈15007MB, non-pt=12921MB
  generate#6 after model.generate:  nvidia≈55335MB, non-pt=12921MB

  inf[0] after moe.generate: pt_alloc=184MB, pt_resv=42414MB, non-pt=12921MB

  non-pt（DeviceCachingAllocator）完全没变，一直是 12921MB。40GB 的跳变全部来自 PyTorch 的 pt_reserved：2086MB → 42414MB。

  而 pt_alloc 只有 184MB，说明 generate 结束后活跃张量很少，但 PyTorch 的 CUDACachingAllocator 保留了 42GB 的已释放内存。

  这意味着 forward pass 期间产生了巨大的中间张量。罪魁祸首几乎可以确定是 vision encoder 的 attention scores：

  pixel_values: shape=[14840, 1536]  →  14840 个 patch

  Vision encoder 的 attention scores：
  - [1, 16, 14840, 14840] × 4 bytes (fp32 softmax) ≈ 14.1GB 每层
  - 加上 softmax 后的 bf16 副本、QKV 投影等中间张量，峰值轻松超过 20GB+
  - PyTorch allocator 会保留这些内存不归还 CUDA

  warmup 阶段图片小，patch 数少，所以 attention 矩阵小得多，只用了 2GB。

● Explore(Find attn_implementation config)
  ⎿  Done (5 tool uses · 46.7k tokens · 44s)

● 找到根因了！在 big_modeling.py 中，qwen3vlmoe 被硬编码禁用了 flash attention：

  if (arch == "switch" or arch == "deepseek" or arch == "deepseek_v3"
      or arch == "nllb" or arch == "qwen3vlmoe"):
      is_flash_attn_available = False

  所以模型用的是 eager attention，对 14840 个 patch 会生成完整的 [1, 16, 14840, 14840] attention 矩阵 ≈ 14GB/层。这就是 42GB PyTorch 内存的来源。

  解决方案：启用 sdpa（PyTorch 内置的 Scaled Dot Product Attention），它会自动使用 flash attention 或 memory-efficient attention，避免 O(n²) 的显存开销

● Read 2 files (ctrl+o to expand)
  ⎿  Interrupted · What should Claude do instead?

❯ 但是我用别的框架也是用同样的eager attention，为什么没有这么大的显存呢？而且我本来就是要强制使用eager attention的，所以你的解决方案不可行 

✻ Sautéed for 2m 31s

❯ 但是我用别的框架也是用同样的eager attention，为什么没有这么大的显存呢？而且我本来就是要强制使用eager                                               
  attention的，所以你的解决方案不可行。你确定是因为这个问题吗？有没有其他的解决方案？                                                                

● 主人，你说得对，我重新分析一下数据。

  关键线索是：non-pt 完全没变（12921MB），42GB 全部是 pt_reserved。但 pt_alloc 只有 184MB，说明 generate 结束后这些内存已经释放了，只是 PyTorch 的
  CUDACachingAllocator 保留着。

  问题不在 eager attention 本身，而在于 两个 allocator 互相竞争导致的内存碎片化。

  MoE-Infinity 的 DeviceCachingAllocator 用 raw cudaMalloc 分配了 12921MB，这些块散布在 GPU 内存空间中。当 PyTorch 的 CUDACachingAllocator 需要为
  attention scores 分配大块连续内存时，被这些 cudaMalloc 块切割，无法复用上一层释放的内存，只能每层重新分配。

  Vision encoder 27 层，每层 attention scores ~14GB → 如果每层都无法复用，PyTorch 就需要不断分配新块，累积到 42GB。

  而别的框架只有一个 allocator，释放的内存可以被下一层复用，所以只需要 ~14GB 峰值。

  让我验证一下，然后修复。

● Read 1 file (ctrl+o to expand)

✻ Worked for 1m 22s

❯ 验证一下，然后修复，修复之前，先告诉我修复方案 

● 主人，让我先验证碎片化理论，再提出修复方案。

  先看一下关键数据对比：
  ┌──────────┬─────────┬─────────────┬──────────┐
  │   阶段   │ non-pt  │ pt_reserved │ pt_alloc │
  ├──────────┼─────────┼─────────────┼──────────┤
  │ warmup前 │ 4727MB  │ 24MB        │ 24MB     │
  ├──────────┼─────────┼─────────────┼──────────┤
  │ warmup后 │ 12921MB │ 2086MB      │ 111MB    │
  ├──────────┼─────────┼─────────────┼──────────┤
  │ 推理#6前 │ 12921MB │ 2086MB      │ 184MB    │
  ├──────────┼─────────┼─────────────┼──────────┤
  │ 推理#6后 │ 12921MB │ 42414MB     │ 184MB    │
  └──────────┴─────────┴─────────────┴──────────┘
  关键事实：
  - non-pt 完全不变 → expert cache 不是问题
  - pt_alloc=184MB → generate 结束后几乎没有活跃张量
  - pt_reserved 暴涨 40GB → PyTorch allocator 分配了大量内存但无法回收复用

  warmup 图片小（patch 少），pt_reserved 只到 2086MB。推理图片大（14840 patches），pt_reserved 飙到 42GB。

  碎片化验证方案：在 benchmark 脚本最前面加一行环境变量，让 PyTorch 使用可扩展段（expandable segments），绕过碎片化问题：

  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

  如果加了这行后 pt_reserved 大幅下降，就确认是碎片化问题。

  修复方案（确认后实施）：

  修改 core/memory/device_caching_allocator.cpp，把底层的 raw cudaMalloc 替换为 PyTorch 的 CUDACachingAllocator：

  // allocate_and_cache() 中:
  // 原: cudaMalloc(&ptr, bytes)
  // 改: ptr = c10::cuda::CUDACachingAllocator::raw_alloc(bytes)

  // free() 中 (未命中缓存时):
  // 原: cudaFree(ptr)
  // 改: c10::cuda::CUDACachingAllocator::raw_dealloc(ptr)

  // free_cached() 中:
  // 原: cudaFree(ptr)
  // 改: c10::cuda::CUDACachingAllocator::raw_dealloc(ptr)

  这样所有 GPU 内存都走 PyTorch 的统一 allocator，消除两个 allocator 互相竞争造成的碎片化。expert 权重的按大小缓存逻辑保持不变（DeviceCachingAllocator
   available_map_ 仍然有效），只是底层内存来源统一了。

  要先跑验证，还是直接改？

✻ Churned for 1m 13s

❯ 先验证 

● 好，加上 expandable_segments:True 验证碎片化理论。

● Read 1 file (ctrl+o to expand)

● Update(examples/benchmark_qwen3_vl_moe.py)
  ⎿  Added 3 lines
       4  import sys
       5  import time
       6
       7 +# verify: PyTorch expandable segments to bypass fragmentation                                                                           
       8 +os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"                                                                      
       9 +                                                                                                                                        
      10  import torch
      11  from tqdm import tqdm
      12  from transformers import AutoProcessor

● Bash(git add examples/benchmark_qwen3_vl_moe.py && git commit -m "test: add expandable_segments to verify allocator fragmentation theory…)
  ⎿  Error: Exit code 1
     To github.com:fanyalun/moe-infinity.git
      ! [rejected]        dev -> dev (non-fast-forward)
     error: failed to push some refs to 'github.com:fanyalun/moe-infinity.git'
     hint: Updates were rejected because the tip of your current branch is behind
     hint: its remote counterpart. If you want to integrate the remote changes,
     hint: use 'git pull' before pushing again.
     hint: See the 'Note about fast-forwards' in 'git push --help' for details.

     [dev 489b444] test: add expandable_segments to verify allocator fragmentation theory
     … +1 line (ctrl+o to see all)

● Bash(git pull --rebase origin dev && git push origin dev)
  ⎿  From github.com:fanyalun/moe-infinity
      * branch            dev        -> FETCH_HEAD
     Rebasing (1/1)Successfully rebased and updated refs/heads/dev.
     … +2 lines (ctrl+o to expand)

● 已推送（commit e1d27db）。请在服务器上 git pull 后运行 benchmark，重点看 [BM-DIAG] inf[0] after moe.generate 那行的 pt_resv 是否从 42414MB 大幅下降。

✻ Sautéed for 55s

❯ 显存占用并没有降低    [GPU-DIAG] before setup_archer_hooks: nvidia-smi≈563MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=523MB            
  [diag] setup_archer_hooks: registered=19218, skipped=2, offload_set size=19218                                                                     
    skipped: buf:model.visual.rotary_pos_emb.inv_freq                                                                                                
    skipped: buf:model.language_model.rotary_emb.inv_freq                                                                                            
  [GPU-DIAG] before set_topology: nvidia-smi≈563MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=523MB                                        
  [20260211 20:18:41.213919Z ][2018803 ][INFO  ]Dense nodes: count={}, total_size={}MB 86 3967 - model_topology.cpp:523                              
  [20260211 20:18:41.213973Z ][2018803 ][INFO  ]GPU free before dense load: {}MB 80589 - model_topology.cpp:525                                      
  [20260211 20:18:41.213976Z ][2018803 ][INFO  ]Moving dense parameters to GPU - model_topology.cpp:531                                              
  1 leftt                                                                                                                                            
  finished: 86/86                                                                                                                                    
  [20260211 20:18:48.921290Z ][2018803 ][INFO  ]GPU free after dense load: {}MB 76505 - model_topology.cpp:543                                       
  [20260211 20:18:48.921301Z ][2018803 ][INFO  ]DeviceMemoryPool free_memory: {}MB 3336 - model_topology.cpp:545                                     
  [20260211 20:18:48.921303Z ][2018803 ][INFO  ]Moving sparse parameters to CPU - model_topology.cpp:550                                             
  [20260211 20:18:48.923015Z ][2018803 ][INFO  ]Sparse nodes: count={}, total_size={}MB 6144 55296 - model_topology.cpp:555                          
  1 leftttt                                                                                                                                          
  finished: 6144/6144                                                                                                                                
  [20260211 20:20:31.721228Z ][2018803 ][INFO  ]Sparse cache limit: {}MB 3336 - model_topology.cpp:562                                               
  [GPU-DIAG] after set_topology (InitializeTopology): nvidia-smi≈4767MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=4727MB                  
  [GPU-DIAG] after gc+empty_cache: nvidia-smi≈4767MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=4727MB                                     
  [GPU-DIAG] after setup_archer_hooks: nvidia-smi≈4767MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=4727MB                                 
  [GPU-DIAG] param device=cpu: 0.4MB                                                                                                                 
  Model loaded                                                                                                                                       
  [BM-DIAG] after model load: nvidia≈4767MB, pt_alloc=23MB, pt_resv=40MB, non-pt=4727MB                                                              
  Warming up...                                                                                                                                      
  [BM-DIAG] before warmup: nvidia≈4767MB, pt_alloc=23MB, pt_resv=40MB, non-pt=4727MB                                                                 
  [GEN-DIAG] generate#1 before model.generate: nvidia≈4769MB, non-pt=4727MB                                                                          
  [GEN-DIAG] generate#1 after model.generate: nvidia≈14167MB, non-pt=12919MB                                                                         
  [BM-DIAG] after warmup sample 0: nvidia≈14167MB, pt_alloc=110MB, pt_resv=1248MB, non-pt=12919MB                                                    
  [GEN-DIAG] generate#2 before model.generate: nvidia≈14167MB, non-pt=12919MB                                                                        
  [GEN-DIAG] generate#2 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 1: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#3 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#3 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 2: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#4 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#4 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 3: nvidia≈14395MB, pt_alloc=110MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#5 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#5 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 4: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [BM-DIAG] after warmup (all): nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                       
  Warmup completed                                                                                                                                   
  Inference:   0%|                                                                                                                   | 0/2 [00:00<?, 
   ?it/s][BM-DIAG] inf[0] before build_prompt: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                        
  [BM-DIAG] inf[0] before processor: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                  
  [BM-DIAG] inf[0] before .to(cuda): nvidia≈14395MB, pt_alloc=97MB, pt_resv=1474MB, non-pt=12921MB                                                   
    input[input_ids]: shape=[1, 3808], dtype=torch.int64, dev=cpu                                                                                    
    input[attention_mask]: shape=[1, 3808], dtype=torch.int64, dev=cpu                                                                               
    input[pixel_values]: shape=[14840, 1536], dtype=torch.float32, dev=cpu                                                                           
    input[image_grid_thw]: shape=[1, 3], dtype=torch.int64, dev=cpu                                                                                  
  [BM-DIAG] inf[0] after .to(cuda): nvidia≈14395MB, pt_alloc=184MB, pt_resv=1474MB, non-pt=12921MB                                                   
  [BM-DIAG] inf[0] before moe.generate: nvidia≈14395MB, pt_alloc=184MB, pt_resv=1474MB, non-pt=12921MB                                               
  [GEN-DIAG] generate#6 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#6 after model.generate: nvidia≈54075MB, non-pt=12921MB                                                                         
  [BM-DIAG] inf[0] after moe.generate: nvidia≈54075MB, pt_alloc=184MB, pt_resv=41154MB, non-pt=12921MB                                               

✻ Churned for 2m 25s

❯ 显存占用并没有降低    [GPU-DIAG] before setup_archer_hooks: nvidia-smi≈563MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=523MB            
  [diag] setup_archer_hooks: registered=19218, skipped=2, offload_set size=19218                                                                     
    skipped: buf:model.visual.rotary_pos_emb.inv_freq                                                                                                    
    skipped: buf:model.language_model.rotary_emb.inv_freq                                                                                            
  [GPU-DIAG] before set_topology: nvidia-smi≈563MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=523MB                                            
  [20260211 20:18:41.213919Z ][2018803 ][INFO  ]Dense nodes: count={}, total_size={}MB 86 3967 - model_topology.cpp:523                                
  [20260211 20:18:41.213973Z ][2018803 ][INFO  ]GPU free before dense load: {}MB 80589 - model_topology.cpp:525                                      
  [20260211 20:18:41.213976Z ][2018803 ][INFO  ]Moving dense parameters to GPU - model_topology.cpp:531                                              
  1 leftt                                                                                                                                            
  finished: 86/86                                                                                                                                    
  [20260211 20:18:48.921290Z ][2018803 ][INFO  ]GPU free after dense load: {}MB 76505 - model_topology.cpp:543                                       
  [20260211 20:18:48.921301Z ][2018803 ][INFO  ]DeviceMemoryPool free_memory: {}MB 3336 - model_topology.cpp:545                                     
  [20260211 20:18:48.921303Z ][2018803 ][INFO  ]Moving sparse parameters to CPU - model_topology.cpp:550                                             
  [20260211 20:18:48.923015Z ][2018803 ][INFO  ]Sparse nodes: count={}, total_size={}MB 6144 55296 - model_topology.cpp:555                          
  1 leftttt                                                                                                                                          
  finished: 6144/6144                                                                                                                                
  [20260211 20:20:31.721228Z ][2018803 ][INFO  ]Sparse cache limit: {}MB 3336 - model_topology.cpp:562                                               
  [GPU-DIAG] after set_topology (InitializeTopology): nvidia-smi≈4767MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=4727MB                  
  [GPU-DIAG] after gc+empty_cache: nvidia-smi≈4767MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=4727MB                                     
  [GPU-DIAG] after setup_archer_hooks: nvidia-smi≈4767MB, pt_alloc=23MB, pt_reserved=40MB, non-pt(cudaMalloc)=4727MB                                 
  [GPU-DIAG] param device=cpu: 0.4MB                                                                                                                 
  Model loaded                                                                                                                                       
  [BM-DIAG] after model load: nvidia≈4767MB, pt_alloc=23MB, pt_resv=40MB, non-pt=4727MB                                                              
  Warming up...                                                                                                                                      
  [BM-DIAG] before warmup: nvidia≈4767MB, pt_alloc=23MB, pt_resv=40MB, non-pt=4727MB                                                                 
  [GEN-DIAG] generate#1 before model.generate: nvidia≈4769MB, non-pt=4727MB                                                                          
  [GEN-DIAG] generate#1 after model.generate: nvidia≈14167MB, non-pt=12919MB                                                                         
  [BM-DIAG] after warmup sample 0: nvidia≈14167MB, pt_alloc=110MB, pt_resv=1248MB, non-pt=12919MB                                                    
  [GEN-DIAG] generate#2 before model.generate: nvidia≈14167MB, non-pt=12919MB                                                                        
  [GEN-DIAG] generate#2 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 1: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#3 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#3 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 2: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#4 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#4 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 3: nvidia≈14395MB, pt_alloc=110MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [GEN-DIAG] generate#5 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#5 after model.generate: nvidia≈14395MB, non-pt=12921MB                                                                         
  [BM-DIAG] after warmup sample 4: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                    
  [BM-DIAG] after warmup (all): nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                       
  Warmup completed                                                                                                                                   
  Inference:   0%|                                                                                                                   | 0/2 [00:00<?, 
   ?it/s][BM-DIAG] inf[0] before build_prompt: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                        
  [BM-DIAG] inf[0] before processor: nvidia≈14395MB, pt_alloc=111MB, pt_resv=1474MB, non-pt=12921MB                                                  
  [BM-DIAG] inf[0] before .to(cuda): nvidia≈14395MB, pt_alloc=97MB, pt_resv=1474MB, non-pt=12921MB                                                   
    input[input_ids]: shape=[1, 3808], dtype=torch.int64, dev=cpu                                                                                    
    input[attention_mask]: shape=[1, 3808], dtype=torch.int64, dev=cpu                                                                               
    input[pixel_values]: shape=[14840, 1536], dtype=torch.float32, dev=cpu                                                                           
    input[image_grid_thw]: shape=[1, 3], dtype=torch.int64, dev=cpu                                                                                  
  [BM-DIAG] inf[0] after .to(cuda): nvidia≈14395MB, pt_alloc=184MB, pt_resv=1474MB, non-pt=12921MB                                                   
  [BM-DIAG] inf[0] before moe.generate: nvidia≈14395MB, pt_alloc=184MB, pt_resv=1474MB, non-pt=12921MB                                               
  [GEN-DIAG] generate#6 before model.generate: nvidia≈14395MB, non-pt=12921MB                                                                        
  [GEN-DIAG] generate#6 after model.generate: nvidia≈54075MB, non-pt=12921MB                                                                         
  [BM-DIAG] inf[0] after moe.generate: nvidia≈54075MB, pt_alloc=184MB, pt_resv=41154MB, non-pt=12921MB                                               

● 碎片化理论被推翻了。expandable_segments 没有效果，pt_reserved 依然是 41GB。这说明 PyTorch 在 forward pass 期间确实分配了 41GB 的内存。

  问题一定在 MoE-Infinity 的 hook 逻辑里。让我查一下 copy_args_to_device 和 copy_kwargs_to_device — 这些 hook 在每个模块的 forward                       
  前后都会触发，如果它们在复制张量，就会导致大量额外内存。
                                                                                                                                                         
● Read 1 file (ctrl+o to expand)                                                                                                                       
                                                                                                                                                         
● copy_args_to_device 对已在目标设备上的张量调用 .to() 是 no-op，不会复制。所以 hook 不是问题。                                                          
   
  碎片化也不是。那 41GB 就是 forward pass 的真实峰值。我需要定位到底是 vision encoder 还是 language model 消耗的。让我加一个精确的分段追踪。             
                                                                                                                                                       
● Read 1 file (ctrl+o to expand)                                                                                                                         
                                                                                                                                                         
● Update(moe_infinity/entrypoints/big_modeling.py)                                                                                                       
  ⎿  Added 55 lines
      212          _diag = self._gen_call_count <= 6                                                                                                     
      213          if _diag:                                                                                                                           
      214              _diag_gen(f"generate#{self._gen_call_count} before model.generate")                                                               
      215 +                                                                                                                                              
      216 +        # one-shot hooks to trace memory per component                                                                                 
      217 +        _handles = []                                                                                                                  
      218 +        if self._gen_call_count == 6:                                                                                                  
      219 +            _call_idx = [0]                                                                                                            
      220 +            def _mem_pre(name):                                                                                                        
      221 +                def hook(mod, args, kwargs=None):                                                                                      
      222 +                    _call_idx[0] += 1                                                                                                  
      223 +                    if _call_idx[0] <= 3:                                                                                              
      224 +                        torch.cuda.synchronize()                                                                                       
      225 +                        peak = torch.cuda.max_memory_allocated(0)/(1024**2)                                                            
      226 +                        cur = torch.cuda.memory_allocated(0)/(1024**2)                                                                 
      227 +                        resv = torch.cuda.memory_reserved(0)/(1024**2)                                                                 
      228 +                        print(                                                                                                         
      229 +                            f"[FWD-DIAG] >> {name} call#{_call_idx[0]}: "                                                              
      230 +                            f"alloc={cur:.0f}MB peak={peak:.0f}MB "                                                                    
      231 +                            f"resv={resv:.0f}MB",                                                                                      
      232 +                            flush=True,                                                                                                
      233 +                        )                                                                                                              
      234 +                return hook                                                                                                            
      235 +            def _mem_post(name):                                                                                                       
      236 +                def hook(mod, inp, out):                                                                                               
      237 +                    if _call_idx[0] <= 3:                                                                                              
      238 +                        torch.cuda.synchronize()                                                                                       
      239 +                        peak = torch.cuda.max_memory_allocated(0)/(1024**2)                                                            
      240 +                        cur = torch.cuda.memory_allocated(0)/(1024**2)                                                                 
      241 +                        resv = torch.cuda.memory_reserved(0)/(1024**2)                                                                 
      242 +                        print(                                                                                                         
      243 +                            f"[FWD-DIAG] << {name} call#{_call_idx[0]}: "                                                              
      244 +                            f"alloc={cur:.0f}MB peak={peak:.0f}MB "                                                                    
      245 +                            f"resv={resv:.0f}MB",                                                                                      
      246 +                            flush=True,                                                                                                
      247 +                        )                                                                                                              
      248 +                return hook                                                                                                            
      249 +            torch.cuda.reset_peak_memory_stats(0)                                                                                      
      250 +            for tag, attr in [                                                                                                         
      251 +                ("visual", "visual"),                                                                                                  
      252 +                ("lang_model", "language_model"),                                                                                      
      253 +                ("lm_head", "language_model.lm_head"),                                                                                 
      254 +            ]:                                                                                                                         
      255 +                mod = self.model                                                                                                       
      256 +                for a in attr.split("."):                                                                                              
      257 +                    mod = getattr(mod, a, None)                                                                                        
      258 +                    if mod is None:                                                                                                    
      259 +                        break                                                                                                          
      260 +                if mod is not None:                                                                                                    
      261 +                    _handles.append(mod.register_forward_pre_hook(                                                                     
      262 +                        _mem_pre(tag), with_kwargs=True))                                                                              
      263 +                    _handles.append(mod.register_forward_hook(                                                                         
      264 +                        _mem_post(tag)))                                                                                               
      265 +                                                                                                                                       
      266          with torch.no_grad():
      267              output = self.model.generate(input_ids, **kwargs)
      268 +                                                                                                                                       
      269 +        for h in _handles:                                                                                                             
      270 +            h.remove()                                                                                                                 
      271 +                                                                                                                                       
      272          if _diag:
      273              _diag_gen(f"generate#{self._gen_call_count} after model.generate")
      274

● 同时去掉 expandable_segments 验证行（已证明无效）。

● Update(examples/benchmark_qwen3_vl_moe.py)                                                                                                             
  ⎿  Removed 3 lines
      4  import sys                                                                                                                                      
      5  import time                                                                                                                                   
      6                                                                                                                                                  
      7 -# verify: PyTorch expandable segments to bypass fragmentation                                                                                 
      8 -os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"                                                                       
      9 -                                                                                                                                         
      7  import torch
      8  from tqdm import tqdm
      9  from transformers import AutoProcessor
  ⎿  API Error: 401 {"code":401,"msg":{"error":{"message":"Balance exhausted. 余额不足，请联系管理员充值。Do not retry. [claude_code/mixedcc] |       
     balance_exceeded (200.0662/200)","type":"invalid_api_key"}}} · Please run /login