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
