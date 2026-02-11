# MoE-Infinity 开发对话记录

## 日期: 2026-02-10

## 已完成的工作

### 1. Dense 参数绕过 OffloadEngine 常驻 GPU
修改 `moe_infinity/runtime/model_offload.py`，将 dense 参数（attention、norm、embedding、lm_head、shared_expert、buffers）从 C++ OffloadEngine 中分离，直接加载到 GPU 常驻。

**核心改动：**
- `_is_expert_param()`: 判断参数是否为 expert（排除 shared_expert）
- `_offload_state_dict()`: expert 参数进 C++ 引擎，dense 参数保存为 `dense_shard_*.pt`
- `_load_dense_params()`: 从 `dense_shard_*.pt` 加载 dense 参数到 GPU，加入 `offload_exemption`
- `_pre_forward_module_hook` / `_post_forward_module_hook`: 检查 `offload_exemption` 跳过 dense 参数
- NLLB `lm_head.weight` 加了 model_type 守卫

**C++ 端修复：**
- `core/model/model_topology.cpp`: 空 `dense_nodes` 守卫 + `std::max(1, ...)` 防除零

### 2. Benchmark 脚本
创建 `examples/benchmark_qwen3_vl_moe.py`，对标 `qwen3_vl_moe_test/run_inference_hybrid.py`。
- StopWatch 类测量 TTFT/TPOT，首 token 后调用 `clear_expert_cache_counts()`
- 支持 MMStar/MMBench/RealWorldQA 数据集
- 已修复图像处理参数对齐（image_patch_size=16, max_pixels, min_pixels）

### 3. 性能对比结果（MoE-Infinity vs Keep8 Hybrid, 19 个 MMStar 样本）
注意：MoE-Infinity 的结果是在图像参数修复前跑的，prompt_length 偏小，对比不完全公平。
```
Avg TTFT: MoE-Infinity=2.7909s, Keep8=2.3236s, Speedup=1.2022x (Keep8 更快)
Avg TPOT: MoE-Infinity=0.2680s, Keep8=0.2078s, Speedup=1.2901x (Keep8 更快)
```

## 待解决问题

### 1. device_memory_ratio 爆显存问题（高优先级）
**根因**：`SetMemoryRatio` 用 `GetTotalDeviceMemory`（GPU 总显存）* ratio 作为 cache budget。
Dense 参数绕过 C++ 引擎后，topology 中 `dense_cache_size=0`，C++ 引擎不知道 dense 参数已占用 GPU 显存，
导致 `sparse_cache_size = total_memory * ratio - 0`，尝试分配过多显存 → OOM。

**修复方向**：`_load_dense_params` 完成后，计算 dense 参数总大小，通知 C++ 引擎减少 cache budget。
需要在 C++ 端暴露接口扣减 memory capacity。

**临时方案**：用户手动降低 `device_memory_ratio`（当前设为 0.0675）。

### 2. 为 Qwen3-VL-MoE 添加预取支持
当前预取代码只在 Grok 和 Arctic 模型中有活跃调用。
需要在 `moe_infinity/models/qwen3_vl_moe.py` 的 `forward()` 中添加：
- `expert_predictor.predict()` 调用
- `expert_prefetcher.prefetch_experts()` 调用
参考 `moe_infinity/models/grok.py:61-68`。

### 3. 重新跑 benchmark（图像参数修复后）
修复了 `process_vision_info` 和 `processor()` 的参数后，需要重新跑 benchmark 获取正确的对比数据。

## 预取机制概要
- 配置 `prefetch: True` + `trace_capacity: 1000`
- ExpertTracer 记录激活 → ExpertPredictor 余弦相似度匹配历史 → ExpertPrefetcher 异步预取
- C++ 端 `EnqueuePrefetch()` 创建低优先级异步任务，后台线程执行 Disk→CPU→GPU
- 按需加载 `begin()` 优先级高于预取

## 关键文件
- `moe_infinity/runtime/model_offload.py` — 核心修改文件
- `core/model/model_topology.cpp` — C++ topology 修复
- `examples/benchmark_qwen3_vl_moe.py` — 性能测试脚本
- `moe_infinity/models/qwen3_vl_moe.py` — Qwen3 VL MoE 模型层
- `moe_infinity/models/grok.py` — 预取实现参考
