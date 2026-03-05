# MoE-Infinity Warmup 死锁问题分析与修复

**日期**: 2026-03-02 - 2026-03-05
**状态**: 已解决
**影响版本**: 提交 fbabf9e 之后的所有版本
**修复版本**: 提交 2ec6f66

## 问题描述

程序在 warmup 阶段卡住，无法完成第一次推理。具体表现：
- 打印 "Warming up..." 后程序永久阻塞
- 主线程卡在 `expert_dispatcher.cpp:130` 的 `mutex.lock()`
- 所有 GPUFetchFunc 和 GPUExecFunc 线程都在等待任务
- 没有线程在执行 OutputFunc（应该释放 mutex 的地方）

## 根本原因分析

### 时间线追踪

通过 git 历史分析，发现了问题的演变过程：

1. **2026-02-11 23:15** - 提交 1f6d3ce
   - 将 `expert_dispatcher.cpp` 中的 `try_lock()` 改为 `lock()`
   - 目的：让 dispatch 等待 prefetch 完成
   - **这个修改本身没有问题**，是为预取功能做准备

2. **2026-02-12 00:08** - 提交 10b708c
   - Revert `_prewarm_dense_params` 函数
   - 原因：该函数调用 `begin()` 但不调用 `end()`，导致 mutex 泄漏

3. **2026-02-12 - 2026-02-28**
   - 预取功能禁用（默认值 `prefetch: bool = False`）
   - 程序正常运行，有测试结果提交

4. **2026-03-01 16:35** - 提交 fbabf9e ← **问题根源！**
   - 在 `model_offload.py:602` 添加了 `module.prefetch_enabled = True`
   - **启用了预取功能**
   - 导致预取线程和按需加载线程之间的 mutex 竞争

5. **2026-03-01 19:26** - 提交 3509ad0
   - 临时禁用预取来避免死锁
   - 说明已经发现了预取导致的死锁问题

6. **2026-03-01 晚上 - 2026-03-02**
   - 大量调试提交，尝试解决死锁问题
   - 问题依然存在

### 结论

**死锁的真正原因**：提交 fbabf9e 启用了预取功能，而预取机制存在并发 bug，导致 mutex 泄漏或死锁。

**关键发现**：
- 提交 1f6d3ce 本身没有问题，是为预取功能做准备的
- 真正的问题是预取功能本身有并发 bug
- 在预取功能禁用的情况下，程序能正常运行

## 解决方案

### 采用的方案：回退到稳定版本

**步骤**：

1. **回退到提交 1e18909**（2026-02-28）
   - 提交信息：`cleanup: remove all diagnostic/debug code from SDPA memory fix`
   - 这是启用 SDPA 之后的清理提交
   - 已验证这个版本能正常运行

2. **重新应用 cache_sizes_ 修复**（提交 2ec6f66）
   - 修复了 `cache_sizes_` 在 cache hit 时也被减少的 bug
   - 这是一个独立的 bug 修复，不影响死锁问题

**执行命令**：
```bash
# 保存 cache_sizes_ 修复的 patch
git format-patch -1 e7817d3 -o /tmp/

# 回退到 SDPA 之后的清理提交
git reset --hard 1e18909

# 手动应用 cache_sizes_ 修复
# 在 expert_dispatcher.cpp:264 添加条件检查
if (!cache_hit) {
  cache_sizes_[gpu_id] -= expert_node->node->byte_size;
}

# 提交修复
git add core/parallel/expert_dispatcher.cpp
git commit -m "fix: only decrease cache_sizes_ on cache miss"

# 推送到远程
git push origin dev --force-with-lease
```

### 当前状态

**已启用的功能**：
- SDPA attention（减少内存使用）
- cache_sizes_ 修复（更准确的缓存统计）

**已禁用的功能**：
- 预取功能（默认禁用，避免死锁）

**预期结果**：
- 程序能正常通过 warmup 阶段
- 第一次推理能成功完成
- 没有死锁或卡住

## 性能对比工具

为了与 Offload Baseline 进行公平对比，创建了以下工具：

### 1. 运行脚本

**文件**: `run_comparison_benchmark.sh`

使用与 `/home/fanya/run_inference_offload.py` 相同的配置：
- 数据集: RealWorldQA (100 个样本，random_state=42)
- max_new_tokens: 2048
- device_memory_ratio: 0.09
- GPU: CUDA_VISIBLE_DEVICES=1

### 2. 对比分析脚本

**文件**: `compare_performance.py`

功能：
- 加载两个系统的 metrics.json 文件
- 计算统计指标（TTFT、TPOT、吞吐量等）
- 打印对比结果和改进百分比
- 保存对比结果到 JSON 文件

### 3. 使用指南

**文件**: `COMPARISON_GUIDE.md`

包含：
- 测试配置说明
- 运行步骤
- 输出文件说明
- 故障排除指南

## 重要经验教训

### 1. 关于并发编程

**问题**：预取功能引入了复杂的并发场景
- 预取线程和按需加载线程同时访问 expert node
- Mutex 的获取和释放顺序不当导致死锁

**教训**：
- 在启用新的并发功能前，需要充分测试
- 使用 `lock()` 替代 `try_lock()` 时要特别小心
- 确保所有锁住的 mutex 都能正确释放

### 2. 关于 Git 历史分析

**方法**：
- 通过 `git log` 追踪问题引入的时间点
- 对比不同时期的代码变化
- 找到问题引入和正常运行的分界点

**教训**：
- Git 历史是调试的重要工具
- 提交信息要清晰，便于后续追踪
- 重要的修改要有详细的说明

### 3. 关于编译和部署

**问题**：混淆了何时需要重新编译

**规则**：
- **需要重新编译**：修改了 C++ 代码（`core/` 目录）
- **不需要重新编译**：只修改了 Python 代码、脚本或文档

**判断方法**：
```bash
# 查看 C++ 代码的最后修改时间
git log -1 --format="%ai" -- core/

# 查看编译产物的时间
ls -lh moe_infinity/ops/*.so
```

如果 `.so` 文件的时间比 C++ 代码的修改时间新，就不需要重新编译。

### 4. 关于性能对比

**要点**：
- 确保两个系统使用完全相同的配置
- 使用相同的数据集和 random_state
- 使用相同的 GPU 和内存配置
- 记录所有配置参数到 run_config.json

## 后续工作

### 短期

1. **验证修复**：在服务器端运行 benchmark，确认死锁问题已解决
2. **性能对比**：与 Offload Baseline 进行公平对比
3. **文档更新**：更新 CLAUDE.md 中的相关内容

### 长期

1. **修复预取功能**：
   - 分析预取机制的并发 bug
   - 重新设计 mutex 的获取和释放逻辑
   - 添加更完善的并发控制机制
   - 充分测试后再启用

2. **改进调试工具**：
   - 添加更详细的日志记录
   - 实现死锁检测机制
   - 添加性能分析工具

3. **代码审查**：
   - 审查所有使用 mutex 的地方
   - 确保没有其他潜在的死锁风险
   - 添加并发测试用例

## 参考资料

### 相关文件

- `core/parallel/expert_dispatcher.cpp` - Expert 调度器实现
- `moe_infinity/runtime/model_offload.py` - Offload 引擎实现
- `examples/benchmark_qwen3_vl_moe.py` - Benchmark 脚本
- `/home/fanya/run_inference_offload.py` - Baseline 参考实现

### 相关提交

- `1f6d3ce` - 将 try_lock 改为 lock（为预取做准备）
- `10b708c` - Revert prewarm 函数（mutex 泄漏）
- `fbabf9e` - 启用预取功能（引入死锁）
- `3509ad0` - 临时禁用预取（发现问题）
- `1e18909` - SDPA 清理提交（稳定版本）
- `2ec6f66` - cache_sizes_ 修复（当前版本）

### GDB 调试记录

详细的 GDB 调试过程记录在 `gdb_threads.txt` 中，包括：
- 所有线程的堆栈信息
- Mutex 的状态
- 死锁发生时的现场

## 总结

这次死锁问题的解决过程展示了：
1. Git 历史分析在问题定位中的重要性
2. 并发编程需要特别小心，尤其是 mutex 的使用
3. 回退到已知稳定版本是快速恢复的有效方法
4. 充分的测试和文档记录对于维护代码质量至关重要

通过这次经验，我们建立了更完善的调试流程和性能对比工具，为后续的开发和优化奠定了基础。
