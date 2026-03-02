# MoE-Infinity 性能对比测试指南

本指南说明如何运行 MoE-Infinity 与 Offload Baseline 的性能对比测试。

## 测试配置

为了确保公平对比，两个系统使用完全相同的配置：

- **模型**: Qwen3-VL-30B-A3B-Instruct
- **数据集**: RealWorldQA (100 个样本，random_state=42)
- **生成参数**: max_new_tokens=2048, do_sample=False
- **GPU**: CUDA_VISIBLE_DEVICES=1
- **Device Memory Ratio**: 0.09 (约 5% expert cache)

## 运行步骤

### 1. 准备环境

确保已经安装了 MoE-Infinity：

```bash
cd /home/fanya/moe-infinity
pip install -e . --force-reinstall --no-deps
```

### 2. 运行 MoE-Infinity Benchmark

```bash
cd /home/fanya/moe-infinity
./run_comparison_benchmark.sh
```

这将运行 benchmark 并将结果保存到 `./results/moe_infinity_benchmark/`。

### 3. 运行 Offload Baseline Benchmark（如果还没有运行）

```bash
cd /home/fanya
python run_inference_offload.py \
    --model-path /data1/fanya/models/Qwen/Qwen3-VL-30B-A3B-Instruct \
    --data-dir /data1/fanya/datasets/realworldqa/data \
    --output-dir ./perf_results/realworldqa/cache5%/perf_offload/keep8 \
    --max-new-tokens 2048
```

### 4. 对比性能

使用对比脚本分析两个系统的性能：

```bash
cd /home/fanya/moe-infinity
python compare_performance.py \
    --baseline /home/fanya/perf_results/realworldqa/cache5%/perf_offload/keep8/metrics.json \
    --moeinf ./results/moe_infinity_benchmark/metrics.json \
    --output ./results/performance_comparison.json
```

## 输出文件

### MoE-Infinity 输出

- `results/moe_infinity_comparison/output.json` - 推理结果
- `results/moe_infinity_comparison/metrics.json` - 性能指标
- `results/moe_infinity_comparison/run_config.json` - 运行配置
- `results/moe_infinity_comparison/accuracy.json` - 准确率评估

### 对比结果

- `results/performance_comparison.json` - 性能对比数据

## 性能指标说明

- **TTFT (Time To First Token)**: 从输入到生成第一个 token 的时间
- **TPOT (Time Per Output Token)**: 每个输出 token 的平均生成时间
- **Throughput**: 吞吐量 (tokens/second)
- **Prompt Length**: 输入 prompt 的长度
- **Num New Tokens**: 生成的 token 数量

## 注意事项

1. **数据集一致性**: 确保两个系统使用相同的数据集和相同的 random_state (42)
2. **GPU 一致性**: 使用相同的 GPU (CUDA_VISIBLE_DEVICES=1)
3. **配置一致性**: 确保 device_memory_ratio 和其他配置参数相同
4. **Warmup**: 两个系统都进行了 5 次 warmup
5. **样本数量**: 都使用 100 个样本进行测试
6. **生成长度**: max_new_tokens=2048，用于测试长文本生成性能

## 预期结果

对比脚本将输出：

- 平均 TTFT 对比
- 平均 TPOT 对比
- 平均吞吐量对比
- 性能改进百分比
- 详细统计（最小值、最大值等）

## 故障排除

### 如果 MoE-Infinity 卡在 warmup

1. 检查是否已经回退到稳定版本：
   ```bash
   git log --oneline -5
   ```
   应该看到 "cleanup: remove all diagnostic/debug code from SDPA memory fix"

2. 重新编译：
   ```bash
   pip install -e . --force-reinstall --no-deps
   ```

3. 检查预取功能是否禁用：
   ```bash
   grep -n "prefetch_enabled" moe_infinity/runtime/model_offload.py
   ```
   应该没有输出（预取功能默认禁用）

### 如果内存不足

1. 减少 device_memory_ratio（例如从 0.09 降到 0.05）
2. 减少样本数量（修改 run_comparison_benchmark.sh 中的 NUM_SAMPLES）
3. 减少 max_new_tokens（例如从 2048 降到 1024）

## 参考

- Baseline 实现: `/home/fanya/run_inference_offload.py`
- MoE-Infinity 实现: `/home/fanya/moe-infinity/examples/benchmark_qwen3_vl_moe.py`
