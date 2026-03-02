#!/bin/bash
# MoE-Infinity Comparison Benchmark
# 使用与 qwen3_vl_moe_test 相同的配置进行对比测试

set -e

# 配置参数（与参考脚本保持一致）
MODEL_PATH="/data1/fanya/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
DATA_DIR="/data1/fanya/datasets/realworldqa/realworldQA.tsv"
OFFLOAD_PATH="$HOME/moe-infinity-qwen3vl"
OUTPUT_DIR="./results/moe_infinity_comparison"
MAX_NEW_TOKENS=50
NUM_SAMPLES=20
DEVICE_MEMORY_RATIO=0.09

# 设置 GPU
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "MoE-Infinity Comparison Benchmark"
echo "=========================================="
echo "Model:              $MODEL_PATH"
echo "Dataset:            $DATA_DIR"
echo "Offload Path:       $OFFLOAD_PATH"
echo "Output Dir:         $OUTPUT_DIR"
echo "Max New Tokens:     $MAX_NEW_TOKENS"
echo "Num Samples:        $NUM_SAMPLES"
echo "Device Memory Ratio: $DEVICE_MEMORY_RATIO"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# 运行 benchmark
python examples/benchmark_qwen3_vl_moe.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --offload-path "$OFFLOAD_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens $MAX_NEW_TOKENS \
    --num-samples $NUM_SAMPLES \
    --device-memory-ratio $DEVICE_MEMORY_RATIO

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
