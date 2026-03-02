#!/usr/bin/env python3
"""
性能对比分析脚本
比较 MoE-Infinity 和 Offload Baseline 的性能指标
"""

import argparse
import json
import os
from typing import Dict, List


def load_metrics(metrics_file: str) -> List[Dict]:
    """加载 metrics.json 文件"""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_stats(metrics: List[Dict]) -> Dict:
    """计算统计指标"""
    if not metrics:
        return {}

    ttfts = [m['ttft'] for m in metrics]
    tpots = [m['tpot'] for m in metrics if m['tpot'] > 0]
    prompt_lengths = [m['prompt_length'] for m in metrics]
    num_tokens = [m['num_new_tokens'] for m in metrics]

    stats = {
        'num_samples': len(metrics),
        'avg_ttft': sum(ttfts) / len(ttfts) if ttfts else 0,
        'min_ttft': min(ttfts) if ttfts else 0,
        'max_ttft': max(ttfts) if ttfts else 0,
        'avg_tpot': sum(tpots) / len(tpots) if tpots else 0,
        'min_tpot': min(tpots) if tpots else 0,
        'max_tpot': max(tpots) if tpots else 0,
        'avg_throughput': 1.0 / (sum(tpots) / len(tpots)) if tpots else 0,
        'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        'avg_num_tokens': sum(num_tokens) / len(num_tokens) if num_tokens else 0,
    }

    return stats


def print_comparison(baseline_stats: Dict, moeinf_stats: Dict):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("性能对比分析")
    print("=" * 80)
    print()

    print(f"{'指标':<30} {'Baseline':<20} {'MoE-Infinity':<20} {'改进':<15}")
    print("-" * 80)

    # 样本数
    print(f"{'样本数':<30} {baseline_stats['num_samples']:<20} {moeinf_stats['num_samples']:<20} {'-':<15}")

    # TTFT
    baseline_ttft = baseline_stats['avg_ttft']
    moeinf_ttft = moeinf_stats['avg_ttft']
    ttft_improvement = ((baseline_ttft - moeinf_ttft) / baseline_ttft * 100) if baseline_ttft > 0 else 0
    print(f"{'平均 TTFT (s)':<30} {baseline_ttft:<20.4f} {moeinf_ttft:<20.4f} {ttft_improvement:>+.2f}%")

    # TPOT
    baseline_tpot = baseline_stats['avg_tpot']
    moeinf_tpot = moeinf_stats['avg_tpot']
    tpot_improvement = ((baseline_tpot - moeinf_tpot) / baseline_tpot * 100) if baseline_tpot > 0 else 0
    print(f"{'平均 TPOT (s)':<30} {baseline_tpot:<20.4f} {moeinf_tpot:<20.4f} {tpot_improvement:>+.2f}%")

    # Throughput
    baseline_throughput = baseline_stats['avg_throughput']
    moeinf_throughput = moeinf_stats['avg_throughput']
    throughput_improvement = ((moeinf_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0
    print(f"{'平均吞吐量 (tok/s)':<30} {baseline_throughput:<20.2f} {moeinf_throughput:<20.2f} {throughput_improvement:>+.2f}%")

    # Prompt Length
    print(f"{'平均 Prompt 长度':<30} {baseline_stats['avg_prompt_length']:<20.1f} {moeinf_stats['avg_prompt_length']:<20.1f} {'-':<15}")

    # Num Tokens
    print(f"{'平均生成 Token 数':<30} {baseline_stats['avg_num_tokens']:<20.1f} {moeinf_stats['avg_num_tokens']:<20.1f} {'-':<15}")

    print("=" * 80)
    print()

    # 详细统计
    print("详细统计:")
    print("-" * 80)
    print(f"{'指标':<30} {'Baseline':<35} {'MoE-Infinity':<35}")
    print("-" * 80)
    print(f"{'TTFT 范围 (s)':<30} [{baseline_stats['min_ttft']:.4f}, {baseline_stats['max_ttft']:.4f}]"
          f"{'':<10} [{moeinf_stats['min_ttft']:.4f}, {moeinf_stats['max_ttft']:.4f}]")
    print(f"{'TPOT 范围 (s)':<30} [{baseline_stats['min_tpot']:.4f}, {baseline_stats['max_tpot']:.4f}]"
          f"{'':<10} [{moeinf_stats['min_tpot']:.4f}, {moeinf_stats['max_tpot']:.4f}]")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description='比较 MoE-Infinity 和 Baseline 的性能')
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Baseline metrics.json 文件路径'
    )
    parser.add_argument(
        '--moeinf',
        type=str,
        required=True,
        help='MoE-Infinity metrics.json 文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出对比结果到 JSON 文件'
    )

    args = parser.parse_args()

    # 加载数据
    print(f"加载 Baseline 数据: {args.baseline}")
    baseline_metrics = load_metrics(args.baseline)

    print(f"加载 MoE-Infinity 数据: {args.moeinf}")
    moeinf_metrics = load_metrics(args.moeinf)

    # 计算统计
    baseline_stats = calculate_stats(baseline_metrics)
    moeinf_stats = calculate_stats(moeinf_metrics)

    # 打印对比
    print_comparison(baseline_stats, moeinf_stats)

    # 保存结果
    if args.output:
        comparison = {
            'baseline': baseline_stats,
            'moe_infinity': moeinf_stats,
            'improvements': {
                'ttft_percent': ((baseline_stats['avg_ttft'] - moeinf_stats['avg_ttft']) / baseline_stats['avg_ttft'] * 100) if baseline_stats['avg_ttft'] > 0 else 0,
                'tpot_percent': ((baseline_stats['avg_tpot'] - moeinf_stats['avg_tpot']) / baseline_stats['avg_tpot'] * 100) if baseline_stats['avg_tpot'] > 0 else 0,
                'throughput_percent': ((moeinf_stats['avg_throughput'] - baseline_stats['avg_throughput']) / baseline_stats['avg_throughput'] * 100) if baseline_stats['avg_throughput'] > 0 else 0,
            }
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"对比结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
