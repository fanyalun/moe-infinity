import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.generation.streamers import BaseStreamer

PROJECT_DIR = "/home/fanya/qwen3_vl_moe_test"
MOE_INFINITY_DIR = os.path.join(PROJECT_DIR, "moe-infinity")
if MOE_INFINITY_DIR not in sys.path:
    sys.path.insert(0, MOE_INFINITY_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from qwen_vl_utils import process_vision_info
from utils.dataset import (
    load_ai2d_dataset,
    load_hallusionbench_dataset,
    load_mme_dataset,
    load_mmbench_dataset,
    load_pope_dataset,
    load_realworldqa_dataset,
    load_scienceqa_dataset,
)
from utils.eval import eval_mme, eval_mmbench, eval_realworldqa
from utils.prompt import (
    build_ai2d_prompt,
    build_hallusionbench_prompt,
    build_mme_prompt,
    build_mmbench_prompt,
    build_pope_prompt,
    build_realworldqa_prompt,
    build_scienceqa_prompt,
)

DATASET_LOADERS = {
    "mmbench": load_mmbench_dataset,
    "hallusionbench": load_hallusionbench_dataset,
    "ai2d": load_ai2d_dataset,
    "mme": load_mme_dataset,
    "realworldqa": load_realworldqa_dataset,
    "scienceqa": load_scienceqa_dataset,
    "pope": load_pope_dataset,
}

PROMPT_BUILDERS = {
    "mmbench": build_mmbench_prompt,
    "hallusionbench": build_hallusionbench_prompt,
    "ai2d": build_ai2d_prompt,
    "mme": build_mme_prompt,
    "realworldqa": build_realworldqa_prompt,
    "scienceqa": build_scienceqa_prompt,
    "pope": build_pope_prompt,
}

EVAL_FNS = {
    "mmbench": eval_mmbench,
    "hallusionbench": eval_realworldqa,
    "ai2d": eval_realworldqa,
    "mme": eval_mme,
    "realworldqa": eval_realworldqa,
    "scienceqa": eval_realworldqa,
    "pope": eval_realworldqa,
}

BENCHMARK_NAMES = {
    "mmbench": "MMBench_DEV_EN_V11",
    "hallusionbench": "HallusionBench",
    "ai2d": "AI2D_TEST",
    "mme": "MME",
    "realworldqa": "RealWorldQA",
    "scienceqa": "ScienceQA_TEST",
    "pope": "POPE",
}

SUMMARY_ROOT = "/home/fanya/qwen3_vl_moe_test/new_results/cache5%"
SUMMARY_JSON = os.path.join(SUMMARY_ROOT, "summary_metrics.json")
SUMMARY_MD = os.path.join(SUMMARY_ROOT, "summary_metrics.md")
SUMMARY_MODE_ORDER = ["AdapMoE", "TableMoE", "MoE_Infinity"]


class StopWatch(BaseStreamer):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.start_time = None
        self.first_token_time = None
        self.last_token_time = None
        self.num_new_tokens = 0

    def put(self, value):
        if isinstance(value, torch.Tensor):
            if value.dim() > 1 and value.shape[-1] > 1:
                return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
            dispatcher = getattr(self.engine, "expert_dispatcher", None)
            if dispatcher is not None and hasattr(dispatcher, "clear_expert_cache_counts"):
                dispatcher.clear_expert_cache_counts()

        self.last_token_time = now
        self.num_new_tokens += 1

    def end(self):
        pass

    def get_metrics(self):
        if self.start_time is None or self.first_token_time is None or self.num_new_tokens == 0:
            return 0.0, 0.0, 0

        ttft = self.first_token_time - self.start_time
        if self.num_new_tokens > 1:
            tpot = (self.last_token_time - self.first_token_time) / (self.num_new_tokens - 1)
        else:
            tpot = 0.0
        return ttft, tpot, self.num_new_tokens


def detect_dataset_key(data_dir: str) -> str:
    lower = data_dir.lower()
    aliases = {
        "mmbench": ["mmbench", "mmbench_dev_en_v11"],
        "hallusionbench": ["hallusionbench"],
        "ai2d": ["ai2d_test", "ai2d"],
        "mme": ["mme"],
        "realworldqa": ["realworldqa"],
        "scienceqa": ["scienceqa_test", "scienceqa"],
        "pope": ["pope"],
    }
    for key, vals in aliases.items():
        if any(v in lower for v in vals):
            return key
    raise ValueError(f"Unknown dataset in {data_dir}")


def sample_dataset_for_benchmark(dataset, sample_ratio: float, sample_seed: int):
    total = len(dataset)
    if total == 0:
        return dataset
    sample_size = max(1, int(total * sample_ratio))
    sample_size = min(sample_size, total)
    sampled = dataset.sample(n=sample_size, random_state=sample_seed).reset_index(drop=True)
    print(f"Benchmark sampling: {sample_size}/{total} ({sample_ratio:.1%}), seed={sample_seed}")
    return sampled


def _load_summary(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_summary(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_summary_initialized(model_name: str):
    summary = _load_summary(SUMMARY_JSON)
    summary.setdefault("models", {})
    summary["models"].setdefault(model_name, {})

    summary["models"][model_name].setdefault("MoE_Infinity", {})
    for bench in BENCHMARK_NAMES.values():
        summary["models"][model_name]["MoE_Infinity"].setdefault(
            bench,
            {"avg_ttft": 0.0, "avg_tpot": 0.0, "num_samples": 0},
        )

    summary["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_summary(SUMMARY_JSON, summary)
    render_summary_markdown()


def update_summary(model_name: str, benchmark_name: str, metrics_list: list):
    summary = _load_summary(SUMMARY_JSON)
    summary.setdefault("models", {})
    summary["models"].setdefault(model_name, {})
    summary["models"][model_name].setdefault("MoE_Infinity", {})

    if metrics_list:
        ttfts = [float(m.get("ttft", 0.0)) for m in metrics_list]
        tpots = [float(m.get("tpot", 0.0)) for m in metrics_list]
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tpot = sum(tpots) / len(tpots)
        num_samples = len(metrics_list)
    else:
        avg_ttft = 0.0
        avg_tpot = 0.0
        num_samples = 0

    summary["models"][model_name]["MoE_Infinity"][benchmark_name] = {
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "num_samples": num_samples,
    }
    summary["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_summary(SUMMARY_JSON, summary)
    render_summary_markdown()

    print(
        f"[Summary] MoE_Infinity/{benchmark_name}: "
        f"avg_ttft={avg_ttft:.6f}, avg_tpot={avg_tpot:.6f}, n={num_samples}"
    )


def render_summary_markdown():
    summary = _load_summary(SUMMARY_JSON)
    if not summary:
        return

    lines = ["# TTFT/TPOT Summary", ""]
    updated_at = summary.get("updated_at", "")
    if updated_at:
        lines.append(f"Updated at: {updated_at}")
        lines.append("")

    models = summary.get("models", {})
    for model_name in sorted(models.keys()):
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("| Mode | Benchmark | Avg TTFT | Avg TPOT | Samples |")
        lines.append("|---|---|---:|---:|---:|")

        mode_data = models[model_name]
        ordered_modes = [m for m in SUMMARY_MODE_ORDER if m in mode_data]
        ordered_modes += [m for m in mode_data.keys() if m not in ordered_modes]

        for mode in ordered_modes:
            bench_map = mode_data.get(mode, {})
            for benchmark in sorted(bench_map.keys()):
                item = bench_map[benchmark]
                ttft = float(item.get("avg_ttft", 0.0))
                tpot = float(item.get("avg_tpot", 0.0))
                ns = int(item.get("num_samples", 0))
                lines.append(f"| {mode} | {benchmark} | {ttft:.6f} | {tpot:.6f} | {ns} |")
        lines.append("")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_benchmark(args):
    if not (0 < args.sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {args.sample_ratio}")

    dataset_key = detect_dataset_key(args.data_dir)
    benchmark_name = BENCHMARK_NAMES[dataset_key]
    model_name = os.path.basename(args.model_path)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.result_root, f"{model_name}_{benchmark_name}")

    os.makedirs(output_dir, exist_ok=True)
    ensure_summary_initialized(model_name)

    print("\n" + "=" * 80)
    print("MoE-Infinity Benchmark: Qwen3-VL-MoE")
    print("=" * 80)
    print(f"Dataset: {benchmark_name}")
    print(f"Output:  {output_dir}\n")

    load_dataset = DATASET_LOADERS[dataset_key]
    build_prompt = PROMPT_BUILDERS[dataset_key]
    eval_fn = EVAL_FNS[dataset_key]

    print(f"Loading {dataset_key} from {args.data_dir}...")
    dataset = load_dataset(args.data_dir)
    print(f"Loaded {len(dataset)} samples")

    print(f"Loading model: {args.model_path}")
    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(
            "Failed to import moe_infinity runtime dependencies. "
            "Please ensure moe-infinity requirements are installed in current env."
        ) from exc
    moe_config = {
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
    }
    moe = MoE(args.model_path, moe_config)
    model = moe.model
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    print("Model loaded")

    parent_dir = os.path.dirname(args.data_dir)
    img_root = os.path.join(parent_dir, "images")
    os.makedirs(img_root, exist_ok=True)

    print("Warming up...")
    for i in range(min(5, len(dataset))):
        try:
            row = dataset.iloc[i]
            msgs = build_prompt(row, img_root)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                msgs,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos = list(videos)
                video_metadatas = list(video_metadatas)
            warmup_inputs = processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=True,
                max_pixels=5120 * 28 * 28,
                min_pixels=768 * 28 * 28,
                padding=True,
                return_tensors="pt",
                **(video_kwargs or {}),
            ).to(args.device)
            _ = moe.generate(**warmup_inputs, max_new_tokens=10)
        except Exception as exc:
            print(f"Warmup sample {i} failed: {exc}")
    print("Warmup completed")

    dataset = sample_dataset_for_benchmark(dataset, args.sample_ratio, args.sample_seed)

    results = []
    metrics_list = []
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Inference-MoEInfinity"):
        try:
            messages = build_prompt(row, img_root)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos = list(videos)
                video_metadatas = list(video_metadatas)

            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=True,
                max_pixels=5120 * 28 * 28,
                min_pixels=768 * 28 * 28,
                return_tensors="pt",
                **(video_kwargs or {}),
            ).to(args.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            streamer = StopWatch(moe.engine)
            streamer.start_time = time.perf_counter()

            generated_ids = moe.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft, tpot, n_tok = streamer.get_metrics()

            trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            sample_id = row.get("id", row.get("index", idx))
            metrics_list.append(
                {
                    "id": str(sample_id),
                    "ttft": ttft,
                    "tpot": tpot,
                    "prompt_length": inputs.input_ids.shape[1],
                    "num_new_tokens": n_tok,
                }
            )
            results.append(
                {
                    "id": int(sample_id) if isinstance(sample_id, int) else str(sample_id),
                    "question": row.get("question", ""),
                    "gt_answer": row.get("answer", ""),
                    "prediction": output_text,
                }
            )

        except Exception as exc:
            print(f"Error sample {idx}: {exc}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)

    avg_ttft = 0.0
    avg_tpot = 0.0
    if metrics_list:
        ttfts = [m["ttft"] for m in metrics_list]
        tpots = [m["tpot"] for m in metrics_list]
        tokens = [m["num_new_tokens"] for m in metrics_list]
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tpot = sum(tpots) / len(tpots)

        print(f"Samples:        {len(metrics_list)}")
        print(f"Avg TTFT:       {avg_ttft:.4f}s")
        print(f"Avg TPOT:       {avg_tpot:.4f}s")
        if avg_tpot > 0:
            print(f"Avg Throughput: {1.0 / avg_tpot:.2f} tok/s")
        print(f"Avg New Tokens: {sum(tokens) / len(tokens):.1f}")

    with open(os.path.join(output_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)

    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "resolved": {
            "dataset_key": dataset_key,
            "benchmark_name": benchmark_name,
            "output_dir": output_dir,
            "avg_ttft": avg_ttft,
            "avg_tpot": avg_tpot,
        },
        "moe_config": moe_config,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("\nEvaluating results...")
    eval_fn(results=results, output_dir=output_dir)

    update_summary(model_name, benchmark_name, metrics_list)


def main():
    parser = argparse.ArgumentParser(description="MoE-Infinity Qwen3-VL-MoE Benchmark")
    parser.add_argument("--model-path", type=str, default="/data1/fanya/models/Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--data-dir", type=str, default="/home/fanya/LMUData/RealWorldQA.tsv")
    parser.add_argument("--output-dir", type=str, default="", help="If empty, auto uses result-root/<model>_<benchmark>.")
    parser.add_argument(
        "--result-root",
        type=str,
        default="/home/fanya/qwen3_vl_moe_test/new_results/cache5%/MoE_Infinity",
    )
    parser.add_argument(
        "--offload-path",
        type=str,
        default="/home/fanya/qwen3_vl_moe_test/moe-infinity-qwen3vl",
    )
    parser.add_argument("--device-memory-ratio", type=float, default=0.09)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
