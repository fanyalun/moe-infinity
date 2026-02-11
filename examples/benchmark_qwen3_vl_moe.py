import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.generation.streamers import BaseStreamer

from moe_infinity import MoE

# Add utils from qwen3_vl_moe_test
sys.path.insert(0, "/home/fanya/qwen3_vl_moe_test")
from qwen_vl_utils import process_vision_info
from utils.dataset import (
    load_mmbench_dataset,
    load_mmstar_dataset,
    load_realworldqa_dataset,
)
from utils.eval import eval_mmbench, eval_mmstar, eval_realworldqa
from utils.prompt import (
    build_mmbench_prompt,
    build_mmstar_prompt,
    build_realworldqa_prompt,
)

DATASET_LOADERS = {
    "realworldqa": load_realworldqa_dataset,
    "mmbench": load_mmbench_dataset,
    "mmstar": load_mmstar_dataset,
}
PROMPT_BUILDERS = {
    "realworldqa": build_realworldqa_prompt,
    "mmbench": build_mmbench_prompt,
    "mmstar": build_mmstar_prompt,
}
EVAL_FNS = {
    "realworldqa": eval_realworldqa,
    "mmbench": eval_mmbench,
    "mmstar": eval_mmstar,
}


class StopWatch(BaseStreamer):
    """Performance timer for MoE-Infinity inference.

    Measures TTFT and TPOT. Clears expert cache counts
    after prefill to get clean decode-phase statistics.
    """

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

        t = time.perf_counter()

        if self.first_token_time is None:
            self.first_token_time = t
            self.engine.expert_dispatcher.clear_expert_cache_counts()

        self.last_token_time = t
        self.num_new_tokens += 1

    def end(self):
        pass

    def get_metrics(self):
        if (
            self.start_time is None
            or self.first_token_time is None
            or self.num_new_tokens == 0
        ):
            return 0.0, 0.0, 0

        ttft = self.first_token_time - self.start_time
        if self.num_new_tokens > 1:
            tpot = (
                self.last_token_time - self.first_token_time
            ) / (self.num_new_tokens - 1)
        else:
            tpot = 0.0
        return ttft, tpot, self.num_new_tokens


def detect_dataset_key(data_dir):
    lower = data_dir.lower()
    for key in DATASET_LOADERS:
        if key in lower:
            return key
    raise ValueError(f"Unknown dataset in {data_dir}")


def run_benchmark(args):
    print("\n" + "=" * 80)
    print("MoE-Infinity Benchmark: Qwen3-VL-MoE")
    print("=" * 80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset
    dataset_key = detect_dataset_key(args.data_dir)
    load_dataset = DATASET_LOADERS[dataset_key]
    build_prompt = PROMPT_BUILDERS[dataset_key]
    eval_fn = EVAL_FNS[dataset_key]

    print(f"Loading {dataset_key} from {args.data_dir}...")
    dataset = load_dataset(args.data_dir)
    print(f"Loaded {len(dataset)} samples")

    # 2. Build Model
    print(f"Loading model: {args.model_path}")
    moe_config = {
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
    }
    moe = MoE(args.model_path, moe_config)
    model = moe.model
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = (
            processor.tokenizer.pad_token_id
        )
    print("Model loaded")

    parent_dir = os.path.dirname(args.data_dir)
    img_root = os.path.join(parent_dir, "images")
    os.makedirs(img_root, exist_ok=True)

    # 3. Warmup
    print("Warming up...")
    for i in range(min(5, len(dataset))):
        try:
            row = dataset.iloc[i]
            msgs = build_prompt(row, img_root)
            text = processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = (
                process_vision_info(
                    msgs,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            )
            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos = list(videos)
                video_metadatas = list(video_metadatas)
            inputs = processor(
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
            ).to("cuda:0")
            _ = model.generate(**inputs, max_new_tokens=10)
        except Exception as e:
            print(f"Warmup sample {i} failed: {e}")
    print("Warmup completed")

    # 4. Inference
    results = []
    metrics_list = []
    dataset = dataset.sample(
        n=min(args.num_samples, len(dataset)),
        random_state=42,
    ).reset_index(drop=True)

    for idx, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc="Inference",
    ):
        try:
            messages = build_prompt(row, img_root)
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos, video_kwargs = (
                process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
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
            ).to("cuda:0")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            streamer = StopWatch(moe.engine)
            streamer.start_time = time.perf_counter()

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft, tpot, n_tok = streamer.get_metrics()

            trimmed = [
                o[len(i) :]
                for i, o in zip(
                    inputs.input_ids, generated_ids
                )
            ]
            output_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            sid = row.get("id", row.get("index", idx))
            metrics_list.append({
                "id": str(sid),
                "ttft": ttft,
                "tpot": tpot,
                "prompt_length": inputs.input_ids.shape[1],
                "num_new_tokens": n_tok,
            })
            results.append({
                "id": (
                    int(sid)
                    if isinstance(sid, int)
                    else str(sid)
                ),
                "question": row.get("question", ""),
                "gt_answer": row.get("answer", ""),
                "prediction": output_text,
            })

        except Exception as e:
            print(f"Error sample {idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 5. Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)

    if metrics_list:
        ttfts = [m["ttft"] for m in metrics_list]
        tpots = [
            m["tpot"] for m in metrics_list if m["tpot"] > 0
        ]
        tokens = [m["num_new_tokens"] for m in metrics_list]

        print(f"Samples:        {len(metrics_list)}")
        print(
            f"Avg TTFT:       {sum(ttfts)/len(ttfts):.4f}s"
        )
        if tpots:
            avg_tpot = sum(tpots) / len(tpots)
            print(f"Avg TPOT:       {avg_tpot:.4f}s")
            print(f"Avg Throughput: {1.0/avg_tpot:.2f} tok/s")
        print(
            f"Avg New Tokens: {sum(tokens)/len(tokens):.1f}"
        )

    # 6. Save
    with open(
        os.path.join(args.output_dir, "output.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(
        os.path.join(args.output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_list, f, indent=2, ensure_ascii=False
        )

    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "moe_config": moe_config,
    }
    with open(
        os.path.join(args.output_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_config, f, indent=2, ensure_ascii=False
        )

    # 7. Evaluate
    print("\nEvaluating results...")
    eval_fn(results=results, output_dir=args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="MoE-Infinity Qwen3-VL-MoE Benchmark"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/fanya/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/data1/fanya/datasets/mmstar/MMStar.tsv"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/moe_infinity_benchmark",
    )
    parser.add_argument(
        "--offload-path",
        type=str,
        default=os.path.join(
            os.path.expanduser("~"),
            "moe-infinity-qwen3vl",
        ),
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50
    )
    parser.add_argument(
        "--num-samples", type=int, default=20
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
