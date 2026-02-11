"""Offline trace generation for MoE-Infinity prefetching.

Runs inference on a dataset and records expert activation patterns.
The resulting trace file can be loaded via `trace_path` config to
enable activation-aware prefetching.
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from moe_infinity import MoE

sys.path.insert(0, "/home/fanya/qwen3_vl_moe_test")
from qwen_vl_utils import process_vision_info
from utils.dataset import (
    load_mmbench_dataset,
    load_mmstar_dataset,
    load_realworldqa_dataset,
)
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


def detect_dataset_key(data_dir):
    lower = data_dir.lower()
    for key in DATASET_LOADERS:
        if key in lower:
            return key
    raise ValueError(f"Unknown dataset in {data_dir}")


def generate_trace(args):
    print("\n" + "=" * 60)
    print("MoE-Infinity Trace Generation")
    print("=" * 60 + "\n")

    # 1. Load Dataset
    dataset_key = detect_dataset_key(args.data_dir)
    load_dataset = DATASET_LOADERS[dataset_key]
    build_prompt = PROMPT_BUILDERS[dataset_key]

    print(f"Loading {dataset_key} from {args.data_dir}...")
    dataset = load_dataset(args.data_dir)
    dataset = dataset.sample(
        n=min(args.num_samples, len(dataset)),
        random_state=42,
    ).reset_index(drop=True)
    print(f"Using {len(dataset)} samples")

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

    parent_dir = os.path.dirname(args.data_dir)
    img_root = os.path.join(parent_dir, "images")
    os.makedirs(img_root, exist_ok=True)

    # 3. Run inference and collect traces
    completed = 0
    for idx, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc="Generating traces",
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

            # set up trace entries
            moe._configure_hook(inputs.input_ids)

            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            # finish trace entries
            for seq_id in moe.seq_id_list:
                moe.engine.expert_tracer.finish_entry(
                    seq_id
                )
            completed += 1

        except Exception as e:
            print(f"Error sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. Save trace
    os.makedirs(args.output_dir, exist_ok=True)
    trace_path = os.path.join(
        args.output_dir, "expert_trace.npy"
    )
    moe.engine.expert_tracer.save_trace(trace_path)
    print(
        f"\nDone. {completed}/{len(dataset)} samples "
        f"processed. Trace saved to {trace_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate expert activation traces"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/fanya/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data1/fanya/datasets/mmstar/MMStar.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/traces",
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
        default=0.0675,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50
    )
    parser.add_argument(
        "--num-samples", type=int, default=100
    )

    args = parser.parse_args()
    generate_trace(args)


if __name__ == "__main__":
    main()
