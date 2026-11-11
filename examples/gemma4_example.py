import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor

from moe_infinity import MoE


def main():
    parser = argparse.ArgumentParser(
        description="Gemma4 BF16 expert offloading"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offload_dir", required=True)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--image")
    parser.add_argument("--device_memory_ratio", type=float, default=0.25)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    content = []
    if args.image:
        content.append(
            {"type": "image", "url": str(Path(args.image).resolve())}
        )
    content.append({"type": "text", "text": args.prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {
        key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    model = MoE(
        args.checkpoint,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
            "use_native_engine": False,
        },
    )
    if model.engine.dtype_cls != torch.bfloat16:
        raise ValueError("This example requires the BF16 Gemma4 checkpoint")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    print(
        processor.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    main()
