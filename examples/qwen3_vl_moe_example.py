import os

import torch
from transformers import AutoProcessor

from moe_infinity import MoE

user_home = os.path.expanduser("~")

model_name = os.environ.get(
    "MODEL_PATH",
    "/data1/fanya/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
)

# load model with MoE-Infinity
print("=" * 50)
print("Step 1: loading model with MoE-Infinity")
print("=" * 50)

moe_config = {
    "offload_path": os.path.join(user_home, "moe-infinity-qwen3vl"),
    "device_memory_ratio": 0.75,
}

model = MoE(model_name, moe_config)
print("[OK] model loaded successfully")

# text-only inference
print("\n" + "=" * 50)
print("Step 2: text-only inference")
print("=" * 50)

processor = AutoProcessor.from_pretrained(model_name)

messages = [{"role": "user", "content": "Hello, who are you?"}]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=[text], return_tensors="pt")
input_ids = inputs.pop("input_ids").to("cuda:0")
kwargs = {
    k: v.to("cuda:0") if hasattr(v, "to") else v
    for k, v in inputs.items()
}

output = model.generate(input_ids, max_new_tokens=64, **kwargs)
print(
    "[OK] text output:",
    processor.decode(output[0], skip_special_tokens=True),
)

# multimodal inference (image + text)
print("\n" + "=" * 50)
print("Step 3: multimodal inference (image + text)")
print("=" * 50)

try:
    import requests
    from io import BytesIO

    from PIL import Image

    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    print(f"  Downloading test image from: {url}")
    response = requests.get(url, timeout=30)
    image = Image.open(BytesIO(response.content))
    print(f"  Image size: {image.size}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Describe this image briefly.",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    )
    input_ids = inputs.pop("input_ids").to("cuda:0")
    kwargs = {
        k: v.to("cuda:0") if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    output = model.generate(
        input_ids, max_new_tokens=128, **kwargs
    )
    print(
        "[OK] multimodal output:",
        processor.decode(output[0], skip_special_tokens=True),
    )
except Exception as e:
    print(f"[FAIL] multimodal inference failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)
