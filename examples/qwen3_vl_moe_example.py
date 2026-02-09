import os
import sys

from moe_infinity import MoE

user_home = os.path.expanduser("~")

model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# step 1: check transformers version and model support
print("=" * 50)
print("Step 1: checking environment")
print("=" * 50)

try:
    from transformers import Qwen3VLMoeForConditionalGeneration

    print("[OK] Qwen3VLMoeForConditionalGeneration available")
except ImportError:
    print("[FAIL] transformers does not have Qwen3VLMoeForConditionalGeneration")
    print("Please upgrade transformers to a version that supports this model")
    sys.exit(1)

try:
    from transformers import AutoProcessor

    print("[OK] AutoProcessor available")
except ImportError:
    print("[FAIL] AutoProcessor not available")
    sys.exit(1)

# step 2: load config and verify attributes
print("\n" + "=" * 50)
print("Step 2: loading model config")
print("=" * 50)

from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"[OK] model_type: {config.model_type}")
print(f"[OK] architectures: {config.architectures}")

text_config = getattr(config, "text_config", config)
attrs = [
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "decoder_sparse_step",
    "mlp_only_layers",
    "hidden_size",
    "num_hidden_layers",
]
for attr in attrs:
    if hasattr(text_config, attr):
        val = getattr(text_config, attr)
        if isinstance(val, list) and len(val) > 5:
            print(f"  {attr}: [{val[0]}, {val[1]}, ... ] (len={len(val)})")
        else:
            print(f"  {attr}: {val}")
    else:
        print(f"  [WARN] {attr} not found in text_config")

# step 3: check monkey-patch target class
print("\n" + "=" * 50)
print("Step 3: checking monkey-patch target class")
print("=" * 50)

try:
    import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as mod

    target_cls = "Qwen3VLMoeTextSparseMoeBlock"
    if hasattr(mod, target_cls):
        print(f"[OK] {target_cls} found in modeling_qwen3_vl_moe")
    else:
        print(f"[WARN] {target_cls} NOT found")
        moe_classes = [
            n for n in dir(mod) if "moe" in n.lower() or "sparse" in n.lower()
        ]
        print(f"  Available MoE-related classes: {moe_classes}")
        print("  You may need to update the monkey-patch class name")
except ImportError:
    print("[WARN] Cannot import transformers.models.qwen3_vl_moe")

# step 4: check checkpoint param names
print("\n" + "=" * 50)
print("Step 4: checking checkpoint parameter names")
print("=" * 50)

from huggingface_hub import snapshot_download

model_path = snapshot_download(
    model_name,
    cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
    ignore_patterns=["flax*", "tf*"],
)
print(f"[OK] model downloaded to: {model_path}")

from safetensors import safe_open

safetensor_files = [
    f for f in os.listdir(model_path) if f.endswith(".safetensors")
]
if safetensor_files:
    with safe_open(
        os.path.join(model_path, safetensor_files[0]),
        framework="pt",
        device="cpu",
    ) as f:
        keys = list(f.keys())
    expert_keys = [k for k in keys if "expert" in k][:6]
    print(f"  Total params in first shard: {len(keys)}")
    print(f"  Expert param examples:")
    for k in expert_keys:
        print(f"    {k}")

# step 5: load model with MoE-Infinity
print("\n" + "=" * 50)
print("Step 5: loading model with MoE-Infinity")
print("=" * 50)

moe_config = {
    "offload_path": os.path.join(user_home, "moe-infinity-qwen3vl"),
    "device_memory_ratio": 0.75,
}

model = MoE(model_name, moe_config)
print("[OK] model loaded successfully")

# step 6: text-only inference
print("\n" + "=" * 50)
print("Step 6: text-only inference")
print("=" * 50)

processor = AutoProcessor.from_pretrained(model_name)

messages = [{"role": "user", "content": "Hello, who are you?"}]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=[text], return_tensors="pt")
input_ids = inputs.pop("input_ids").to("cuda:0")
kwargs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

import torch

output = model.generate(input_ids, max_new_tokens=64, **kwargs)
print("[OK] text output:", processor.decode(output[0], skip_special_tokens=True))

# step 7: multimodal inference (image + text)
print("\n" + "=" * 50)
print("Step 7: multimodal inference (image + text)")
print("=" * 50)

try:
    from PIL import Image
    import requests
    from io import BytesIO

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
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    input_ids = inputs.pop("input_ids").to("cuda:0")
    kwargs = {
        k: v.to("cuda:0") if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    output = model.generate(input_ids, max_new_tokens=128, **kwargs)
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
