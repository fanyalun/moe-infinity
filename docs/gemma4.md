# Gemma4 BF16

The initial checkpoint is `google/gemma-4-26B-A4B-it`, using its original
BF16 safetensors weights. Routed experts stay BF16 in the host store and GPU
cache. The adapter does not convert them to FP16 or a quantized format.

Only `Gemma4TextExperts` is replaced. Hugging Face retains the router,
per-expert routing scales, shared MLP, normalization, sliding and global
attention, tied embeddings, vision tower, and generation/KV lifecycle.
Non-expert parameters remain resident on the GPU. Expert computation uses
GELU with the PyTorch tanh approximation, followed by the up-projection
product and down projection.

## Run

Use a Transformers release providing `Gemma4ForConditionalGeneration` and
`Gemma4TextExperts`, and the matching moe-store Gemma4 adapter. Development
uses Transformers 5.18.0. The native MoE-Infinity extension must be rebuilt
for the new GELU-tanh activation. An older native extension does not provide
this activation.

The companion moe-store changes are included in
[`patches/moe_store_gemma4_bf16.patch`](../patches/moe_store_gemma4_bf16.patch).
They apply to moe-store commit
`1ded9ff58147f5b9d0234bedc1d924cdb7878b3d`. From that moe-store checkout,
apply the patch before building or importing this runtime:

```bash
git apply --check /path/to/MoE-Infinity/patches/moe_store_gemma4_bf16.patch
git apply /path/to/MoE-Infinity/patches/moe_store_gemma4_bf16.patch
```

Ensure that this patched moe-store checkout is on `PYTHONPATH` or installed
in the runtime environment; an unpatched released moe-store lacks the new
wrapper. Set `MOE_STORE_CSRC` to its `moe_store/csrc` directory when building
the native extension from source.

The runtime requires the Python `nvtx` package through moe-store. The
validated environment uses Python 3.12, PyTorch 2.14.1+cu130, Transformers
5.18.0, and nvtx 0.2.13, with the native extension built for an A100 (sm80).

```bash
mkdir -p artifacts/gemma4/tmp
export TMPDIR="$PWD/artifacts/gemma4/tmp"
CUDA_VISIBLE_DEVICES=0 python examples/gemma4_example.py \
  --checkpoint /path/to/gemma-4-26B-A4B-it \
  --offload_dir "$PWD/artifacts/gemma4/store" \
  --device_memory_ratio 0.25 \
  --prompt "What is the capital of France?" \
  --max_new_tokens 32
```

Add `--image /path/to/image.png --prompt "Describe this image."` for an
image request. The example uses the checkpoint's processor and chat
template and forwards all processor outputs to HF generation. Pixel values
are moved to BF16; routing probabilities retain the HF router's FP32 dtype.

Use a new, model-specific offload directory. `device_memory_ratio` controls
the expert cache allocation, not total GPU memory. Leave room for resident
weights, KV cache, and activations. Loading all BF16 expert weights into the
offload store also requires disk space for those weights.

## Validation and scope

The synchronous `MoE.generate()` path is the supported integration surface.
`MoE.serve()` has no Gemma4 paged-attention shim. Native paged generation,
CUDA Graph decoding, DFlash, adaptive expert precision, and quantized
Gemma4 checkpoints are outside this adapter's validated scope.

`tests/python/unit/test_gemma4_moe.py` covers packed expert shapes, BF16
preservation, resident parameters, override restoration, and tiny-model
text/image prefill, cached decoding, and greedy generation against the
unmodified HF model. Its CPU executor is a numerical reference rather than
native GPU offloading. The GPU integration test separately exercises the
real loader, expert cache, and native dispatcher.

HF references load with `dtype=torch.bfloat16` and SDPA, preserving HF's
FP32 RoPE frequency buffers. The native dispatcher accumulates weighted
expert outputs in FP32, so general bitwise parity with HF's eager BF16
expert reduction is not claimed. Repeated greedy tokens and prefill
logits are checked separately, including `torch.inference_mode()` calls.

```bash
mkdir -p artifacts/gemma4_validation/tmp
export TMPDIR="$PWD/artifacts/gemma4_validation/tmp"
python -m pytest -q tests/python/unit/test_gemma4_moe.py \
  --basetemp="$PWD/artifacts/gemma4_validation/pytest"
```

On 2026-11-11, validation on one A100 80 GB with CUDA 13.0 recorded:

| Check | Result |
| --- | --- |
| Runtime CPU regression selection (Gemma4, Qwen3-VL, activation, grouped-kernel gate) | 85 passed, 2 GPU/integration cases deselected |
| moe-store registry, conversion, and multimodal wrapper selection | 24 passed |
| Native CUDA activation operator selection | 17 passed |
| Tiny BF16 native offload, repeated image generation under `torch.inference_mode()` | 1 passed |
| Full `google/gemma-4-26B-A4B-it`, text and image greedy generation | Both requests passed, with tokens exactly matching the original HF BF16 model |

The full-checkpoint run used SDPA, 30 expert layers, 128 experts per layer,
top-8 routing, `device_memory_ratio=0.03`, and at most 16 new tokens per
request. All model parameters and stored expert weights stayed BF16; the
HF router and native weighted accumulation use FP32. The text prompt
`What is the capital of France?` produced
`The capital of France is **Paris**.` The image prompt
`What color is this image?` with a solid red 128x128 image produced
`The image is red.` Both output token lists matched the original HF model
loaded on CPU. The native routing counters recorded 450 batches, zero
failures, zero fallbacks, and no outstanding completion events.

These two short requests establish a full-checkpoint smoke test, not
general accuracy, universal token parity, or throughput. The run was marked
`formal_eligible=false` and `performance_benchmark=false`. The local
`artifacts/gemma4_adapter_20261111/full_smoke.json` records prompts, input
shapes, token lists, model dtypes, routing counters, and runtime scope.
