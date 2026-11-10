# Qwen3-VL-MoE

The initial target is `Qwen/Qwen3-VL-30B-A3B-Instruct`, using its unquantized
BF16 safetensors checkpoint. The synchronous `MoE.generate()` path accepts
text and image inputs. Routed experts use MoE-Infinity's existing dispatcher
and cache; the vision encoder, DeepStack mergers, text attention, router,
embeddings, norms, and language-model head stay resident on GPU.

Generation and KV ownership stay with Hugging Face. This preserves the
model's multimodal position handling and cached decoding. Native paged
serving, CUDA graph decoding, DFlash, and quantized checkpoints are not
validated for this family. `MoE.serve()` has no Qwen3-VL paged-attention shim.

## Run an image request

Use `moe-store` 0.2.2 and a Transformers version that provides
`Qwen3VLMoeTextTopKRouter`. CPU validation uses Transformers 5.18.0 and
PyTorch 2.14.1. A compatible build of the native MoE-Infinity extensions is
required for GPU expert offloading; Python-only tests do not replace it.

Start with a fresh, model-specific offload directory. Published checkpoints
store `gate_up_proj` as `[E, H, 2I]` and `down_proj` as `[E, I, H]`; newer HF
exports use `[E, 2I, H]` and `[E, H, I]`. The loader validates both layouts
against the text config before splitting experts. Stores produced by an
older loader or external converter are not covered by this validation.

```bash
mkdir -p artifacts/qwen3_vl/tmp
export TMPDIR="$PWD/artifacts/qwen3_vl/tmp"
CUDA_VISIBLE_DEVICES=0 python examples/vl_example.py \
  --checkpoint /path/to/Qwen3-VL-30B-A3B-Instruct \
  --image /path/to/image.png \
  --prompt "Describe this image briefly." \
  --offload_dir "$PWD/artifacts/qwen3_vl/store" \
  --device_memory_ratio 0.25 \
  --max_new_tokens 32
```

`device_memory_ratio` configures the expert-cache allocation. It does not
cap total GPU usage: resident weights, activations, and KV consume memory
as well. Set the cache size with those allocations in mind.

For an in-process caller, pass every processor output to `MoE.generate()`.
In particular, preserve `pixel_values`, `image_grid_thw`,
`mm_token_type_ids`, and `attention_mask` when the processor returns them.
Do not reconstruct multimodal positions from text token IDs alone.

## Validation boundary

`tests/python/unit/test_qwen3_vl_moe.py` compares both packed layouts with
HF's expert computation, checks resident-weight selection and override
restoration, and compares a tiny text/vision model's prefill logits, cached
decode logits, and greedy tokens with an unmodified HF reference. The CPU
dispatcher used by those tests is a numerical reference, not native GPU
offloading.

The same file includes a GPU integration test that builds a tiny legacy-layout
checkpoint, initializes the real `MoE` loader and native store, verifies that
expert parameters are placeholders bound to the native executor, and compares
two image generations with HF. It skips when CUDA or the native extension is
unavailable.

```bash
mkdir -p artifacts/qwen3_vl_validation/tmp
export TMPDIR="$PWD/artifacts/qwen3_vl_validation/tmp"
python -m pytest -q tests/python/unit/test_qwen3_vl_moe.py \
  --basetemp="$PWD/artifacts/qwen3_vl_validation/pytest"
```

A completed 30B GPU load, real image generation, greedy reference comparison,
and performance measurements must be recorded separately before reporting a
validated full-model baseline. The adapter leaves upstream prefetch and
expert-drop settings at their configured values; record those values in
benchmark results.
