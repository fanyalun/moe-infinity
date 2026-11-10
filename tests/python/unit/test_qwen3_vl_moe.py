from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from moe_store.wrappers import SyncQwen3VLMoeTextSparseMoeBlock
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as hf

from moe_infinity.boundary import transformers_model_overrides as overrides
from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.runtime.model_offload import (
    OffloadEngine,
    _remap_v5_batched_experts,
)


def _config():
    config = Qwen3VLMoeConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 128,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=125,
        video_token_id=124,
        vision_start_token_id=126,
        vision_end_token_id=127,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3VLMoeForConditionalGeneration"]
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("legacy_layout", [False, True])
@pytest.mark.parametrize("separate_shards", [False, True])
def test_packed_weights_preserve_hf_expert_outputs(
    legacy_layout, separate_shards
):
    torch.manual_seed(5)
    config = _config()
    reference = hf.Qwen3VLMoeTextSparseMoeBlock(config.text_config).eval()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state = dict(reference.state_dict())
    if legacy_layout:
        for key in ("experts.gate_up_proj", "experts.down_proj"):
            state[key] = state[key].transpose(1, 2).contiguous()
    prefix = "model.language_model.layers.0.mlp."
    if separate_shards:
        remapped = {}
        for key, value in state.items():
            shard = {prefix + key: value}
            _remap_v5_batched_experts(shard, config)
            remapped.update(shard)
    else:
        remapped = {prefix + key: value for key, value in state.items()}
        _remap_v5_batched_experts(remapped, config)
    block = SyncQwen3VLMoeTextSparseMoeBlock(config.text_config).eval()
    block.load_state_dict(
        {key.removeprefix(prefix): value for key, value in remapped.items()}
    )
    hidden = torch.randn(2, 5, config.text_config.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(
            block(hidden), reference(hidden), rtol=1e-5, atol=1e-6
        )


def test_invalid_packed_shape_is_rejected_before_offload():
    config = _config()
    state = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": (
            torch.zeros(4, 17, 19)
        )
    }
    with pytest.raises(ValueError, match="shape"):
        _remap_v5_batched_experts(state, config)


@pytest.mark.parametrize(
    "name,resident",
    [
        ("model.visual.pos_embed.weight", True),
        ("model.visual.deepstack_merger_list.0.linear_fc1.weight", True),
        ("model.language_model.embed_tokens.weight", True),
        ("model.language_model.layers.0.self_attn.q_proj.weight", True),
        ("model.language_model.layers.0.mlp.gate.weight", True),
        ("lm_head.weight", True),
        ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", False),
        ("model.language_model.layers.1.mlp.experts.3.down_proj.weight", False),
    ],
)
def test_only_routed_expert_weights_are_offloaded(name, resident):
    engine = object.__new__(OffloadEngine)
    engine.config = _config()
    assert engine._is_shared_expert_param(name) is resident


def test_override_installs_and_restores_qwen3_vl_block():
    original = hf.Qwen3VLMoeTextSparseMoeBlock
    try:
        overrides.install_transformers_model_overrides()
        assert (
            hf.Qwen3VLMoeTextSparseMoeBlock is SyncQwen3VLMoeTextSparseMoeBlock
        )
        overrides.restore_context_transformers_model_overrides()
        assert hf.Qwen3VLMoeTextSparseMoeBlock is original
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()
        hf.Qwen3VLMoeTextSparseMoeBlock = original


def test_generate_preserves_multimodal_kwargs_and_hf_kv_path():
    moe = MoE.__new__(MoE)
    moe.model = MagicMock()
    moe.model.config = _config()
    moe.use_native_engine = True
    moe._native_generation_engine = MagicMock()
    moe._native_generation_engine.generate.return_value = SimpleNamespace(
        output_token_ids=[7]
    )
    moe._configure_hook = MagicMock()
    moe._resolve_spec_strategy = MagicMock()
    moe.max_seq_length = 128
    moe.eos_token_id = 2
    input_ids = torch.tensor([[1, 126, 125, 127, 5]])
    kwargs = {
        "pixel_values": torch.randn(4, 24),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": (input_ids == 125).long(),
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": 3,
        "do_sample": False,
    }
    output = moe.generate(input_ids, **kwargs)
    assert output is moe.model.generate.return_value
    moe.model.generate.assert_called_once_with(input_ids, **kwargs)
    moe._native_generation_engine.generate.assert_not_called()


class _CpuExecutor:
    def __init__(self, weights):
        self.weights = weights
        self.calls = []
        self.output = None

    def dispatch_local(
        self, layer_id, hidden, mask, weights, router_logits=None
    ):
        assert router_logits is not None
        self.calls.append((layer_id, hidden.shape[0]))
        output = torch.zeros_like(hidden)
        prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        for expert in range(mask.shape[1]):
            rows = mask[:, expert].nonzero().flatten()
            if rows.numel() == 0:
                continue
            base = f"{prefix}.{expert}"
            gate = F.linear(
                hidden[rows], self.weights[f"{base}.gate_proj.weight"]
            )
            up = F.linear(hidden[rows], self.weights[f"{base}.up_proj.weight"])
            values = F.linear(
                F.silu(gate) * up, self.weights[f"{base}.down_proj.weight"]
            )
            output.index_add_(0, rows, values * weights[rows, expert, None])
        self.output = output

    def wait_dispatch_local(self):
        return self.output


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_tiny_model_prefill_cached_decode_and_greedy_tokens_match_hf(
    with_image, dtype, attention
):
    torch.manual_seed(17)
    config = _config()
    config._attn_implementation = attention
    reference = Qwen3VLMoeForConditionalGeneration(config).to(dtype).eval()
    state = {
        key: value.clone() for key, value in reference.state_dict().items()
    }
    for key in list(state):
        if key.endswith(("experts.gate_up_proj", "experts.down_proj")):
            state[key] = state[key].transpose(1, 2).contiguous()
    _remap_v5_batched_experts(state, config)
    try:
        overrides.install_transformers_model_overrides()
        adapted = Qwen3VLMoeForConditionalGeneration(config).to(dtype).eval()
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()
    adapted.load_state_dict(state)
    executor = _CpuExecutor(state)
    blocks = []
    for name, module in adapted.named_modules():
        if isinstance(module, SyncQwen3VLMoeTextSparseMoeBlock):
            module.expert_executor = executor
            module.layer_id = int(name.split(".layers.")[1].split(".")[0])
            blocks.append(module)
            for parameter in module.experts.parameters():
                parameter.data.zero_()
    assert len(blocks) == config.text_config.num_hidden_layers
    kwargs = {}
    if with_image:
        input_ids = torch.tensor([[1, 126, 125, 127, 5, 6]])
        kwargs = {
            "pixel_values": torch.randn(4, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": (input_ids == 125).long(),
        }
    else:
        input_ids = torch.tensor([[1, 5, 6, 7]])
    with torch.no_grad():
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-3)
        expected = reference(input_ids, use_cache=True, **kwargs)
        actual = adapted(input_ids, use_cache=True, **kwargs)
        torch.testing.assert_close(
            actual.logits, expected.logits, rtol=rtol, atol=atol
        )
        next_token = expected.logits[:, -1].argmax(dim=-1, keepdim=True)
        expected_decode = reference(
            next_token, past_key_values=expected.past_key_values, use_cache=True
        )
        actual_decode = adapted(
            next_token, past_key_values=actual.past_key_values, use_cache=True
        )
        torch.testing.assert_close(
            actual_decode.logits, expected_decode.logits, rtol=rtol, atol=atol
        )
        expected_tokens = reference.generate(
            input_ids, max_new_tokens=3, do_sample=False, **kwargs
        )
        actual_tokens = adapted.generate(
            input_ids, max_new_tokens=3, do_sample=False, **kwargs
        )
    assert torch.equal(actual_tokens, expected_tokens)
    assert any(tokens > 1 for _, tokens in executor.calls)
    assert any(tokens == 1 for _, tokens in executor.calls)


@pytest.mark.gpu
@pytest.mark.integration
def test_native_offload_matches_hf_on_image_and_repeated_generation(tmp_path):
    import moe_infinity._store as native_store

    if not isinstance(native_store, ModuleType):
        pytest.skip("MoE-Infinity native store extension is not built")
    torch.manual_seed(23)
    config = _config()
    config.dtype = torch.bfloat16
    config.text_config.dtype = torch.bfloat16
    config.text_config.hidden_size = 64
    config.text_config.head_dim = 16
    config.text_config.moe_intermediate_size = 48
    config.text_config.rope_parameters["mrope_section"] = [2, 2, 4]
    config.vision_config.out_hidden_size = 64
    reference = Qwen3VLMoeForConditionalGeneration(config).bfloat16().eval()
    legacy = {
        key: value.clone() for key, value in reference.state_dict().items()
    }
    for key in list(legacy):
        if key.endswith(("experts.gate_up_proj", "experts.down_proj")):
            legacy[key] = legacy[key].transpose(1, 2).contiguous()
    checkpoint = tmp_path / "qwen3_vl_checkpoint"
    reference.save_pretrained(checkpoint, state_dict=legacy)
    reference = reference.to("cuda:0")
    moe = MoE(
        str(checkpoint),
        {
            "offload_path": str(tmp_path / "offload_store"),
            "device_memory_ratio": 0.001,
            "trace_capacity": 4,
        },
    )
    assert moe.use_native_engine is False
    assert len(moe.engine.expert_layer_modules) == 2
    assert {module.layer_id for module in moe.engine.expert_layer_modules} == {
        0,
        1,
    }
    for module in moe.engine.expert_layer_modules:
        assert module.expert_executor is moe.engine.expert_executor
        assert all(
            parameter.numel() == 1 for parameter in module.experts.parameters()
        )
    assert all(".experts." in name for name in moe.engine.name_id_map)
    input_ids = torch.tensor([[1, 126, 125, 127, 5, 6]], device="cuda:0")
    kwargs = {
        "pixel_values": torch.randn(4, 24, device="cuda:0"),
        "image_grid_thw": torch.tensor([[1, 2, 2]], device="cuda:0"),
        "mm_token_type_ids": (input_ids == 125).long(),
        "max_new_tokens": 3,
        "do_sample": False,
    }
    with torch.no_grad():
        expected = reference.generate(input_ids, **kwargs)
        for _ in range(2):
            actual = moe.generate(input_ids, **kwargs)
            assert torch.equal(actual, expected)
