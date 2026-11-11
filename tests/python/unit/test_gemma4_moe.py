import copy
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from moe_store.wrappers import SyncGemma4TextExperts
from transformers import Gemma4Config, Gemma4ForConditionalGeneration
from transformers.models.gemma4 import modeling_gemma4 as hf

from moe_infinity.boundary import transformers_model_overrides as overrides
from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.runtime.hf_attention import select_attn_implementation
from moe_infinity.runtime.model_offload import (
    OffloadEngine,
    _remap_v5_batched_experts,
)


def _config(
    hidden_size=32, head_dim=8, global_head_dim=8, moe_intermediate_size=24
):
    config = Gemma4Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": hidden_size,
            "intermediate_size": 64,
            "moe_intermediate_size": moe_intermediate_size,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": head_dim,
            "global_head_dim": global_head_dim,
            "num_experts": 4,
            "top_k_experts": 2,
            "enable_moe_block": True,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size_per_layer_input": 0,
            "max_position_embeddings": 128,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "attention_k_eq_v": True,
            "use_bidirectional_attention": "vision",
            "final_logit_softcapping": 30.0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": True,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "standardize": True,
            "position_embedding_size": 16,
            "max_position_embeddings": 64,
            "rope_parameters": {"rope_type": "axial", "rope_theta": 100.0},
        },
        audio_config=None,
        image_token_id=125,
        video_token_id=124,
        audio_token_id=123,
        boi_token_id=126,
        eoi_token_id=127,
        tie_word_embeddings=True,
    )
    config.architectures = ["Gemma4ForConditionalGeneration"]
    config._attn_implementation = "eager"
    # HF grouped_mm uses a different BF16 reduction from its eager reference.
    config._experts_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    return config


def _set_nonuniform_router_scales(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, hf.Gemma4TextRouter):
                module.per_expert_scale.copy_(
                    torch.tensor([0.5, 1.25, 2.0, 3.0])
                )
                module.scale.copy_(torch.linspace(0.7, 1.3, module.hidden_size))
            elif isinstance(module, hf.Gemma4TextDecoderLayer):
                module.layer_scalar.fill_(1.25)
            elif isinstance(module, hf.Gemma4VisionModel):
                module.std_bias.copy_(torch.linspace(0.1, 0.4, 16))
                module.std_scale.fill_(1.5)


def _inputs(with_image, device="cpu"):
    if with_image:
        input_ids = torch.tensor([[1, 126, 125, 127, 5, 6]], device=device)
        kwargs = {
            "pixel_values": torch.rand(1, 4, 12, device=device),
            "image_position_ids": torch.tensor(
                [[[0, 0], [1, 0], [0, 1], [1, 1]]], device=device
            ),
            "mm_token_type_ids": (input_ids == 125).long(),
        }
    else:
        input_ids = torch.tensor([[1, 5, 6, 7]], device=device)
        kwargs = {}
    kwargs["attention_mask"] = torch.ones_like(input_ids)
    return input_ids, kwargs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_sentinel", [False, True])
def test_packed_weights_preserve_expert_outputs_and_router_scaling(
    dtype, with_sentinel
):
    torch.manual_seed(31)
    config = _config()
    reference = hf.Gemma4TextExperts(config.text_config).to(dtype).eval()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    prefix = "model.language_model.layers.0.experts."
    state = {
        prefix + key: value for key, value in reference.state_dict().items()
    }
    _remap_v5_batched_experts(state, config)
    assert state and all(value.dtype == dtype for value in state.values())
    assert not any(key.endswith(("gate_up_proj", "down_proj")) for key in state)
    adapted = SyncGemma4TextExperts(config.text_config).to(dtype).eval()
    adapted.load_state_dict(
        {key.removeprefix(prefix): value for key, value in state.items()}
    )
    router = hf.Gemma4TextRouter(config.text_config).to(dtype).eval()
    _set_nonuniform_router_scales(router)
    hidden = torch.randn(10, config.text_config.hidden_size, dtype=dtype)
    with torch.no_grad():
        _, weights, selected = router(hidden)
        assert weights.dtype == torch.float32
        assert not torch.allclose(weights.sum(-1), torch.ones(10))
        if with_sentinel:
            selected[0, 0] = config.text_config.num_experts
        expected = reference(hidden, selected, weights)
        actual = adapted(hidden, selected, weights)
        assert actual.dtype == dtype
        tolerance = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-3)
        torch.testing.assert_close(
            actual, expected, rtol=tolerance[0], atol=tolerance[1]
        )


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_invalid_packed_shape_is_rejected(projection):
    state = {
        f"model.language_model.layers.0.experts.{projection}": torch.zeros(
            4, 17, 19, dtype=torch.bfloat16
        )
    }
    with pytest.raises(ValueError, match="Gemma4.*shape"):
        _remap_v5_batched_experts(state, _config())


@pytest.mark.parametrize(
    "name,resident",
    [
        ("model.vision_tower.patch_embedder.input_proj.weight", True),
        ("model.embed_vision.embedding_projection.weight", True),
        ("model.language_model.embed_tokens.weight", True),
        ("model.language_model.layers.0.self_attn.q_proj.weight", True),
        ("model.language_model.layers.0.router.proj.weight", True),
        ("model.language_model.layers.0.router.per_expert_scale", True),
        ("model.language_model.layers.0.mlp.gate_proj.weight", True),
        (
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
            True,
        ),
        ("model.language_model.layers.0.layer_scalar", True),
        ("lm_head.weight", True),
        ("model.language_model.layers.0.experts.0.gate_proj.weight", False),
        ("model.language_model.layers.1.experts.3.down_proj.weight", False),
    ],
)
def test_only_routed_experts_are_offloaded(name, resident):
    engine = object.__new__(OffloadEngine)
    engine.config = _config()
    assert engine._is_shared_expert_param(name) is resident


def test_gemma4_topology_groups_experts_and_preserves_projection_slots():
    config = _config()
    try:
        overrides.install_transformers_model_overrides()
        model = Gemma4ForConditionalGeneration(config)
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()
    engine = object.__new__(OffloadEngine)
    engine.config = config
    engine.name_id_map = {
        name: index
        for index, (name, _) in enumerate(model.named_parameters())
        if ".experts." in name
    }
    topology = engine.get_topology(model)
    assert len(topology) == config.text_config.num_hidden_layers
    for layer, (name, groups) in enumerate(topology):
        assert name == f"model.language_model.layers.{layer}.experts"
        assert len(groups) == config.text_config.num_experts
        for expert, ids in enumerate(groups):
            assert ids == [
                engine.name_id_map[f"{name}.{expert}.{projection}_proj.weight"]
                for projection in ("gate", "up", "down")
            ]
    engine.name_id_map.pop(next(iter(engine.name_id_map)))
    with pytest.raises(ValueError, match="Gemma4 expert topology"):
        engine.get_topology(model)


def test_override_installs_and_restores_only_gemma4_experts():
    original = hf.Gemma4TextExperts
    original_decoder = hf.Gemma4TextDecoderLayer
    original_router = hf.Gemma4TextRouter
    original_init_weights = hf.Gemma4PreTrainedModel._init_weights
    try:
        overrides.install_transformers_model_overrides()
        assert hf.Gemma4TextExperts is SyncGemma4TextExperts
        assert hf.Gemma4TextDecoderLayer is original_decoder
        assert hf.Gemma4TextRouter is original_router
        overrides.restore_context_transformers_model_overrides()
        assert hf.Gemma4TextExperts is original
        assert hf.Gemma4PreTrainedModel._init_weights is original_init_weights
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()
        hf.Gemma4TextExperts = original


def test_missing_gemma4_module_keeps_other_model_overrides_usable(monkeypatch):
    model_module = overrides._model_module
    qwen3 = model_module("qwen3_moe")
    original_qwen3 = qwen3.Qwen3MoeSparseMoeBlock
    original_gemma4 = hf.Gemma4TextExperts
    original_init_weights = hf.Gemma4PreTrainedModel._init_weights

    def without_gemma4(name):
        if name == "gemma4":
            raise ModuleNotFoundError("transformers.models.gemma4 unavailable")
        return model_module(name)

    monkeypatch.setattr(overrides, "_model_module", without_gemma4)
    try:
        overrides.install_transformers_model_overrides()
        assert qwen3.Qwen3MoeSparseMoeBlock is overrides.Qwen3MoEBlock
        assert hf.Gemma4TextExperts is original_gemma4
        assert hf.Gemma4PreTrainedModel._init_weights is original_init_weights
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()
        assert qwen3.Qwen3MoeSparseMoeBlock is original_qwen3
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()


def test_generate_keeps_hf_multimodal_and_cache_ownership():
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
    input_ids, kwargs = _inputs(with_image=True)
    kwargs.update(max_new_tokens=3, do_sample=False)
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
        assert weights.dtype == torch.float32
        self.calls.append((layer_id, hidden.shape[0]))
        output = torch.zeros_like(hidden)
        prefix = f"model.language_model.layers.{layer_id}.experts"
        for expert in range(mask.shape[1]):
            rows = mask[:, expert].nonzero().flatten()
            if rows.numel() == 0:
                continue
            base = f"{prefix}.{expert}"
            packed = torch.cat(
                (
                    self.weights[f"{base}.gate_proj.weight"],
                    self.weights[f"{base}.up_proj.weight"],
                ),
                dim=0,
            )
            gate, up = F.linear(hidden[rows], packed).chunk(2, dim=-1)
            values = F.linear(
                F.gelu(gate, approximate="tanh") * up,
                self.weights[f"{base}.down_proj.weight"],
            )
            weighted = values * weights[rows, expert, None]
            output.index_add_(0, rows, weighted.to(output.dtype))
        self.output = output

    def wait_dispatch_local(self):
        return self.output


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_tiny_bf16_prefill_cached_decode_and_greedy_match_hf(
    with_image, attention
):
    _assert_tiny_bf16_model_parity(with_image, attention)


def test_tiny_bf16_sdpa_global_head_dim512_matches_hf():
    config = _config(global_head_dim=512)
    implementation = select_attn_implementation(
        Gemma4ForConditionalGeneration, config, False
    )
    assert implementation == "sdpa"
    _assert_tiny_bf16_model_parity(False, "sdpa", global_head_dim=512)


def _assert_tiny_bf16_model_parity(with_image, attention, global_head_dim=8):
    torch.manual_seed(37)
    config = _config(global_head_dim=global_head_dim)
    config._attn_implementation = attention
    reference = Gemma4ForConditionalGeneration(config).bfloat16().eval()
    full_attention = reference.model.language_model.layers[1].self_attn
    assert full_attention.head_dim == global_head_dim
    assert full_attention.v_proj is None
    _set_nonuniform_router_scales(reference)
    state = {
        key: value.clone() for key, value in reference.state_dict().items()
    }
    _remap_v5_batched_experts(state, config)
    adapted_config = copy.deepcopy(config)
    adapted_config._attn_implementation = attention
    try:
        overrides.install_transformers_model_overrides()
        adapted = (
            Gemma4ForConditionalGeneration(adapted_config).bfloat16().eval()
        )
    finally:
        overrides.restore_context_transformers_model_overrides()
        overrides.restore_runtime_transformers_model_overrides()
    adapted.load_state_dict(state)
    assert adapted.lm_head.weight is adapted.get_input_embeddings().weight
    executor = _CpuExecutor(state)
    blocks = []
    for name, module in adapted.named_modules():
        if isinstance(module, SyncGemma4TextExperts):
            module.expert_executor = executor
            module.layer_id = int(name.split(".layers.")[1].split(".")[0])
            blocks.append(module)
            for parameter in module.parameters():
                parameter.data.zero_()
    assert len(blocks) == config.text_config.num_hidden_layers
    input_ids, kwargs = _inputs(with_image)
    with torch.no_grad():
        expected = reference(input_ids, use_cache=True, **kwargs)
        actual = adapted(input_ids, use_cache=True, **kwargs)
        torch.testing.assert_close(
            actual.logits, expected.logits, rtol=1e-2, atol=1e-3
        )
        next_token = expected.logits[:, -1].argmax(dim=-1, keepdim=True)
        decode_mask = torch.ones(1, input_ids.shape[1] + 1, dtype=torch.long)
        expected_decode = reference(
            next_token,
            attention_mask=decode_mask,
            past_key_values=expected.past_key_values,
            use_cache=True,
        )
        actual_decode = adapted(
            next_token,
            attention_mask=decode_mask,
            past_key_values=actual.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(
            actual_decode.logits, expected_decode.logits, rtol=1e-2, atol=1e-3
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
def test_native_bf16_offload_tied_embeddings_and_image_generation(tmp_path):
    import moe_infinity._store as native_store

    if not isinstance(native_store, ModuleType):
        pytest.skip("MoE-Infinity native store extension is not built")
    torch.manual_seed(41)
    config = _config(
        hidden_size=64,
        head_dim=16,
        global_head_dim=512,
        moe_intermediate_size=48,
    )
    config.dtype = torch.bfloat16
    config.text_config.dtype = torch.bfloat16
    # Construct in BF16 so HF keeps nonpersistent RoPE frequencies in FP32.
    reference = Gemma4ForConditionalGeneration._from_config(
        config, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    assert (
        reference.model.language_model.rotary_emb.full_attention_inv_freq.dtype
        == torch.float32
    )
    _set_nonuniform_router_scales(reference)
    checkpoint = tmp_path / "gemma4_checkpoint"
    reference.save_pretrained(checkpoint)
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
    assert moe.model.lm_head.weight is moe.model.get_input_embeddings().weight
    assert moe.model.lm_head.weight.dtype == torch.bfloat16
    assert moe.model.model.language_model.layers[1].self_attn.head_dim == 512
    reference_buffers = dict(reference.named_buffers())
    for name, buffer in moe.model.named_buffers():
        if name.endswith(("layer_scalar", "std_bias", "std_scale")):
            torch.testing.assert_close(
                buffer.cpu(), reference_buffers[name].cpu(), rtol=0, atol=0
            )
    for module in moe.engine.expert_layer_modules:
        assert module.expert_executor is moe.engine.expert_executor
        assert all(parameter.numel() == 1 for parameter in module.parameters())
    assert all(".experts." in name for name in moe.engine.name_id_map)
    for name, parameter in moe.model.named_parameters():
        if ".experts." not in name:
            assert parameter.dtype == torch.bfloat16
    input_ids, kwargs = _inputs(with_image=True, device="cuda:0")
    with torch.no_grad():
        moe._configure_hook(input_ids)
        expected = reference(input_ids, use_cache=True, **kwargs)
        actual = moe.model(input_ids, use_cache=True, **kwargs)
        torch.testing.assert_close(
            actual.logits, expected.logits, rtol=5e-2, atol=1e-2
        )
        expected_tokens = reference.generate(
            input_ids, max_new_tokens=3, do_sample=False, **kwargs
        )
        with torch.inference_mode():
            for _ in range(2):
                actual_tokens = moe.generate(
                    input_ids, max_new_tokens=3, do_sample=False, **kwargs
                )
                assert torch.equal(actual_tokens, expected_tokens)
