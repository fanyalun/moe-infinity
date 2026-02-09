# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoE-Infinity is a library for efficient Mixture-of-Experts (MoE) model inference on resource-constrained GPUs. It offloads MoE experts to host memory and uses activation-aware prefetching/caching to serve large MoE models. Based on the paper "MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache" (arXiv:2401.14361).

Supported model architectures: Switch Transformers, NLLB-MoE, Mixtral, Grok, Arctic, DeepSeek-V2/V3.

## Build & Development Commands

### Install from source (editable)
```bash
pip install -e .
conda install -c conda-forge libstdcxx-ng=12
```

### Build with pre-compiled C++ ops
```bash
BUILD_OPS=1 python -m build
```

### Lint and format
```bash
pip install -r requirements-lint.txt
pre-commit run --all-files
```

Pre-commit runs: ruff (lint + format), clang-format (C++), codespell, and standard hooks (trailing whitespace, YAML check, etc.).

### Run the OpenAI-compatible server
```bash
python -m moe_infinity.entrypoints.openai.api_server --model <model> --offload-dir <dir>
```

### Tests
Tests are integration tests (not pytest-based). They require a running server:
```bash
python tests/test_oai_completions.py
python tests/test_oai_chat_completions.py
```

C++ queue tests use CMake in `tests/queues/`.

## Architecture

### Two-layer design: Python orchestration + C++/CUDA core

**Python layer** (`moe_infinity/`):
- `entrypoints/big_modeling.py` — `MoE` class: main user-facing API. Wraps HuggingFace models, detects architecture from config, initializes the offload engine.
- `runtime/model_offload.py` — `OffloadEngine`: central coordinator. Manages expert loading, hooks into model forward passes, orchestrates tracing/prefetching/caching.
- `memory/` — Expert lifecycle management:
  - `expert_tracer.py`: records which experts activate per sequence
  - `expert_predictor.py`: predicts future expert usage from traces
  - `expert_prefetcher.py`: proactively loads predicted experts to GPU
  - `expert_cache.py`: priority-based GPU/CPU expert cache with eviction
- `models/` — Custom MoE layer implementations that replace HuggingFace originals (e.g., `SyncMixtralSparseMoeBlock`, `DeepseekMoEBlock`, `SyncSwitchTransformersSparseMLP`). Each model file patches the forward pass to route through the offload engine.
- `distributed/expert_executor.py` — Multi-GPU expert dispatch
- `utils/config.py` — `ArcherConfig`: configuration loading (JSON or dict) with fields like `offload_path`, `device_memory_ratio`, `trace_capacity`

**C++/CUDA layer** (`core/`):
- `aio/` — Async I/O for disk-to-memory expert transfers
- `memory/` — Memory pool management and caching primitives
- `parallel/` — Expert dispatcher for GPU execution
- `prefetch/` — Prefetching engine implementation
- `python/` — pybind11 bindings exposed as `moe_infinity.ops.prefetch`

**Op builder** (`op_builder/`): Build system for C++ extensions (adapted from DeepSpeed). Uses PyTorch's `cpp_extension` with Ninja. The `prefetch` op is the main extension.

### Data flow
1. User calls `MoE(model_name, config)` → downloads/loads HuggingFace model
2. `OffloadEngine` replaces MoE layers with custom implementations and registers forward hooks
3. During inference, custom MoE layers consult the `ExpertTracer` and `ExpertPrefetcher`
4. Experts are loaded from disk → CPU → GPU on demand via async I/O
5. `ExpertCache` manages GPU memory with activation-aware eviction

### Key configuration options (passed as dict to `MoE`)
- `offload_path`: directory for expert weight storage
- `device_memory_ratio`: fraction of GPU memory to use for expert cache (0.0–1.0)
- `trace_capacity`: size of the activation trace buffer

## Conventions

- Python: ruff for linting/formatting, line length 80, ruff handles isort
- C++: clang-format v18.1.4
- License: Apache 2.0
- Python ≥ 3.8, PyTorch ≥ 2.1.1, transformers ≥ 4.37.1 and < 4.47
- CUDA toolkit required; C++17 compiler required for building ops

## Project Conventions

- 使用中文回复
- 文件名小写 + 下划线连接
- 代码注释用单行注释
- 需要安装/卸载的依赖告知用户，不自行安装
- 保持代码简洁
- 对话时称呼用户为“主人”