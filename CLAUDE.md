# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openpi is Physical Intelligence's open-source robotics VLA (Vision-Language-Action) framework. It contains three model families: pi0 (flow-based), pi0-FAST (autoregressive with FAST tokenizer), and pi0.5 (improved generalization via knowledge insulation). Supports both JAX/Flax and PyTorch backends.

## Common Commands

### Environment Setup
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
Requires Python 3.11. Uses `uv` as package manager (not pip). `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot dependency.

### Linting and Formatting
```bash
ruff check .              # Lint
ruff check . --fix        # Lint with auto-fix
ruff format .             # Format
```
Ruff config: line length 120, target Python 3.11. Excludes `docker/`, `third_party/`, `src/openpi/models_pytorch/transformers_replace/*`.

### Pre-commit
```bash
pre-commit install
pre-commit run --all-files
```
Hooks: uv-lock validation, ruff lint (--fix), ruff format.

### Tests
```bash
uv run pytest                                    # All tests (including GPU)
uv run pytest --strict-markers -m "not manual"   # CI mode (no GPU tests)
uv run pytest src/openpi/shared/normalize_test.py # Single test file
```
Tests are co-located with source using `_test.py` suffix. `manual` marker = requires GPU. `conftest.py` auto-detects GPU; sets `JAX_PLATFORMS=cpu` if none found.

### Training (JAX)
```bash
# Compute normalization stats (required before training)
uv run scripts/compute_norm_stats.py --config-name <config>

# Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name> --overwrite

# Resume training
uv run scripts/train.py <config> --exp-name=<name> --resume

# LoRA fine-tuning (lower memory)
uv run scripts/train.py <config>_low_mem_finetune --exp-name=<name>
```

### Training (PyTorch)
```bash
# Single GPU
uv run scripts/train_pytorch.py <config> --exp_name <name>

# Multi-GPU DDP
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<n> scripts/train_pytorch.py <config> --exp_name <name>
```
Requires patching transformers: `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`

### Inference / Serving
```bash
# Serve policy via WebSocket (port 8000)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_path>

# Quick serve with default checkpoint
uv run scripts/serve_policy.py --env=[DROID|ALOHA|LIBERO]
```

### JAX to PyTorch Conversion
```bash
uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir <jax_ckpt> --config_name <config> --output_path <output>
```

## Architecture

### Package Structure (`src/openpi/`)
- **models/** — JAX/Flax model definitions: pi0.py (flow matching), pi0_fast.py (autoregressive), gemma.py (LLM backbone), siglip.py (vision encoder), tokenizer.py, lora.py
- **models_pytorch/** — PyTorch equivalents: pi0_pytorch.py, gemma_pytorch.py, plus patched HuggingFace transformers in `transformers_replace/`
- **policies/** — Robot-specific input/output transforms: aloha_policy.py (14-dim dual arm), droid_policy.py (8-dim Franka), libero_policy.py (7-dim Panda). policy.py wraps model + transforms into `Policy.infer()`. policy_config.py has `create_trained_policy()` factory.
- **training/** — config.py (all TrainConfig/DataConfig definitions, ~25 named configs), data_loader.py (TorchDataLoader, RLDSDataLoader), checkpoints.py (Orbax), optimizer.py (AdamW, cosine schedule), weight_loaders.py, sharding.py (FSDP)
- **transforms.py** — Data pipeline: Normalize/Unnormalize, ResizeImages, DeltaActions, TokenizePrompt, RepackTransform, PadStatesAndActions
- **serving/** — websocket_policy_server.py (async WebSocket server)
- **shared/** — normalize.py (NormStats, RunningStats), download.py, image_tools.py

### Data Flow
Training data → LeRobot dataset format → `DataConfig` repack/transforms → Normalize (z-score for pi0, quantile for pi0.5/pi0-FAST) → Model transforms (resize to 224x224, tokenize prompt, pad) → Model

Inference: observation dict → Policy.infer() → input transforms → model.sample_actions() → output transforms → action chunk [action_horizon, action_dim]

### Key Config Pattern
All configs are defined programmatically in `src/openpi/training/config.py`. Named configs are registered in `_CONFIGS` list and accessed via `get_config("name")`. Data configs inherit from `DataConfigFactory` with robot-specific `repack_transforms`, `data_transforms`, `model_transforms`.

### Checkpoint Format
Orbax format: `checkpoints/<config>/<exp_name>/<step>/` contains `params/` (OCDBT), `train_state/`, `assets/` (norm_stats.json). Checkpoints are auto-downloaded from GCS (`gs://openpi-assets/`) and cached in `~/.cache/openpi`.

### Client-Server Architecture
Policy server (`serve_policy.py`) runs model on GPU machine. Client (`packages/openpi-client/`) sends msgpack-encoded observations via WebSocket, receives actions. Client has minimal dependencies (no JAX/PyTorch needed).

## Memory Requirements
- Inference: >8 GB VRAM
- LoRA fine-tuning: >22.5 GB
- Full fine-tuning: >70 GB
Use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` and `--fsdp-devices <n>` for memory optimization.

## Important: `uv run` vs `python`

`uv run` reinstalls dependencies from `uv.lock` before executing, which **overwrites** any packages installed via `pip`. This causes repeated dependency conflicts (e.g., protobuf downgraded from 6.x to 4.x).

**Rule:** After initial `uv sync`, always use `python` directly — never `uv run`:
```bash
# Correct
python scripts/train.py pi05_libero --exp-name=my_exp --overwrite

# Wrong — will break protobuf and other pip-installed packages
uv run scripts/train.py pi05_libero --exp-name=my_exp --overwrite
```

Similarly, use `python -m pip` instead of bare `pip` to ensure packages install into the venv:
```bash
python -m pip install <package>
```

## Enabling LoRA for Single-GPU Fine-Tuning

The default `pi05_libero` config uses `batch_size=256` and full fine-tuning (~50 GB), which exceeds single-GPU memory. To enable LoRA, edit `src/openpi/training/config.py`:

```python
# In the pi05_libero TrainConfig, change:
model=pi0_config.Pi0Config(
    pi05=True, action_horizon=10, discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",        # add this
    action_expert_variant="gemma_300m_lora",   # add this
),
batch_size=4,                                  # reduce from 256
ema_decay=None,                                # disable EMA for LoRA
freeze_filter=pi0_config.Pi0Config(            # add this block
    pi05=True, action_horizon=10, discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter(),
```

LoRA reduces memory from ~50 GB to ~20 GB. With `batch_size=4` on A6000 (48 GB), training fits comfortably.

## Network / Proxy

HuggingFace downloads often fail in restricted networks. Set proxy before running:
```bash
export https_proxy=http://127.0.0.1:<port>
export http_proxy=http://127.0.0.1:<port>
```

## Git: What NOT to Commit

These directories are large and should be in `.gitignore`:
```
checkpoints/
droid_examples/
wandb/
outputs/
.cache/
*.tfrecord
```
Checkpoints can be regenerated from base weights via training. Use `git rm -r --cached <dir>` to untrack already-committed files.

## Troubleshooting
- `uv sync` fails: `rm -rf .venv && uv sync`
- GPU OOM: set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, use FSDP, reduce batch_size, use LoRA configs
- Missing norm stats: run `compute_norm_stats.py` before training
- Diverging loss: check norm_stats.json for extreme q01/q99/std values
- CUDA conflicts: system CUDA libs can conflict with uv-managed ones; consider uninstalling system CUDA
- Protobuf conflict (`cannot import name 'runtime_version'`): `python -m pip install "protobuf>=6.31.1,<7.0.0" "flatbuffers>=25.9.23" "ml_dtypes>=0.5.1,<1.0.0"`
- `uv run` breaks pip packages: stop using `uv run`, use `python` directly (see above)
- `pip` installs to wrong location: use `python -m pip` instead of bare `pip` (system pip may shadow venv pip)
- `ensurepip` needed: uv venvs don't include pip by default; run `python -m ensurepip && python -m pip install --upgrade pip` if `python -m pip` fails
