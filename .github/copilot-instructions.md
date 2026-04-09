# Project Guidelines

## Overview

Few-Shot Object Detection research project (FSOD_LLM): YOLO11s + Cosine Classifier + Prototype + lightweight VLM guidance on PASCAL VOC.

## Architecture

- `fsod/modules/` — Custom detection modules (CosineConv2d, FSODDetect, Prototype, AdaptationMLP)
- `scripts/` — CLI entry points (train, eval, data prep). All share `resolve_repo_path()` + `PROJECT_ROOT` pattern
- `configs/` — YAML experiment configs. All hyperparams are config-driven, never hardcoded
- `third_party/ultralytics/` — Vendored Ultralytics with FSODDetect lazy-registered in `nn/tasks.py`
- `docs/` — Operational runbooks

## Code Conventions

- **Path resolution**: Always use `resolve_repo_path(raw_path)` for any path from config or CLI. Supports `~` expansion and repo-relative resolution.
- **FSODDetect registration**: Lazy-imported in `third_party/ultralytics/ultralytics/nn/tasks.py` via `from fsod.modules.cosine_head import FSODDetect`. Scripts must `import fsod.modules` before loading YAML-defined models.
- **Config structure**: Nested dicts for stage-specific params (`epochs.base`, `epochs.finetune`, `batch_size.base`, etc.). See `configs/baseline_voc_10shot.yaml` as canonical example.
- **Training stages**: Always two-stage — base pretrain (15 classes) → novel finetune (10-shot). Checkpoint flows from stage 1 to stage 2 via `--weights` or auto-detection from `runs_dir`.
- **Seed**: Use `3407` as default seed for reproducibility.

## Build and Test

```bash
# Install dependencies
pip install -r requirements.txt

# Data preparation
python scripts/prepare_voc_fewshot.py --config configs/baseline_voc_10shot.yaml

# Baseline training
python scripts/train_baseline.py --config configs/baseline_voc_10shot.yaml --stage all

# Cosine classifier training
python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage finetune

# Evaluation
python scripts/eval_baseline.py --config configs/baseline_voc_10shot.yaml
```

## Server Environment

- Project root: `~/epfs/07_FSOD_LLM/fsod`
- VOC data: `~/epfs/07_FSOD_LLM/datasets/VOCdevkit`
- GPU: NVIDIA H20 (`device: 0`)
- Conda env: `FSOD_LLM` (Python 3.9+)
- GitLab: `https://gitlab.goertek.com/ai_team_j01/fsod.git`, branch `FSOD_LLM`
- Sync: `git fetch origin && git reset --hard origin/FSOD_LLM` (not `git pull`)

## Pitfalls

- **Do NOT modify** files under `third_party/ultralytics/` unless adding new head registration
- New scripts must add `PROJECT_ROOT` to `sys.path` before importing `fsod.*`
- Config paths should stay relative (`./data/...`, `./runs/...`) for portability; use `~` only for `voc_root`
- The `runs/` and `data/voc_fsod_*` directories are gitignored — never commit weights or generated data
