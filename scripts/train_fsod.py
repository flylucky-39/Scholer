"""Train YOLO-FSOD with Cosine Classifier.

This script handles the cosine-classifier ablation experiments:
- Base pretrain: standard YOLO11s (same as baseline)
- Finetune: loads base weights into FSODDetect (cosine head) architecture,
  mismatched classification layers are randomly initialized.

Usage:
  # Base pretrain (same as baseline, only run if not already done)
  python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage base

  # Finetune with cosine classifier
  python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage finetune

  # Both stages
  python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure FSODDetect is importable before model parsing
import fsod.modules  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-FSOD with Cosine Classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--stage",
        type=str,
        default="finetune",
        choices=["all", "base", "finetune"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="",
        help="Base pretrain checkpoint for finetune. Defaults to runs/fsod_baseline/base_pretrain/weights/best.pt.",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="configs/yolo11s-fsod.yaml",
        help="Model architecture YAML with FSODDetect head.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def run_base_stage(config: dict, output_root: Path) -> Path:
    """Base pretrain with standard YOLO (same as baseline)."""
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / "voc_fsod_base.yaml"

    model_source = config["model"]
    if str(model_source).endswith((".pt", ".yaml", ".yml")) and "/" in str(model_source):
        model_source = str(resolve_repo_path(model_source))

    model = YOLO(model_source)
    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]["base"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["base"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["base"]),
        project=str(runs_dir),
        name="base_pretrain",
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / "base_pretrain" / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Base stage finished but checkpoint not found: {best_path}")
    return best_path


def run_finetune_stage(
    config: dict, output_root: Path, base_weights: Path, model_arch: Path
) -> Path:
    """Finetune with FSODDetect (cosine classifier) architecture.

    1. Creates model from model_arch YAML (has FSODDetect head)
    2. Loads base_weights — matching layers transfer, cosine head stays random init
    3. Trains on novel-only data
    """
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / "voc_fsod_finetune.yaml"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]["finetune"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["finetune"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["finetune"]),
        freeze=int(config.get("freeze", {}).get("backbone", 0)),
        project=str(runs_dir),
        name="novel_finetune_cosine",
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / "novel_finetune_cosine" / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Finetune stage finished but checkpoint not found: {best_path}")
    return best_path


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])

    if not output_root.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {output_root}. Run scripts/prepare_voc_fewshot.py first."
        )

    model_arch = resolve_repo_path(args.model_arch)

    if args.stage == "base":
        best_path = run_base_stage(config, output_root)
        print(f"Base stage checkpoint: {best_path}")
        return

    if args.stage == "finetune":
        base_weights = (
            resolve_repo_path(args.base_weights)
            if args.base_weights
            else runs_dir / "base_pretrain" / "weights" / "best.pt"
        )
        if not base_weights.exists():
            raise FileNotFoundError(
                f"Base pretrain weights not found: {base_weights}. Run --stage base first."
            )
        best_path = run_finetune_stage(config, output_root, base_weights, model_arch)
        print(f"Finetune stage checkpoint: {best_path}")
        return

    # stage == "all"
    base_best = run_base_stage(config, output_root)
    finetune_best = run_finetune_stage(config, output_root, base_best, model_arch)
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint (cosine): {finetune_best}")


if __name__ == "__main__":
    main()
