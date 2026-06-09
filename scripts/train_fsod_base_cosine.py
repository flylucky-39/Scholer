"""Base pretrain with FSODDetect (pure cosine head, no objectness).

Trains the cosine classifier head together with backbone on abundant base class
data, producing a checkpoint where both the feature extractor and the cosine
classifier are well-trained. This checkpoint is useful for 1-shot FSOD where
freezing a well-trained head + backbone prevents overfitting from only 5 images.

Architecture: yolo11s-fsod.yaml → FSODDetect head (CosineConv2d, no obj_pred)
Initial weights: yolo11s.pt — backbone/neck/box layers transfer, cosine head random init
Data: base classes (e.g., 15 VOC base classes)
Output: runs/fsod_baseline/base_pretrain_cosine/weights/best.pt

Usage:
  # Standard base pretrain
  python scripts/train_fsod_base_cosine.py --config configs/baseline_voc_10shot.yaml

  # With custom epochs
  python scripts/train_fsod_base_cosine.py --config configs/baseline_voc_10shot.yaml --epochs 150

  # Then use the checkpoint for 1-shot freeze-head:
  python scripts/train_fsod.py --config configs/baseline_voc_1shot.yaml --stage finetune \
      --prototype --base-weights runs/fsod_baseline/base_pretrain_cosine/weights/best.pt \
      --model-arch configs/yolo11s-fsod.yaml --freeze-head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from ultralytics import YOLO

# Import FSODDetect so it's available for model creation
import fsod.modules  # noqa: F401

# Explicitly register FSODDetect in ultralytics.nn.tasks for YAML parsing.
# ultralytics' parse_model() resolves module names via globals() of nn.tasks,
# so FSODDetect must be in that namespace.
import ultralytics.nn.tasks as _tasks
from fsod.modules.cosine_head import FSODDetect

_tasks.FSODDetect = FSODDetect

from scripts.train_fsod import (
    PROJECT_ROOT,
    load_config,
    resolve_repo_path,
    get_yaml_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base pretrain with FSODDetect (pure cosine head)."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config yaml (e.g., configs/baseline_voc_10shot.yaml).",
    )
    parser.add_argument(
        "--model-arch", type=str, default="configs/yolo11s-fsod.yaml",
        help="Model architecture YAML with FSODDetect head.",
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11s.pt",
        help="Pretrained weights for initialization (backbone/neck/box only; head skipped automatically).",
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Override base epochs (0 = use config value).",
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="Override batch size (0 = use config value).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])

    if not output_root.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {output_root}. "
            f"Run scripts/prepare_voc_fewshot.py first."
        )

    model_arch = resolve_repo_path(args.model_arch)
    if not model_arch.exists():
        raise FileNotFoundError(f"Model architecture not found: {model_arch}")

    data_yaml = output_root / f"{get_yaml_prefix(config)}_base.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    base_epochs = args.epochs if args.epochs > 0 else int(config["epochs"]["base"])
    base_batch = args.batch if args.batch > 0 else int(config["batch_size"]["base"])
    run_name = "base_pretrain_cosine"

    # --- Create model with FSODDetect (cosine classifier head) ---
    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    # --- Load pretrained weights (backbone/neck/box only) ---
    # model.load() automatically skips mismatched keys, so FSODDetect's CosineConv2d
    # layers (which have different shapes from standard Detect Conv2d) stay random init.
    if args.weights:
        # Built-in ultralytics names like "yolo11s.pt" have no "/" and load automatically
        if "/" in args.weights:
            weights_path = str(resolve_repo_path(args.weights))
        else:
            weights_path = args.weights
        print(f"Loading initial weights: {weights_path}")
        model.load(weights_path)

    # --- Train on base classes ---
    print(f"Training FSODDetect (cosine head) on base classes for {base_epochs} epochs...")
    model.train(
        data=str(data_yaml),
        epochs=base_epochs,
        imgsz=int(config["image_size"]),
        batch=base_batch,
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["base"]),
        freeze=int(config.get("freeze", {}).get("backbone", 0)),
        project=str(runs_dir),
        name=run_name,
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / run_name / "weights" / "best.pt"
    if best_path.exists():
        print(f"\n✓ Base pretrain (cosine) checkpoint: {best_path}")
    else:
        print(f"\nTraining finished but checkpoint not found at {best_path}")


if __name__ == "__main__":
    main()
