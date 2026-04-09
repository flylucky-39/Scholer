"""Evaluate YOLO-FSOD (cosine classifier) model.

Reports:
  - Overall mAP@50, mAP@50:95
  - Per-class AP@50, AP@50:95
  - Separate base vs novel class summaries

Usage:
  # Evaluate cosine finetune (all 20 classes)
  python scripts/eval_fsod.py --config configs/cosine_voc_10shot.yaml

  # Evaluate with specific weights
  python scripts/eval_fsod.py --config configs/cosine_voc_10shot.yaml \
      --weights runs/fsod_cosine/novel_finetune_cosine/weights/best.pt

  # Evaluate baseline for comparison
  python scripts/eval_fsod.py --config configs/baseline_voc_10shot.yaml \
      --weights runs/fsod_baseline/novel_finetune/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure FSODDetect is loadable (needed when loading .pt with FSODDetect head)
import fsod.modules  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-FSOD model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config yaml.")
    parser.add_argument("--weights", type=str, default="",
                        help="Checkpoint to evaluate. If empty, auto-detect from runs_dir.")
    parser.add_argument("--run-name", type=str, default="",
                        help="Run name under runs_dir to evaluate (e.g. novel_finetune_cosine).")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def find_best_weights(runs_dir: Path, run_name: str = "") -> Path:
    """Auto-detect best.pt from runs_dir, trying common run names."""
    if run_name:
        candidates = [run_name]
    else:
        candidates = [
            "novel_finetune_cosine",
            "novel_finetune_cosine_proto",
            "novel_finetune_cosine_florence2",
            "novel_finetune_cosine_fused",
            "novel_finetune",
        ]

    for name in candidates:
        best = runs_dir / name / "weights" / "best.pt"
        if best.exists():
            return best

    raise FileNotFoundError(
        f"No best.pt found in {runs_dir}. "
        f"Tried: {', '.join(candidates)}. Use --weights to specify path."
    )


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])
    novel_classes = config["novel_classes"]

    # Determine weights path
    if args.weights:
        weights_path = resolve_repo_path(args.weights)
    else:
        weights_path = find_best_weights(runs_dir, args.run_name)

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Weights: {weights_path}")

    # Use the all-class eval yaml
    data_yaml = output_root / "voc_fsod_eval_all.yaml"
    if not data_yaml.exists():
        # Fallback to finetune yaml (also has all 20 classes)
        data_yaml = output_root / "voc_fsod_finetune.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No eval yaml found in {output_root}")

    print(f"Data: {data_yaml}")

    # Load and evaluate
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(config["image_size"]),
        device=config["device"],
        project=str(runs_dir),
        name="eval",
        exist_ok=True,
        plots=True,
    )

    # Get class names from metrics
    class_names = list(metrics.names.values()) if hasattr(metrics, "names") else []

    # Build per-class AP table
    per_class_ap50 = metrics.box.ap50 if hasattr(metrics.box, "ap50") else []
    per_class_ap = metrics.box.ap if hasattr(metrics.box, "ap") else []

    novel_set = set(novel_classes)

    # Print header
    print("\n" + "=" * 70)
    print(f"{'Class':<20} {'Type':<8} {'AP@50':>10} {'AP@50:95':>10}")
    print("-" * 70)

    base_ap50_list = []
    novel_ap50_list = []
    base_ap_list = []
    novel_ap_list = []

    for i, name in enumerate(class_names):
        cls_type = "novel" if name in novel_set else "base"
        ap50_val = float(per_class_ap50[i]) if i < len(per_class_ap50) else 0.0
        ap_val = float(per_class_ap[i]) if i < len(per_class_ap) else 0.0

        print(f"  {name:<18} {cls_type:<8} {ap50_val:>10.4f} {ap_val:>10.4f}")

        if name in novel_set:
            novel_ap50_list.append(ap50_val)
            novel_ap_list.append(ap_val)
        else:
            base_ap50_list.append(ap50_val)
            base_ap_list.append(ap_val)

    print("-" * 70)

    # Compute group averages
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    base_map50 = safe_mean(base_ap50_list)
    base_map = safe_mean(base_ap_list)
    novel_map50 = safe_mean(novel_ap50_list)
    novel_map = safe_mean(novel_ap_list)
    all_map50 = float(metrics.box.map50)
    all_map = float(metrics.box.map)
    all_map75 = float(metrics.box.map75)

    print(f"  {'Base (15 cls)':<18} {'avg':<8} {base_map50:>10.4f} {base_map:>10.4f}")
    print(f"  {'Novel (5 cls)':<18} {'avg':<8} {novel_map50:>10.4f} {novel_map:>10.4f}")
    print(f"  {'All (20 cls)':<18} {'avg':<8} {all_map50:>10.4f} {all_map:>10.4f}")
    print("=" * 70)

    # Save summary
    summary = {
        "weights": str(weights_path),
        "data_yaml": str(data_yaml),
        "overall": {
            "mAP50": all_map50,
            "mAP50_95": all_map,
            "mAP75": all_map75,
        },
        "base_classes": {
            "mAP50": base_map50,
            "mAP50_95": base_map,
            "classes": {name: {"AP50": float(per_class_ap50[i]), "AP50_95": float(per_class_ap[i])}
                        for i, name in enumerate(class_names) if name not in novel_set and i < len(per_class_ap50)},
        },
        "novel_classes": {
            "mAP50": novel_map50,
            "mAP50_95": novel_map,
            "classes": {name: {"AP50": float(per_class_ap50[i]), "AP50_95": float(per_class_ap[i])}
                        for i, name in enumerate(class_names) if name in novel_set and i < len(per_class_ap50)},
        },
    }

    # Determine run name from weights path for filename
    run_label = weights_path.parent.parent.name if weights_path.parent.name == "weights" else "unknown"
    summary_path = runs_dir / f"eval_{run_label}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
