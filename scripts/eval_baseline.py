from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO FSOD baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Checkpoint to evaluate. If empty, defaults to novel_finetune best.pt.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    output_root = Path(config["output_root"]).resolve()
    data_yaml = output_root / "voc_fsod_finetune.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    default_weights = Path(config["runs_dir"]) / "novel_finetune" / "weights" / "best.pt"
    weights_path = Path(args.weights) if args.weights else default_weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(config["image_size"]),
        device=config["device"],
        project=str(Path(config["runs_dir"])),
        name="eval",
        exist_ok=True,
        plots=False,
    )

    summary = {
        "weights": str(weights_path),
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        "map75": float(metrics.box.map75),
    }

    summary_path = Path(config["runs_dir"]) / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()