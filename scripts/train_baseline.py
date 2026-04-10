from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO FSOD baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "base", "finetune"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Checkpoint used when stage=finetune. If empty, defaults to base_pretrain best.pt.",
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


def get_yaml_prefix(config: dict) -> str:
    if "coco_root" in config:
        return "coco_fsod"
    return "voc_fsod"


def run_base_stage(config: dict, output_root: Path) -> Path:
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_base.yaml"
    project = runs_dir
    name = "base_pretrain"

    model_source = str(resolve_repo_path(config["model"])) if str(config["model"]).endswith((".pt", ".yaml", ".yml")) and "/" in str(config["model"]) else config["model"]
    model = YOLO(model_source)
    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]["base"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["base"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["base"]),
        project=str(project),
        name=name,
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = project / name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Base stage finished but checkpoint not found: {best_path}")
    return best_path


def run_finetune_stage(config: dict, output_root: Path, weights_path: Path) -> Path:
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_finetune.yaml"
    project = runs_dir
    name = "novel_finetune"

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]["finetune"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["finetune"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["finetune"]),
        freeze=int(config.get("freeze", {}).get("backbone", 0)),
        project=str(project),
        name=name,
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = project / name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Finetune stage finished but checkpoint not found: {best_path}")
    return best_path


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])

    if not output_root.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {output_root}. Run scripts/prepare_voc_fewshot.py first."
        )

    if args.stage == "base":
        best_path = run_base_stage(config, output_root)
        print(f"Base stage checkpoint: {best_path}")
        return

    if args.stage == "finetune":
        weights_path = resolve_repo_path(args.weights) if args.weights else resolve_repo_path(config["runs_dir"]) / "base_pretrain" / "weights" / "best.pt"
        best_path = run_finetune_stage(config, output_root, weights_path)
        print(f"Finetune stage checkpoint: {best_path}")
        return

    base_best = run_base_stage(config, output_root)
    finetune_best = run_finetune_stage(config, output_root, base_best)
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint: {finetune_best}")


if __name__ == "__main__":
    main()