"""Train YOLO-FSOD with Cosine Classifier (± Prototype / Florence-2 Adaptation).

This script handles cosine-classifier ablation experiments:
- Base pretrain: standard YOLO11s (same as baseline)
- Finetune (Exp 1): cosine head with random init
- Finetune + Prototype (Exp 2): cosine head initialized with class prototypes
  extracted from support set using the base-pretrained backbone
- Finetune + Florence-2 (Exp 3): cosine head initialized via Florence-2 text
  encoder → adaptation MLP trained on base class pairs

Usage:
  # Exp 1: Cosine classifier (random init)
  python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage finetune

  # Exp 2: Cosine classifier + Prototype init
  python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage finetune --prototype

  # Exp 3: Cosine classifier + Florence-2 adaptation init
  python scripts/train_fsod.py --config configs/baseline_voc_10shot.yaml --stage finetune --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/
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
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="Initialize cosine head with class prototypes from support set (Exp 2).",
    )
    parser.add_argument(
        "--florence2",
        type=str,
        default="",
        help="Path to Florence-2 model for adaptation init (Exp 3). Empty = disabled.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override finetune epochs (0 = use config value).",
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
    config: dict, output_root: Path, base_weights: Path, model_arch: Path,
    use_prototype: bool = False, florence2_model: str = "",
    epochs_override: int = 0,
) -> Path:
    """Finetune with FSODDetect (cosine classifier) architecture.

    1. Creates model from model_arch YAML (has FSODDetect head)
    2. Loads base_weights — matching layers transfer, cosine head stays random init
    3. (Optional) Initialize cosine head with class prototypes from support set
    4. (Optional) Initialize cosine head via Florence-2 adaptation MLP
    5. Trains on novel-only data
    """
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / "voc_fsod_finetune.yaml"
    novel_classes = config["novel_classes"]

    finetune_epochs = epochs_override if epochs_override > 0 else int(config["epochs"]["finetune"])

    if florence2_model and use_prototype:
        run_name = "novel_finetune_cosine_fused"
    elif florence2_model:
        run_name = "novel_finetune_cosine_florence2"
    elif use_prototype:
        run_name = "novel_finetune_cosine_proto"
    else:
        run_name = "novel_finetune_cosine"

    if epochs_override > 0:
        run_name += f"_ep{epochs_override}"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    if use_prototype and not florence2_model:
        from fsod.voc import VOC_CLASSES
        from fsod.modules.prototype import extract_prototypes, init_cosine_head_with_prototypes

        print("Extracting class prototypes from support set...")
        prototypes = extract_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            novel_classes=novel_classes,
            all_classes=VOC_CLASSES,
            imgsz=int(config["image_size"]),
            device=f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"],
        )
        print("Initializing cosine head with prototypes...")
        init_cosine_head_with_prototypes(
            model=model,
            prototypes=prototypes,
            novel_classes=novel_classes,
            all_classes=VOC_CLASSES,
        )

    if florence2_model:
        from fsod.voc import VOC_CLASSES
        from fsod.modules.adaptation import (
            generate_and_encode_descriptions,
            extract_base_prototypes,
            extract_target_weights,
            train_modulation_network,
            init_cosine_head_modulated,
        )
        from fsod.modules.prototype import extract_prototypes

        device_str = f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"]
        base_classes = [c for c in VOC_CLASSES if c not in novel_classes]

        print("Step 1/6: Generating descriptions for base classes (base_train)...")
        base_desc = generate_and_encode_descriptions(
            model_path=florence2_model,
            data_root=output_root,
            split_name="base_train",
            target_classes=base_classes,
            all_classes=VOC_CLASSES,
            device=device_str,
        )

        print("Step 2/6: Generating descriptions for novel classes (novel_finetune)...")
        novel_desc = generate_and_encode_descriptions(
            model_path=florence2_model,
            data_root=output_root,
            split_name="novel_finetune",
            target_classes=novel_classes,
            all_classes=VOC_CLASSES,
            device=device_str,
        )
        desc_embs = {**base_desc, **novel_desc}

        print("Step 3/6: Extracting base class visual prototypes...")
        base_protos = extract_base_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            base_classes=base_classes,
            all_classes=VOC_CLASSES,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        print("Step 4/6: Extracting novel class visual prototypes...")
        novel_protos = extract_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            novel_classes=novel_classes,
            all_classes=VOC_CLASSES,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        print("Step 5/6: Training FiLM modulation network on base classes...")
        target_wts = extract_target_weights(
            base_weights=base_weights,
            target_classes=base_classes,
            all_classes=VOC_CLASSES,
        )
        film = train_modulation_network(
            desc_embeddings=desc_embs,
            visual_prototypes=base_protos,
            target_weights=target_wts,
            base_classes=base_classes,
            device=device_str,
        )

        print("Step 6/6: Initializing novel class weights via text-modulated prototypes...")
        init_cosine_head_modulated(
            model=model,
            film=film,
            desc_embeddings=desc_embs,
            visual_prototypes=novel_protos,
            novel_classes=novel_classes,
            all_classes=VOC_CLASSES,
        )

    model.train(
        data=str(data_yaml),
        epochs=finetune_epochs,
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["finetune"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["finetune"]),
        freeze=int(config.get("freeze", {}).get("backbone", 0)),
        project=str(runs_dir),
        name=run_name,
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / run_name / "weights" / "best.pt"
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
        best_path = run_finetune_stage(
            config, output_root, base_weights, model_arch,
            use_prototype=args.prototype, florence2_model=args.florence2,
            epochs_override=args.epochs,
        )
        print(f"Finetune stage checkpoint: {best_path}")
        return

    # stage == "all"
    base_best = run_base_stage(config, output_root)
    finetune_best = run_finetune_stage(
        config, output_root, base_best, model_arch,
        use_prototype=args.prototype, florence2_model=args.florence2,
        epochs_override=args.epochs,
    )
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint: {finetune_best}")


if __name__ == "__main__":
    main()
