"""Train YOLO-FSOD with ObjPretrain head (cosine classifier + pretrained objectness).

Two-stage pipeline:

  Stage 1 — Base pretrain:
    Train FSODDetectWithObjPretrain on base classes (cosine cls + obj_pred + box).
    Produces weights with a strong class-agnostic objectness branch.

  Stage 2 — Few-shot finetune:
    Load base weights, freeze obj_pred, replace cls with cosine prototypes.
    Inference: score = sigmoid(obj) * sigmoid(cls)

Usage:
  # Base pretrain
  python scripts/train_fsod_obj_pretrain.py --config configs/baseline_voc_10shot.yaml --stage base

  # Finetune (with prototype init + frozen obj)
  python scripts/train_fsod_obj_pretrain.py --config configs/baseline_voc_10shot.yaml --stage finetune

  # End-to-end
  python scripts/train_fsod_obj_pretrain.py --config configs/baseline_voc_10shot.yaml --stage all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from ultralytics import YOLO

import fsod.modules  # noqa: F401
import fsod.modules.obj_pretrain_head  # noqa: F401 — registers monkey-patch
from fsod.modules.obj_pretrain_head import FSODDetectWithObjPretrain, FSODObjPretrainLoss

# Reuse helper functions from train_fsod
from scripts.train_fsod import (
    PROJECT_ROOT,
    load_config,
    resolve_repo_path,
    get_all_classes,
    get_yaml_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO-FSOD with pretrained objectness branch."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    parser.add_argument(
        "--stage", type=str, default="all", choices=["all", "base", "finetune"],
        help="Training stage to run.",
    )
    parser.add_argument(
        "--base-weights", type=str, default="",
        help="Base pretrain checkpoint for finetune.",
    )
    parser.add_argument(
        "--model-arch", type=str, default="configs/yolo11s-fsod-objpretrain.yaml",
        help="Model architecture YAML with FSODDetectWithObjPretrain head.",
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Override finetune epochs (0 = use config value).",
    )
    parser.add_argument(
        "--unfreeze-obj", action="store_true",
        help="Unfreeze obj_pred branch during finetune (default: freeze).",
    )
    return parser.parse_args()


def run_base_stage_with_obj(config: dict, output_root: Path, model_arch: Path) -> Path:
    """Base pretrain: train FSODDetectWithObjPretrain on base classes.

    All parameters (backbone, neck, box, cosine cls, obj_pred) are trained.
    Loss includes box, cls, dfl, and objectness BCE.
    """
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_base.yaml"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    # Use a unique run name for this variant
    run_name = "base_pretrain_obj"

    # --- Callback: set custom loss with objectness term ---
    def _on_pretrain_routine_end(trainer):
        """Set FSODObjPretrainLoss (includes obj BCE) after model setup."""
        if isinstance(trainer.model.model[-1], FSODDetectWithObjPretrain):
            trainer.model.criterion = FSODObjPretrainLoss(trainer.model)
            trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "obj_loss")
            print("Using FSODObjPretrainLoss with class-agnostic objectness")

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)

    model.train(
        data=str(data_yaml),
        epochs=int(config["epochs"]["base"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["base"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["base"]),
        project=str(runs_dir),
        name=run_name,
        seed=int(config["seed"]),
        exist_ok=True,
    )

    best_path = runs_dir / run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Base stage finished but checkpoint not found: {best_path}")
    return best_path


def run_finetune_stage_with_obj_pretrain(
    config: dict, output_root: Path, base_weights: Path, model_arch: Path,
    epochs_override: int = 0, unfreeze_obj: bool = False,
) -> Path:
    """Finetune: load base weights, freeze obj_pred, init cosine with prototypes.

    1. Creates model from model_arch YAML (FSODDetectWithObjPretrain)
    2. Loads base_weights — obj_pred transferred, cosine cls stays random init
    3. Freezes obj_pred branch (preserves base objectness knowledge)
    4. Initializes cosine classifier with class prototypes from support set
    5. Trains on novel data
    """
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_finetune.yaml"
    novel_classes = config["novel_classes"]
    all_classes = get_all_classes(config)

    finetune_epochs = epochs_override if epochs_override > 0 else int(config["epochs"]["finetune"])

    run_name = "objpretrain_finetune"
    if unfreeze_obj:
        run_name += "_unfrozen"
    if epochs_override > 0:
        run_name += f"_ep{epochs_override}"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    print(f"Loading base pretrain weights (incl. obj_pred): {base_weights}")
    model.load(str(base_weights))

    # --- Pre-extract prototypes from support set ---
    from fsod.modules.prototype import extract_prototypes, init_cosine_head_with_prototypes

    print("Extracting class prototypes from support set...")
    prototypes = extract_prototypes(
        base_weights=base_weights,
        data_root=output_root,
        novel_classes=novel_classes,
        all_classes=all_classes,
        imgsz=int(config["image_size"]),
        device=f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"],
    )

    # --- Callback: optionally freeze obj_pred + init prototypes + set loss ---
    def _on_pretrain_routine_end(trainer):
        """Optionally freeze objectness, inject prototype weights, set custom loss, sync EMA."""
        detect = trainer.model.model[-1]

        # 1. Optionally freeze obj_pred (keep base-trained objectness knowledge)
        if isinstance(detect, FSODDetectWithObjPretrain):
            if unfreeze_obj:
                detect.unfreeze_obj_pred()
                print("obj_pred branch kept TRAINABLE during finetune")
            else:
                detect.freeze_obj_pred()
                print("Frozen obj_pred branch (preserving base objectness prior)")

        # 2. Inject class prototypes into cosine classifier
        class _FakeYOLO:
            def __init__(self, det_model):
                self.model = det_model

        print("Injecting prototype weights into cosine classifier...")
        init_cosine_head_with_prototypes(
            model=_FakeYOLO(trainer.model),
            prototypes=prototypes,
            novel_classes=novel_classes,
            all_classes=all_classes,
        )

        # 3. Set custom loss with objectness term
        if isinstance(trainer.model.model[-1], FSODDetectWithObjPretrain):
            trainer.model.criterion = FSODObjPretrainLoss(trainer.model)
            trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "obj_loss")
            print("Using FSODObjPretrainLoss (obj_pred frozen, loss computed but not updated)")

        # 4. Sync injected weights (prototypes) to EMA
        if getattr(trainer, "ema", None) is not None:
            model_sd = trainer.model.state_dict()
            for k, v in trainer.ema.ema.state_dict().items():
                if k in model_sd:
                    v.copy_(model_sd[k])
            print("Synced injected prototype weights to EMA model")

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)

    model.train(
        data=str(data_yaml),
        epochs=finetune_epochs,
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]["finetune"]),
        workers=int(config["workers"]),
        device=config["device"],
        lr0=float(config["lr0"]["finetune"]),
        freeze=int(config.get("freeze", {}).get("backbone", 0)),
        patience=int(config.get("patience", {}).get("finetune", 30)),
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
        best_path = run_base_stage_with_obj(config, output_root, model_arch)
        print(f"Base stage checkpoint: {best_path}")
        return

    if args.stage == "finetune":
        base_weights = (
            resolve_repo_path(args.base_weights)
            if args.base_weights
            else runs_dir / "base_pretrain_obj" / "weights" / "best.pt"
        )
        if not base_weights.exists():
            raise FileNotFoundError(
                f"Base pretrain weights not found: {base_weights}. Run --stage base first."
            )
        best_path = run_finetune_stage_with_obj_pretrain(
            config, output_root, base_weights, model_arch,
            epochs_override=args.epochs, unfreeze_obj=args.unfreeze_obj,
        )
        print(f"Finetune stage checkpoint: {best_path}")
        return

    # stage == "all"
    base_best = run_base_stage_with_obj(config, output_root, model_arch)
    finetune_best = run_finetune_stage_with_obj_pretrain(
        config, output_root, base_best, model_arch,
        epochs_override=args.epochs, unfreeze_obj=args.unfreeze_obj,
    )
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint: {finetune_best}")


if __name__ == "__main__":
    main()
