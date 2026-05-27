"""Train YOLO-FSOD with Cross-Attention Prototype (CAP) + scale regularisation.

Combines two techniques:
  1. Cross-Attention Prototype (v0.2.0): bidirectional cross-attention between
     spatial visual features and VLM text embeddings for stronger prototype init.
  2. Scale Regularisation (v0.1.0): L2 penalty on CosineConv2d temperatures.

Usage:
    python scripts/train_fsod_cap.py \\
        --config configs/baseline_voc_10shot_scalereg.yaml \\
        --stage finetune --reg-weight 0.05 \\
        --base-weights base_pretrain/weights/best.pt \\
        --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_THIRD_PARTY_ULTRALYTICS = str(PROJECT_ROOT / "third_party" / "ultralytics")
if _THIRD_PARTY_ULTRALYTICS not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_ULTRALYTICS)

from ultralytics import YOLO

import fsod.modules  # noqa: F401 (registers FSODDetect)
from fsod.modules.scale_regularization import ScaleRegDetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO-FSOD with Cross-Attention Prototype + scale regularisation."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument(
        "--stage", type=str, default="finetune", choices=["all", "base", "finetune"],
    )
    parser.add_argument(
        "--base-weights", type=str, default="",
        help="Base pretrain checkpoint. Defaults to runs_dir/base_pretrain/weights/best.pt.",
    )
    parser.add_argument(
        "--model-arch", type=str, default="configs/yolo11s-fsod.yaml",
    )
    parser.add_argument(
        "--florence2", type=str, default="",
        help="Path to Florence-2 model for text embeddings.",
    )
    parser.add_argument("--epochs", type=int, default=0, help="Override finetune epochs.")
    parser.add_argument("--reg-weight", type=float, default=0.05, help="Scale regularisation weight.")
    parser.add_argument(
        "--roi-size", type=int, default=3,
        help="RoIAlign output size for spatial features (default: 3 for 3x3).",
    )
    parser.add_argument(
        "--cap-epochs", type=int, default=300,
        help="Training epochs for CAP on base classes.",
    )
    parser.add_argument(
        "--cap-lr", type=float, default=1e-3,
        help="Learning rate for CAP training.",
    )
    parser.add_argument(
        "--cap-hidden", type=int, default=256,
        help="Hidden dimension for CAP cross-attention.",
    )
    parser.add_argument(
        "--cap-nheads", type=int, default=4,
        help="Number of attention heads in CAP.",
    )
    parser.add_argument(
        "--fusion-mode", type=str, default="learnable",
        choices=["learnable", "fixed", "init_only"],
        help="Prior fusion mode for CosineConv2d.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Blend alpha for visual prototype prior.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def get_all_classes(config: dict) -> list[str]:
    if "all_classes" in config:
        return config["all_classes"]
    if "coco_root" in config:
        from fsod.coco import COCO_CLASSES
        return list(COCO_CLASSES)
    from fsod.voc import VOC_CLASSES
    return list(VOC_CLASSES)


def get_yaml_prefix(config: dict) -> str:
    if "yaml_prefix" in config:
        return config["yaml_prefix"]
    if "coco_root" in config:
        return "coco_fsod"
    return "voc_fsod"


def run_base_stage(config: dict, output_root: Path) -> Path:
    """Base pretrain — identical to original."""
    runs_dir = resolve_repo_path(config["runs_dir"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_base.yaml"

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
        raise FileNotFoundError(f"Base stage checkpoint not found: {best_path}")
    return best_path


def run_finetune_stage(
    config: dict,
    output_root: Path,
    base_weights: Path,
    model_arch: Path,
    reg_weight: float,
    florence2_model: str = "",
    roi_size: int = 3,
    cap_epochs: int = 300,
    cap_lr: float = 1e-3,
    cap_hidden: int = 256,
    cap_nheads: int = 4,
    fusion_mode: str = "learnable",
    alpha: float = 0.5,
    epochs_override: int = 0,
) -> Path:
    """Finetune with CAP prototype init + scale regularisation."""
    runs_dir = resolve_repo_path(config["runs_dir"])
    base_data_root = output_root
    if "base_data_root" in config:
        base_data_root = resolve_repo_path(config["base_data_root"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_finetune.yaml"
    novel_classes = config["novel_classes"]
    all_classes = get_all_classes(config)

    finetune_epochs = epochs_override if epochs_override > 0 else int(config["epochs"]["finetune"])

    # Build run name
    run_name = "novel_finetune_cosine_cap_scalereg"
    if fusion_mode != "learnable":
        run_name += f"_{fusion_mode}"
    if epochs_override > 0:
        run_name += f"_ep{epochs_override}"

    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))

    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    # Store init data for callback
    _cap_init_fn = None

    if florence2_model:
        from fsod.modules.cross_attention_prototype import (
            CrossAttentionPrototype,
            extract_spatial_prototypes,
            extract_spatial_prototypes_base,
            extract_target_weights_for_cap,
            train_cap_on_base,
            init_cosine_head_with_cap,
            get_text_embeddings_for_classes,
        )

        device_str = (
            f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"]
        )
        base_classes = [c for c in all_classes if c not in novel_classes]

        # ── Step 1: Florence-2 text embeddings for ALL classes ──
        print("\n=== Step 1: Extracting Florence-2 text embeddings ===")
        text_embs = get_text_embeddings_for_classes(
            florence2_model, all_classes, device=device_str,
        )

        # ── Step 2: Extract spatial visual features from support set (novel) ──
        print("\n=== Step 2: Extracting novel class spatial features ===")
        novel_spatial = extract_spatial_prototypes(
            base_weights=base_weights,
            data_root=output_root,
            target_classes=novel_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
            roi_size=roi_size,
        )
        for cls_name, feats in novel_spatial.items():
            print(f"  {cls_name}: {len(feats)} spatial features")

        # ── Step 3: Extract spatial features from base_train ──
        print("\n=== Step 3: Extracting base class spatial features ===")
        base_spatial_list = extract_spatial_prototypes_base(
            base_weights=base_weights,
            data_root=base_data_root,
            base_classes=base_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
            roi_size=roi_size,
        )
        for cls_name, feats in base_spatial_list.items():
            print(f"  {cls_name}: {len(feats)} spatial features")

        # Average per-class base spatial features
        from fsod.modules.cross_attention_prototype import _average_spatial_features
        base_spatial_avg = _average_spatial_features(base_spatial_list)

        # ── Step 4: Train CAP per scale ──
        print("\n=== Step 4: Training CAP per scale ===")
        detect = model.model.model[-1]
        num_scales = detect.nl
        c3_dim = detect.cv3[0][-1].weight.shape[1]       # visual feature dim (256 for yolo11s)
        text_dim = next(iter(text_embs.values())).shape[0]  # auto-detect

        cap_models = []
        for scale_idx in range(num_scales):
            print(f"\n--- Scale {scale_idx} (P{3+scale_idx}) ---")
            cap = CrossAttentionPrototype(
                visual_dim=c3_dim,
                text_dim=text_dim,
                hidden_dim=cap_hidden,
                n_heads=cap_nheads,
            ).to(device_str)

            # Extract target weights from base-pretrained model
            target_w = extract_target_weights_for_cap(
                base_weights=base_weights,
                target_classes=base_classes,
                all_classes=all_classes,
                scale_idx=scale_idx,
            )

            # Train CAP on base classes
            cap = train_cap_on_base(
                cap=cap,
                text_embeddings=text_embs,
                spatial_features=base_spatial_avg,
                target_weights=target_w,
                base_classes=base_classes,
                lr=cap_lr,
                weight_decay=float(config.get("florence2", {}).get("film_weight_decay", 0.05)),
                epochs=cap_epochs,
                device=device_str,
            )
            cap_models.append(cap.cpu())

        # ── Prepare init function callback ──
        _cap_init_fn = lambda m: init_cosine_head_with_cap(
            model=m,
            cap_models=cap_models,
            text_embeddings=text_embs,
            spatial_prototypes=novel_spatial,
            novel_classes=novel_classes,
            all_classes=all_classes,
            alpha=alpha,
            fusion_mode=fusion_mode,
        )

    # ── Callback: inject CAP weights, then patch loss ──
    def _on_pretrain_routine_end(trainer):
        """Inject CAP prototype weights, then swap loss to ScaleRegDetectionLoss."""
        if _cap_init_fn is not None:
            print("\nInjecting Cross-Attention Prototype weights into rebuilt model...")

            class _FakeYOLO:
                def __init__(self, det_model):
                    self.model = det_model

            _cap_init_fn(_FakeYOLO(trainer.model))

            # Sync injected weights to EMA
            if getattr(trainer, "ema", None) is not None:
                model_sd = trainer.model.state_dict()
                for k, v in trainer.ema.ema.state_dict().items():
                    if k in model_sd:
                        v.copy_(model_sd[k])
                print("Synced injected weights to EMA model")

        # Patch loss with scale regularisation
        print(f"Patching loss with scale regularisation (reg_weight={reg_weight})...")
        trainer.criterion = ScaleRegDetectionLoss(
            trainer.model,
            reg_weight=reg_weight,
        )

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
        raise FileNotFoundError(f"Finetune checkpoint not found: {best_path}")
    return best_path


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
                f"Base pretrain weights not found: {base_weights}. "
                f"Run --stage base first."
            )
        best_path = run_finetune_stage(
            config, output_root, base_weights, model_arch,
            reg_weight=args.reg_weight,
            florence2_model=args.florence2,
            roi_size=args.roi_size,
            cap_epochs=args.cap_epochs,
            cap_lr=args.cap_lr,
            cap_hidden=args.cap_hidden,
            cap_nheads=args.cap_nheads,
            fusion_mode=args.fusion_mode,
            alpha=args.alpha,
            epochs_override=args.epochs,
        )
        print(f"Finetune stage checkpoint: {best_path}")
        return

    # stage == "all"
    base_best = run_base_stage(config, output_root)
    finetune_best = run_finetune_stage(
        config, output_root, base_best, model_arch,
        reg_weight=args.reg_weight,
        florence2_model=args.florence2,
        roi_size=args.roi_size,
        cap_epochs=args.cap_epochs,
        cap_lr=args.cap_lr,
        cap_hidden=args.cap_hidden,
        cap_nheads=args.cap_nheads,
        fusion_mode=args.fusion_mode,
        alpha=args.alpha,
        epochs_override=args.epochs,
    )
    print(f"Base stage checkpoint: {base_best}")
    print(f"Finetune stage checkpoint: {finetune_best}")


if __name__ == "__main__":
    main()
