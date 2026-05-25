"""Train YOLO-FSOD with CV2 frozen + no mosaic + Simple MLP modulation.

Uses Florence-2 BART text encoder + AdaptationMLP (no FiLM) to initialize
cosine classifier weights, then finetunes with cv2 frozen and mosaic disabled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_THIRD_PARTY_ULTRALYTICS = str(PROJECT_ROOT / "third_party" / "ultralytics")
if _THIRD_PARTY_ULTRALYTICS not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_ULTRALYTICS)

from ultralytics import YOLO

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401

import numpy as np
import torch
import cv2 as _cv2

_original_warp_affine = _cv2.warpAffine
_original_warp_perspective = _cv2.warpPerspective

def _warp_affine_fixed(*args, **kwargs):
    args = list(args)
    if len(args) >= 2:
        if not isinstance(args[1], np.ndarray):
            args[1] = np.asarray(args[1], dtype=np.float32)
        if not isinstance(args[0], np.ndarray):
            args[0] = np.asarray(args[0])
    elif 'M' in kwargs:
        if not isinstance(kwargs['M'], np.ndarray):
            kwargs['M'] = np.asarray(kwargs['M'], dtype=np.float32)
    if 'src' in kwargs and not isinstance(kwargs['src'], np.ndarray):
        kwargs['src'] = np.asarray(kwargs['src'])
    return _original_warp_affine(*args, **kwargs)

def _warp_perspective_fixed(*args, **kwargs):
    args = list(args)
    if len(args) >= 2:
        if not isinstance(args[1], np.ndarray):
            args[1] = np.asarray(args[1], dtype=np.float32)
        if not isinstance(args[0], np.ndarray):
            args[0] = np.asarray(args[0])
    elif 'M' in kwargs:
        if not isinstance(kwargs['M'], np.ndarray):
            kwargs['M'] = np.asarray(kwargs['M'], dtype=np.float32)
    if 'src' in kwargs and not isinstance(kwargs['src'], np.ndarray):
        kwargs['src'] = np.asarray(kwargs['src'])
    return _original_warp_perspective(*args, **kwargs)

_cv2.warpAffine = _warp_affine_fixed
_cv2.warpPerspective = _warp_perspective_fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO-FSOD with CV2 frozen + MLP modulation."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-weights", type=str, default="",
                        help="Base pretrain checkpoint.")
    parser.add_argument("--model-arch", type=str, default="configs/yolo11s-fsod.yaml")
    parser.add_argument("--florence2", type=str, default="",
                        help="Path to Florence-2 model.")
    parser.add_argument("--epochs", type=int, default=0)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def freeze_cv2_parameters(model) -> None:
    detect = model.model[-1] if hasattr(model, "model") else model
    if not hasattr(detect, "cv2"):
        print("[WARN] No cv2 found in model, skipping freeze")
        return
    frozen_count = 0
    for i, cv2_module in enumerate(detect.cv2):
        for name, param in cv2_module.named_parameters():
            param.requires_grad = False
            frozen_count += 1
            print(f"  Freezing cv2[{i}].{name}")
    print(f"[CV2 Freeze] Frozen {frozen_count} cv2 parameters")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    output_root = resolve_repo_path(config["output_root"])
    runs_dir = resolve_repo_path(config["runs_dir"])
    base_weights = resolve_repo_path(args.base_weights) if args.base_weights else \
        runs_dir / "base_pretrain" / "weights" / "best.pt"
    model_arch = resolve_repo_path(args.model_arch)
    novel_classes = config["novel_classes"]
    all_classes = get_all_classes(config)
    base_classes = [c for c in all_classes if c not in novel_classes]

    if not output_root.exists():
        raise FileNotFoundError(f"Dataset not found: {output_root}")
    if not base_weights.exists():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")

    device_str = f"cuda:{config['device']}" if str(config["device"]).isdigit() else config["device"]

    run_name = "novel_finetune_cosine_mlp_cv2freeze_nomosaic"

    # ------------------------------------------------------------------
    # Simple MLP preprocessing: text embeddings + train MLP
    # ------------------------------------------------------------------
    if args.florence2:
        from fsod.modules.adaptation import (
            extract_florence2_text_embeddings,
            extract_base_prototypes,
            train_adaptation_mlp,
            init_cosine_head_with_florence2,
        )

        print("Step 1/3: Extracting Florence-2 text embeddings (BART encoder)...")
        text_embs = extract_florence2_text_embeddings(
            model_path=args.florence2,
            class_names=all_classes,
            device=device_str,
        )

        print("Step 2/3: Extracting base class visual prototypes...")
        from fsod.modules.adaptation import extract_base_prototypes
        base_data_root = output_root
        if "base_data_root" in config:
            base_data_root = resolve_repo_path(config["base_data_root"])
        base_protos = extract_base_prototypes(
            base_weights=base_weights,
            data_root=base_data_root,
            base_classes=base_classes,
            all_classes=all_classes,
            imgsz=int(config["image_size"]),
            device=device_str,
        )

        print("Step 3/3: Training adaptation MLP on base classes...")
        mlp = train_adaptation_mlp(
            text_embeddings=text_embs,
            visual_prototypes=base_protos,
            base_classes=base_classes,
            hidden_dim=256,
            lr=1e-3,
            weight_decay=1e-2,
            epochs=1000,
            device=device_str,
        )
    else:
        mlp = None

    # ------------------------------------------------------------------
    # Setup model
    # ------------------------------------------------------------------
    print(f"Creating model from architecture: {model_arch}")
    model = YOLO(str(model_arch))
    print(f"Loading base pretrain weights: {base_weights}")
    model.load(str(base_weights))

    # ------------------------------------------------------------------
    # Callback: init cosine head with MLP, then freeze cv2
    # ------------------------------------------------------------------
    def _on_pretrain_routine_end(trainer):
        if mlp is not None:
            from fsod.modules.adaptation import init_cosine_head_with_florence2
            print("Injecting MLP-modulated weights into rebuilt model...")
            class _FakeYOLO:
                def __init__(self, det_model):
                    self.model = det_model
            init_cosine_head_with_florence2(
                model=_FakeYOLO(trainer.model),
                mlp=mlp,
                text_embeddings=text_embs,
                novel_classes=novel_classes,
                all_classes=all_classes,
            )
            # Sync to EMA
            if getattr(trainer, "ema", None) is not None:
                model_sd = trainer.model.state_dict()
                for k, v in trainer.ema.ema.state_dict().items():
                    if k in model_sd:
                        v.copy_(model_sd[k])
                print("Synced injected weights to EMA model")

        print("Freezing cv2 (bbox regression) parameters...")
        freeze_cv2_parameters(trainer.model)

    model.add_callback("on_pretrain_routine_end", _on_pretrain_routine_end)

    # ------------------------------------------------------------------
    # Finetune
    # ------------------------------------------------------------------
    finetune_epochs = args.epochs if args.epochs > 0 else int(config["epochs"]["finetune"])
    data_yaml = output_root / f"{get_yaml_prefix(config)}_finetune.yaml"

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
        mosaic=0.0,
    )

    best_path = runs_dir / run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}")
    print(f"Finetune stage checkpoint: {best_path}")


if __name__ == "__main__":
    main()
