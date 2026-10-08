"""Base pretrain with ImageNet-backbone initialization (TFA-protocol aligned).

Loads the surgically-prepared checkpoint (detection model + ImageNet backbone,
see runs/coco_imagenet_init.pt) and trains on COCO base classes with RAM
caching enabled (98K images ≈ 115GB, fits in 2TB RAM).

Usage:
  python scripts/train_base_imagenet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401  (registers FSODDetect etc.)
from ultralytics import YOLO

INIT_CKPT = PROJECT_ROOT / "runs" / "coco_imagenet_init.pt"
DATA = PROJECT_ROOT / "data" / "coco_fsod_10shot" / "coco_fsod_base.yaml"


def main() -> None:
    model = YOLO(str(INIT_CKPT))
    model.train(
        data=str(DATA),
        epochs=80,
        imgsz=640,
        batch=64,
        workers=48,
        cache=True,  # 2TB RAM: cache all 98K images (~115GB), kills the dataloader bottleneck
        device=0,
        lr0=0.002,  # fine-tune LR: preserves ImageNet features (lr0=0.01 washes them out)
        seed=3407,
        project=str(PROJECT_ROOT / "runs" / "coco_fsod_10shot_imagenet"),
        name="base_pretrain",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
