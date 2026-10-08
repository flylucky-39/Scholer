"""VOC base pretrain — clean protocol (from scratch, zero external supervision).

Replaces the contaminated VOC base (which was initialized from official COCO
yolo11s.pt). Trains a standard YOLO11s from random initialization on the 15
VOC base classes only. This matches the strictest FSOD protocol: no ImageNet,
no COCO, no supervision of any kind on novel classes.

Usage:
  python scripts/train_base_voc_clean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

ARCH = PROJECT_ROOT / "configs" / "yolo11s.yaml"  # architecture only — no weights
DATA = PROJECT_ROOT / "data" / "voc_fsod_split1_10shot" / "voc_fsod_base.yaml"


def main() -> None:
    model = YOLO(str(ARCH))
    model.train(
        data=str(DATA),
        epochs=200,
        imgsz=640,
        batch=64,
        workers=16,
        cache=True,
        device=0,
        lr0=0.01,  # from-scratch standard (same recipe as the clean COCO base)
        seed=3407,
        project=str(PROJECT_ROOT / "runs" / "voc_base_clean"),
        name="base_pretrain",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
