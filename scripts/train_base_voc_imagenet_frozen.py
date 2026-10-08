"""VOC base pretrain — ImageNet backbone (frozen), TFA-protocol aligned.

Backbone layers 0-8 initialized from yolo11s-cls.pt (ImageNet-1k) and FROZEN.
SPPF, C2PSA, neck, and detection heads train on the 15 VOC base classes.

Design rationale (after two failed unfrozen attempts):
- lr0=0.01 unfrozen: washed out the ImageNet features
- lr0=0.002 unfrozen: too slow, never caught from-scratch
- FROZEN layers 0-8: the ImageNet features pass through untouched; only the
  detection-specific parts (SPPF/C2PSA/neck/head) learn. No washout possible.
- Only cleanly-aligned layers 0-8 are transferred (the cls C2PSA is skipped
  to avoid the SPPF input-misalignment flaw of the previous surgery).

This matches the TFA protocol (ImageNet backbone init is standard there).

Usage:
  python scripts/train_base_voc_imagenet_frozen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401
from ultralytics import YOLO

CLS = PROJECT_ROOT.parent / "yolo11s-cls.pt"
DATA = PROJECT_ROOT / "data" / "voc_fsod_split1_10shot" / "voc_fsod_base.yaml"


def main() -> None:
    # 1) Detection model from scratch architecture
    det = YOLO(str(PROJECT_ROOT / "configs" / "yolo11s.yaml"))
    det_sd = det.model.state_dict()

    # 2) Transfer ONLY cleanly-aligned layers 0-8 from ImageNet classifier
    cls_ckpt = torch.load(str(CLS), map_location="cpu", weights_only=False)
    cls_model = cls_ckpt.get("ema") or cls_ckpt["model"]
    cls_sd = {k: v.float() for k, v in cls_model.state_dict().items()}

    transfer = {}
    for k, v in cls_sd.items():
        parts = k.split(".")
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) <= 8:
            if k in det_sd and det_sd[k].shape == v.shape:
                transfer[k] = v
    det.model.load_state_dict(transfer, strict=False)
    print(f"Transferred {len(transfer)} ImageNet tensors into layers 0-8")

    # 3) Train with layers 0-8 frozen
    det.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=64,
        workers=16,
        cache=True,
        device=0,
        lr0=0.01,  # safe: frozen layers cannot be washed out
        freeze=9,  # freeze modules 0-8 (the ImageNet part)
        seed=3407,
        project=str(PROJECT_ROOT / "runs" / "voc_base_imagenet_frozen"),
        name="base_pretrain",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
