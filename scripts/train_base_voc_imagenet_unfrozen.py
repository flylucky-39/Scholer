"""VOC base pretrain — ImageNet backbone (UNFROZEN), TFA-standard protocol.

The literature-standard approach: ImageNet initialization with the backbone
trainable during base training (features adapt from classification to
detection). Uses the CLEAN surgery (layers 0-8 only, no C2PSA misalignment)
and a middle-ground LR (0.005) that avoids both prior failure modes:
- lr0=0.01 washed out ImageNet features (COCO experiment)
- lr0=0.002 converged too slowly (COCO experiment)

Usage:
  python scripts/train_base_voc_imagenet_unfrozen.py
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

    # 2) Clean surgery: transfer ONLY layers 0-8 (no C2PSA misalignment)
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

    # 3) Train UNFROZEN with middle-ground LR
    det.train(
        data=str(DATA),
        epochs=200,       # match from-scratch protocol for fair comparison
        imgsz=640,
        batch=64,
        workers=16,
        cache=True,
        device=0,
        lr0=0.005,        # middle ground: 0.01 washed out, 0.002 too slow
        freeze=0,         # UNFROZEN — backbone adapts to detection (TFA standard)
        seed=3407,
        project=str(PROJECT_ROOT / "runs" / "voc_base_imagenet_unfrozen"),
        name="base_pretrain",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
