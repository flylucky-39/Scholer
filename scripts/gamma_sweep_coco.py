"""Post-hoc gamma sweep on trained COCO Cosine+Proto checkpoint.

The shot-adaptive schedule (calibrated on VOC) sets gamma~0 at 10-shot, leaving
the 43% background-type FPs (per fp_decomposition.py) unaddressed on COCO.
This script injects the COCO background prototype into the trained checkpoint's
CosineConv2d buffers at various gamma values (zero retraining) and re-evaluates
novel-only mAP50 on val_novel.

Usage:
  python scripts/gamma_sweep_coco.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401
import numpy as np
from ultralytics import YOLO

from fsod.coco import COCO_CLASSES
from fsod.modules.background_suppression import extract_background_prototype

DATA = PROJECT_ROOT / 'data' / 'coco_fsod_10shot'
BASE_W = PROJECT_ROOT / 'runs' / 'coco_fsod_10shot_exp' / 'base_pretrain' / 'weights' / 'best.pt'
TARGET_CKPT = PROJECT_ROOT / 'runs' / 'coco_fsod_10shot_New' / 'novel_finetune_cosine_proto' / 'weights' / 'best.pt'
BG_CACHE = Path('/tmp/coco_bg_proto.pt')
GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
NOVEL_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch',
               'potted plant', 'dining table', 'tv']


def main():
    if BG_CACHE.exists():
        bg_proto = torch.load(BG_CACHE)
        print(f"loaded cached bg_proto, norm={bg_proto.norm():.3f}")
    else:
        bg_proto = extract_background_prototype(
            base_weights=str(BASE_W), data_root=str(DATA), all_classes=list(COCO_CLASSES),
            imgsz=640, device='cuda:0')
        torch.save(bg_proto, BG_CACHE)
        print(f"extracted bg_proto, norm={bg_proto.norm():.3f}")

    data_yaml = str(DATA / 'coco_fsod_finetune.yaml')
    out = []
    print(f"\n{'gamma':>6}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    print('-' * 45)
    for g in GAMMAS:
        model = YOLO(str(TARGET_CKPT))
        detect = model.model.model[-1]
        detect.set_background_proto(bg_proto, g)
        metrics = model.val(data=data_yaml, split='test', imgsz=640, device=0,
                            verbose=False, plots=False, exist_ok=True)
        ap50 = np.array(metrics.box.ap50)
        names = list(metrics.names.values())
        novel_mask = np.array([n in NOVEL_NAMES for n in names])
        m50 = float(ap50[novel_mask[:len(ap50)]].mean()) if len(ap50) else 0.0
        r = dict(gamma=g, novel_map50=m50, P=float(metrics.box.mp),
                 R=float(metrics.box.mr), map50_all=float(metrics.box.map50))
        out.append(r)
        print(f"{g:>6.2f}{m50:>9.4f}{float(metrics.box.map):>10.4f}"
              f"{float(metrics.box.mp):>8.3f}{float(metrics.box.mr):>8.3f}")
        del model
        torch.cuda.empty_cache()

    dst = PROJECT_ROOT.parent / 'final_experiments' / 'results' / 'gamma_sweep_coco.json'
    dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\nsaved → {dst}")


if __name__ == '__main__':
    main()
