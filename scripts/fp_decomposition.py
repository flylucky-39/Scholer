"""FP decomposition: background FP vs cross-class FP vs localization FP.

Compares Standard Finetune vs Cosine+Proto on COCO 10-shot novel-val,
classifying every novel-class false positive into:
  - background FP     : no overlap with any GT box
  - cross-class FP    : overlaps a GT box of a different novel class
  - localization FP   : overlaps same-class GT but IoU < 0.5

Usage:
  python scripts/fp_decomposition.py
"""
from __future__ import annotations

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401  (registers FSODDetect for unpickling)
from ultralytics import YOLO

NOVEL_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch',
               'potted plant', 'dining table', 'tv']

DATA = PROJECT_ROOT / 'data' / 'coco_fsod_10shot'
N_IMAGES = 600
CONF = 0.10
IOU_TP = 0.5


def load_gt(label_path: Path, w: int, h: int):
    boxes, cls = [], []
    if not label_path.exists():
        return boxes, cls
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
        cls.append(c)
    return boxes, cls


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def decompose(weights: str, label: str, novel_ids: set):
    model = YOLO(weights)
    manifest = DATA / 'manifests' / 'val_novel.txt'
    paths = [Path(l.strip()) for l in manifest.read_text().splitlines()]
    rng = random.Random(42)
    paths = rng.sample(paths, min(N_IMAGES, len(paths)))

    counts = dict(tp=0, bg=0, cross=0, loc=0, base_pred=0)
    novel_pred_total = 0
    n_img = 0

    results = model.predict(source=[str(p) for p in paths], conf=CONF, iou=0.7,
                            imgsz=640, device=0, verbose=False, stream=True, max_det=300)
    for r in results:
        p = Path(r.path)
        n_img += 1
        gt_boxes, gt_cls = load_gt(DATA / 'labels' / 'val_novel' / (p.stem + '.txt'),
                                   r.orig_shape[1], r.orig_shape[0])
        for bi in range(len(r.boxes)):
            cid = int(r.boxes.cls[bi])
            if cid not in novel_ids:
                counts['base_pred'] += 1
                continue
            novel_pred_total += 1
            pb = r.boxes.xyxy[bi].tolist()

            best_same, best_any_other = 0.0, 0.0
            for gb, gc in zip(gt_boxes, gt_cls):
                v = iou(pb, gb)
                if gc == cid:
                    best_same = max(best_same, v)
                else:
                    best_any_other = max(best_any_other, v)

            if best_same >= IOU_TP:
                counts['tp'] += 1
            elif best_same > 0.1:
                counts['loc'] += 1
            elif best_any_other >= 0.3:
                counts['cross'] += 1
            else:
                counts['bg'] += 1

    del model
    torch.cuda.empty_cache()
    fp_total = counts['bg'] + counts['cross'] + counts['loc']
    return dict(label=label, n_img=n_img, novel_pred=novel_pred_total, **counts,
                fp_total=fp_total,
                bg_pct=100 * counts['bg'] / fp_total if fp_total else 0,
                cross_pct=100 * counts['cross'] / fp_total if fp_total else 0,
                loc_pct=100 * counts['loc'] / fp_total if fp_total else 0,
                fp_per_img=fp_total / n_img if n_img else 0)


def main():
    ckpts = [
        ('runs/coco_fsod_10shot_exp/novel_finetune/weights/best.pt', 'Standard Finetune'),
        ('runs/coco_fsod_10shot_New/novel_finetune_cosine_proto/weights/best.pt', 'Cosine+Proto'),
    ]
    # derive novel id set from a checkpoint's names
    m = YOLO(ckpts[0][0])
    names = m.names
    novel_ids = {i for i, n in names.items() if n in NOVEL_NAMES}
    del m
    print(f"novel ids ({len(novel_ids)}): {sorted(novel_ids)}\n")

    out = []
    print(f"{'Model':<20}{'imgs':>6}{'novelPred':>11}{'TP':>7}{'FP/img':>8}"
          f"{'bgFP%':>8}{'crossFP%':>10}{'locFP%':>8}")
    print('-' * 78)
    for w, label in ckpts:
        r = decompose(w, label, novel_ids)
        out.append(r)
        print(f"{r['label']:<20}{r['n_img']:>6}{r['novel_pred']:>11}{r['tp']:>7}"
              f"{r['fp_per_img']:>8.1f}{r['bg_pct']:>8.1f}{r['cross_pct']:>10.1f}{r['loc_pct']:>8.1f}")

    dst = PROJECT_ROOT.parent / 'final_experiments' / 'results' / 'fp_decomposition_coco.json'
    dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\nsaved → {dst}")


if __name__ == '__main__':
    main()
