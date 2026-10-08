"""Zero-shot leak ceiling: official COCO yolo11s.pt on VOC novel test set.

Quantifies how much the COCO-pretrained backbone lineage contributes to VOC
few-shot numbers. Evaluates the OFFICIAL COCO checkpoint (80 classes, full
supervision incl. all 20 VOC classes) directly on the VOC split-1 novel test
set, mapping COCO class ids to VOC ids, and computes novel-only mAP50.

This is the upper bound of what the leaked backbone 'knows' zero-shot.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fsod.modules  # noqa: F401
from ultralytics import YOLO

DATA = PROJECT_ROOT / 'data' / 'voc_fsod_split1_10shot'
CKPT = PROJECT_ROOT / 'yolo11s.pt'  # official COCO checkpoint

# VOC split-1 novel: VOC name -> (VOC id, COCO id)
NOVEL_MAP = {'bird': (2, 14), 'bus': (5, 5), 'cow': (9, 19),
             'motorbike': (13, 3), 'sofa': (17, 57)}
VOC_TO_COCO = {2: 14, 5: 5, 9: 19, 13: 3, 17: 57}


def load_gt(label_path: Path, w: int, h: int):
    boxes, cls = [], []
    if not label_path.exists():
        return boxes, cls
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(c)
    return boxes, cls


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(scores, tps, n_gt):
    """AP@0.50 from ranked predictions."""
    if n_gt == 0:
        return float('nan')
    order = np.argsort(-np.asarray(scores))
    tps = np.asarray(tps)[order]
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(1 - tps)
    recall = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # integrate precision envelope
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    model = YOLO(str(CKPT))
    paths = [Path(l.strip()) for l in (DATA / 'manifests' / 'test_novel.txt').read_text().splitlines()]
    results = model.predict(source=[str(p) for p in paths], conf=0.001, iou=0.7,
                            imgsz=640, device=0, verbose=False, stream=True, max_det=300)
    per_cls = {v: {'scores': [], 'tps': [], 'n_gt': 0} for v in VOC_TO_COCO}
    for r in results:
        p = Path(r.path)
        gt_boxes, gt_cls = load_gt(DATA / 'labels' / 'test_novel' / (p.stem + '.txt'),
                                   r.orig_shape[1], r.orig_shape[0])
        for v_id in VOC_TO_COCO:
            per_cls[v_id]['n_gt'] += sum(1 for c in gt_cls if c == v_id)
        matched = {v_id: set() for v_id in VOC_TO_COCO}
        for bi in range(len(r.boxes)):
            cid = int(r.boxes.cls[bi])
            voc_id = next((v for v, c in VOC_TO_COCO.items() if c == cid), None)
            if voc_id is None:
                continue
            pb = r.boxes.xyxy[bi].tolist()
            best_iou, best_j = 0.0, -1
            for j, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
                if gc == voc_id and j not in matched[voc_id]:
                    v = iou(pb, gb)
                    if v > best_iou:
                        best_iou, best_j = v, j
            tp = best_iou >= 0.5
            if tp:
                matched[voc_id].add(best_j)
            per_cls[voc_id]['scores'].append(float(r.boxes.conf[bi]))
            per_cls[voc_id]['tps'].append(1 if tp else 0)

    print("官方 COCO yolo11s.pt 零样本 → VOC split1 novel 测试集")
    print("=" * 56)
    aps = {}
    voc_names = {2: 'bird', 5: 'bus', 9: 'cow', 13: 'motorbike', 17: 'sofa'}
    for v_id in sorted(VOC_TO_COCO):
        d = per_cls[v_id]
        ap = ap50(d['scores'], d['tps'], d['n_gt'])
        aps[voc_names[v_id]] = ap
        print(f"  {voc_names[v_id]:<10} AP50 = {ap:.4f}  ({d['n_gt']} GT, {len(d['scores'])} pred)")
    m = float(np.nanmean(list(aps.values())))
    print("-" * 56)
    print(f"  零样本 novel mAP50 = {m:.4f}   ← 泄漏天花板")
    print(f"  (对比: 论文 Standard 10-shot = 0.5133, 完整方法 10-shot = 0.7838)")
    out = dict(zero_shot_map50=m, per_class=aps,
               reference=dict(standard_10shot=0.5133, ours_10shot=0.7838))
    dst = PROJECT_ROOT.parent / 'final_experiments' / 'results' / 'voc_leak_ceiling.json'
    dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"\nsaved → {dst}")


if __name__ == '__main__':
    main()
