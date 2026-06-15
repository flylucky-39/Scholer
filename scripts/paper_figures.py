#!/usr/bin/env python3
"""Paper figures: per-object matching, clean visualization."""
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ultralytics"))
import fsod.modules  # noqa
from ultralytics import YOLO

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
COLORS = {
    "bird": (255, 128, 0), "bus": (0, 200, 255), "cow": (0, 220, 0),
    "motorbike": (255, 0, 80), "sofa": (200, 0, 255),
}
GT_CLR = (80, 220, 80)
MISS_CLR = (50, 50, 255)

IMGD = PROJECT_ROOT / "data" / "voc_fsod_split1_10shot" / "images" / "test_novel"
LBLD = PROJECT_ROOT / "data" / "voc_fsod_split1_10shot" / "labels" / "test_novel"
OUT = PROJECT_ROOT / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

PICKS = [
    ("VOC2007_004083", "bird"),      # 2 birds, verified
    ("VOC2007_000195", "bus"),
    ("VOC2007_002299", "cow"),
    ("VOC2007_001798", "motorbike"),  # 3 motorbikes
    ("VOC2007_002489", "sofa"),
]


def box_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)


def get_gts(img_id, target_cls, w, h):
    lbl = LBLD / f"{img_id}.txt"
    boxes = []
    if not lbl.exists():
        return boxes
    target_id = CLASSES.index(target_cls)
    with open(lbl) as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            if int(float(p[0])) != target_id:
                continue
            cx, cy, bw, bh = map(float, p[1:5])
            boxes.append(tuple(map(int, [
                (cx - bw / 2) * w, (cy - bh / 2) * h,
                (cx + bw / 2) * w, (cy + bh / 2) * h,
            ])))
    return boxes


model = YOLO(str(PROJECT_ROOT / "runs/voc_fsod_10shot/novel_finetune_cosine_proto/weights/best.pt"))
print(f"Output: {OUT}\n")

for img_id, target in PICKS:
    img_path = str(IMGD / f"{img_id}.jpg")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gts = get_gts(img_id, target, w, h)
    if not gts:
        print(f"  {target} {img_id} — no GT, skip")
        continue

    # Run detection
    r = model(img_path, imgsz=640, conf=0.001, iou=0.5, device=0, verbose=False)
    boxes = r[0].boxes
    preds = []  # [(conf, x1,y1,x2,y2), ...]
    if boxes is not None:
        for b in boxes:
            if CLASSES[int(b.cls[0])] == target:
                c = float(b.conf[0])
                xy = [int(v) for v in b.xyxy[0].tolist()]
                preds.append((c, xy[0], xy[1], xy[2], xy[3]))
    preds.sort(key=lambda x: x[0], reverse=True)

    # Match: for each GT, pick best pred with IoU > 0.2
    used_preds = set()
    matches = []  # [(gt_idx, conf, x1,y1,x2,y2, iou)]
    for gi, gt in enumerate(gts):
        best_pi, best_c, best_iou = -1, 0, 0
        for pi, pdata in enumerate(preds):
            if pi in used_preds:
                continue
            c, px1, py1, px2, py2 = pdata
            i = box_iou(gt, (px1, py1, px2, py2))
            if i > 0.2 and c > best_c:
                best_pi, best_c, best_iou = pi, c, i
        if best_pi >= 0:
            used_preds.add(best_pi)
            _, px1, py1, px2, py2 = preds[best_pi]
            matches.append((gi, best_c, px1, py1, px2, py2, best_iou))

    matched_gi = {gi for gi, _, _, _, _, _, _ in matches}
    missed = [gi for gi in range(len(gts)) if gi not in matched_gi]

    color = COLORS[target]

    # Draw GTs
    for gi, (x1, y1, x2, y2) in enumerate(gts):
        c = GT_CLR if gi in matched_gi else MISS_CLR
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        lbl = "GT" if gi in matched_gi else "GT(?)"
        tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 4, y1), c, -1)
        cv2.putText(img, lbl, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Draw predictions
    for gi, conf, x1, y1, x2, y2, iou_val in matches:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = target
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    path = OUT / f"{target}.jpg"
    cv2.imwrite(str(path), img)
    ious = ", ".join(f"{iou_val:.3f}" for _, _, _, _, _, _, iou_val in matches)
    print(f"  {target}  {len(gts)}GT → {len(matches)} matched ({len(missed)} miss)  IoU=[{ious}]")

print(f"\nDone: {OUT}")
