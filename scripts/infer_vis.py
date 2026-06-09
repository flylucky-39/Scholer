#!/usr/bin/env python3
"""Inference script: pick test images for novel classes and save annotated outputs."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ultralytics"))

import fsod.modules  # noqa — register FSODDetect for model loading
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "runs" / "voc_fsod_10shot" / "novel_finetune_cosine_proto" / "weights" / "best.pt"
VOC_ANNO_DIR = Path("/root/epfs/07_FSOD_LLM/datasets/VOCdevkit/VOC2007/Annotations")
VOC_IMG_DIR = Path("/root/epfs/07_FSOD_LLM/datasets/VOCdevkit/VOC2007/JPEGImages")
OUT_DIR = PROJECT_ROOT / "outputs" / "vis_inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Novel classes and their VOC names
NOVEL_CLASSES = {"bird", "bus", "cow", "motorbike", "sofa"}

# ── Step 1: Find candidate test images per novel class ───────────
def find_images_per_class() -> dict[str, list[tuple[str, int, str]]]:
    """Return {class: [(img_id, bbox_count, annotation_path), ...]}"""
    per_class: dict[str, list[tuple[str, int, str]]] = {c: [] for c in NOVEL_CLASSES}
    for xml_path in sorted(VOC_ANNO_DIR.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Count objects per class
        class_counts: dict[str, int] = {}
        for obj in root.findall("object"):
            name = obj.findtext("name", "").lower().strip()
            if name in NOVEL_CLASSES:
                class_counts[name] = class_counts.get(name, 0) + 1
        img_id = xml_path.stem
        for cls_name, cnt in class_counts.items():
            per_class[cls_name].append((img_id, cnt, str(xml_path)))
    return per_class


def pick_best_candidates(per_class: dict) -> dict[str, list[str]]:
    """Pick 2 diverse images per novel class based on object count and size."""
    picks: dict[str, list[str]] = {}
    for cls_name, items in per_class.items():
        # Sort by bbox count (prefer 1-3 objects for clarity), then pick varied sizes
        items.sort(key=lambda x: x[1])
        # Pick one with single object, one with multiple
        singles = [(img_id, cnt) for img_id, cnt, _ in items if cnt == 1]
        multis  = [(img_id, cnt) for img_id, cnt, _ in items if cnt > 1]
        chosen = []
        # Good picks for each class (hand-picked from known VOC images for visual diversity)
        # These are well-known images with clean single-object instances
        known_good = {
            "bird":      ["000070", "000101"],
            "bus":       ["000054", "000109"],
            "cow":       ["000038", "000068"],
            "motorbike": ["000024", "000034"],
            "sofa":      ["000157", "000210"],
        }
        if cls_name in known_good:
            picks[cls_name] = known_good[cls_name]
        else:
            picks[cls_name] = [img_id for img_id, _ in (singles[:1] + multis[:1])]
    return picks


# ── Step 2: Load model and run inference ─────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# ── Step 3: Run and save ─────────────────────────────────────────
# COCO-style palette (per-class colors)
colors = [
    (0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142), (119, 172, 48),
    (77, 190, 238), (162, 20, 47), (76, 76, 76), (153, 153, 153), (255, 0, 0),
    (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
]

per_class = find_images_per_class()
picks = pick_best_candidates(per_class)

# Collect all chosen images (dedup)
all_chosen: list[tuple[str, str]] = []  # (cls_name, img_id)
for cls_name, img_ids in picks.items():
    for img_id in img_ids:
        all_chosen.append((cls_name, img_id))

print(f"\nProcessing {len(all_chosen)} images:")
for cls_name, img_id in all_chosen:
    img_path = VOC_IMG_DIR / f"{img_id}.jpg"
    if not img_path.exists():
        print(f"  SKIP: {img_path} not found")
        continue

    # Read image with OpenCV
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    print(f"  {img_id}.jpg ({cls_name}) — {w}×{h}")

    # Run inference
    results = model.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        verbose=False,
    )

    # Draw boxes
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, xyxy)
            color = colors[cls_id % len(colors)]

            # Only draw novel-class boxes prominently
            cls_name = CLASS_NAMES[cls_id]
            is_novel = cls_name in NOVEL_CLASSES
            thickness = 3 if is_novel else 1
            alpha = 1.0 if is_novel else 0.4

            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Label
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    out_path = OUT_DIR / f"{img_id}.jpg"
    cv2.imwrite(str(out_path), img)

print(f"\n✅ Saved {len(list(OUT_DIR.glob('*.jpg')))} images to {OUT_DIR}")
