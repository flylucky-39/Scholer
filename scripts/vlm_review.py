#!/usr/bin/env python3
"""VLM Review: Florence-2 audits YOLO predictions, corrects wrong bboxes, filters easy images.

Pipeline:
  1. YOLO-FSOD predicts on each support image
  2. Florence-2 OD provides reference annotations
  3. Match YOLO ↔ Florence-2 bboxes (IoU > 0.3)
  4. Classify image:
     - All matched AND no FP → "easy" → remove from support set
     - Otherwise → "hard" → keep, with Florence-2 corrected labels
  5. Output reviewed support set (hard images only)

Usage:
  python scripts/vlm_review.py \
    --weights runs/fsod_1shot/novel_finetune_cosine_proto/weights/best.pt \
    --config configs/baseline_voc_1shot.yaml \
    --florence2 models/Florence-2-base/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

# ── Project root setup ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "third_party" / "ultralytics") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "ultralytics"))

import fsod.modules  # noqa — register FSODDetect
from fsod.florence2 import (
    Florence2Runner,
    filter_detections_by_classes,
    canonicalize_label,
    LABEL_SYNONYMS,
)
from ultralytics import YOLO

# ── Constants ───────────────────────────────────────────────────
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

IOU_THRESHOLD = 0.3  # Min IoU for YOLO ↔ Florence-2 match
YOLO_CONF = 0.001    # Min YOLO conf (low — we filter by top-K instead)
TOP_K_PER_CLASS = 3   # Keep at most K YOLO boxes per novel class


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM Review support set with Florence-2")
    p.add_argument("--weights", required=True, help="YOLO-FSOD checkpoint")
    p.add_argument("--config", required=True, help="Experiment config yaml")
    p.add_argument("--florence2", default="models/Florence-2-base",
                   help="Florence-2 model dir")
    p.add_argument("--device", default="0", help="CUDA device")
    p.add_argument("--output-suffix", default="_reviewed",
                   help="Suffix for reviewed data dir")
    return p.parse_args()


def box_iou(a, b):
    """IoU between two xyxy boxes."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def yolo_cxcywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Normalized cxcywh → pixel xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def xyxy_to_cxcywh(x1, y1, x2, y2, img_w, img_h):
    """Pixel xyxy → normalized cxcywh."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def load_gt_labels(label_dir: Path, img_id: str) -> list[dict]:
    """Load YOLO-format GT labels for an image.
    Returns list of {class_id, class_name, cx, cy, w, h, x1, y1, x2, y2}.
    """
    label_path = label_dir / f"{img_id}.txt"
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            labels.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "cx": cx, "cy": cy, "w": w, "h": h,
            })
    return labels


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    novel_classes = config["novel_classes"]
    novel_set = set(novel_classes)

    run_name = Path(args.weights).parent.name
    reviewed_dir = output_root.parent / (output_root.name + args.output_suffix)
    label_dir = reviewed_dir / "labels" / "novel_finetune"
    image_dir = reviewed_dir / "images" / "novel_finetune"
    manifest_dir = reviewed_dir / "manifests"
    for d in [label_dir, image_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifests" / "novel_finetune.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    image_paths = [Path(line.strip()) for line in manifest_path.read_text().splitlines() if line.strip()]

    src_label_dir = output_root / "labels" / "novel_finetune"

    # ── Load YOLO model ─────────────────────────────────────
    print(f"Loading YOLO: {args.weights}")
    model = YOLO(str(Path(args.weights).resolve()))

    # ── Load Florence-2 ──────────────────────────────────────
    print(f"Loading Florence-2: {args.florence2}")
    fl2 = Florence2Runner(model_path=str(PROJECT_ROOT.parent / args.florence2),
                          device=f"cuda:{args.device}")

    # ── Per-image review ─────────────────────────────────────
    stats = {
        "total": len(image_paths),
        "easy": 0, "hard": 0,
        "bboxes_total": 0, "bboxes_corrected": 0,
        "bboxes_removed_fp": 0, "bboxes_added_missed": 0,
        "per_class": {c: {"easy": 0, "hard": 0} for c in novel_classes},
        "per_image": [],
    }

    print(f"\nReviewing {len(image_paths)} images...\n")

    for img_idx, img_path in enumerate(image_paths):
        img_id = img_path.stem
        print(f"[{img_idx+1}/{len(image_paths)}] {img_id}")

        # Get image dimensions
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path)
        img_w, img_h = pil_img.size

        # ── 1. YOLO prediction ──
        results = model(str(img_path), imgsz=640, conf=YOLO_CONF, iou=0.5,
                        device=f"cuda:{args.device}", verbose=False)
        # Group YOLO boxes by class, keep top-K per class
        class_boxes: dict[str, list[dict]] = {}
        if results[0].boxes is not None:
            for b in results[0].boxes:
                cid = int(b.cls[0])
                conf = float(b.conf[0])
                name = CLASS_NAMES[cid]
                if name not in novel_set:
                    continue
                if name not in class_boxes:
                    class_boxes[name] = []
                class_boxes[name].append({
                    "class_id": cid, "class_name": name,
                    "conf": conf,
                    "bbox_xyxy": b.xyxy[0].tolist(),
                    "matched": False, "match_label": None,
                })
        # Sort each class by confidence, keep top-K
        yolo_boxes = []
        for cls_name, boxes in class_boxes.items():
            boxes.sort(key=lambda b: b["conf"], reverse=True)
            yolo_boxes.extend(boxes[:TOP_K_PER_CLASS])

        # ── 2. Florence-2 OD ──
        fl2_result = fl2.run(image_path=str(img_path), task="od")
        fl2_dets = filter_detections_by_classes(
            fl2_result.detections, novel_classes, LABEL_SYNONYMS,
        )
        fl2_boxes = []
        for det in fl2_dets:
            canonical = canonicalize_label(det["label"], LABEL_SYNONYMS)
            fl2_boxes.append({"class_name": canonical, "bbox_xyxy": det["bbox"],
                              "matched": False, "match_idx": None})

        # ── 3. Load GT ──
        gt_labels = load_gt_labels(src_label_dir, img_id)
        stats["bboxes_total"] += len(gt_labels)

        # ── 4. Match YOLO ↔ Florence-2 ──
        # For each YOLO box, find best Florence-2 match
        for yi, yb in enumerate(yolo_boxes):
            best_iou, best_fi = 0, -1
            for fj, fb in enumerate(fl2_boxes):
                if fb["matched"]:
                    continue
                iou_val = box_iou(yb["bbox_xyxy"], fb["bbox_xyxy"])
                if iou_val > IOU_THRESHOLD and iou_val > best_iou:
                    best_iou, best_fi = iou_val, fj
            if best_fi >= 0:
                yb["matched"] = True
                yb["match_label"] = fl2_boxes[best_fi]["class_name"]
                yb["match_iou"] = best_iou
                fl2_boxes[best_fi]["matched"] = True

        # ── 5. Classify results ──
        # "easy" = YOLO correctly finds the target object for this image's class
        # (top YOLO box matches Florence-2 in class AND IoU > 0.3)
        # "hard" = YOLO misses or misclassifies its best prediction
        # NOTE: FP on OTHER classes is expected for cosine classifier and does
        # NOT make an image "hard" — only the best detection matters.
        matched_boxes = [yb for yb in yolo_boxes if yb["matched"]]
        correct_matches = [yb for yb in matched_boxes
                           if yb["class_name"] == yb["match_label"]]

        # The image's "target class" is determined by GT labels
        # "easy" = at least 1 correct match for any novel class
        # "hard" = no correct matches at all
        if len(correct_matches) > 0:
            is_easy = True
        else:
            is_easy = False

        # Compute stats for logging
        num_fp = sum(1 for yb in yolo_boxes if not yb["matched"])
        num_missed = sum(1 for fb in fl2_boxes if not fb["matched"])
        num_wrong_class = sum(
            1 for yb in yolo_boxes
            if yb["matched"] and yb["class_name"] != yb["match_label"]
        )

        # Determine which novel classes are in this image
        image_classes = set(gt["class_name"] for gt in gt_labels)

        if is_easy:
            stats["easy"] += 1
            for cls_name in image_classes:
                if cls_name in stats["per_class"]:
                    stats["per_class"][cls_name]["easy"] += 1
            # 1-shot: keep all images (Florence-2 labels are more precise)
            print(f"  → CORRECT (YOLO ✓ Florence-2, keeping with improved labels)")
        else:
            stats["hard"] += 1
            for cls_name in image_classes:
                if cls_name in stats["per_class"]:
                    stats["per_class"][cls_name]["hard"] += 1
            print(f"  → CORRECTED (YOLO ✗ Florence-2, fixing labels)")

        # ── 6. Build corrected labels for ALL images ──
        # Use Florence-2 bboxes as the reference annotation.
        corrected_labels = []
        for fi, fb in enumerate(fl2_boxes):
            cls_id = CLASS_NAMES.index(fb["class_name"])
            cx, cy, w, h = xyxy_to_cxcywh(*fb["bbox_xyxy"], img_w, img_h)
            corrected_labels.append((cls_id, cx, cy, w, h))
        if not fl2_boxes:
            # Florence-2 found nothing novel — keep original GT labels
            for gt in gt_labels:
                corrected_labels.append(
                    (gt["class_id"], gt["cx"], gt["cy"], gt["w"], gt["h"]))

        stats["bboxes_removed_fp"] += num_fp
        stats["bboxes_corrected"] += num_wrong_class
        stats["bboxes_added_missed"] += num_missed

        # Write corrected labels
        label_path = label_dir / f"{img_id}.txt"
        with open(label_path, "w") as f:
            for cls_id, cx, cy, w, h in corrected_labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Copy image
        dest_img = image_dir / f"{img_id}.jpg"
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        detail_parts = []
        if num_fp > 0: detail_parts.append(f"{num_fp} FP removed")
        if num_wrong_class > 0: detail_parts.append(f"{num_wrong_class} class fixed")
        if num_missed > 0: detail_parts.append(f"{num_missed} missed added")
        detail = ", ".join(detail_parts) if detail_parts else "clean"
        print(f"  → labels: {len(gt_labels)} GT → {len(corrected_labels)} FL2 [{detail}]")

        stats["per_image"].append({
            "img_id": img_id,
            "is_easy": is_easy,
            "yolo_boxes": len(yolo_boxes),
            "florence2_boxes": len(fl2_boxes),
            "fp": num_fp,
            "missed": num_missed,
            "wrong_class": num_wrong_class,
        })

    # ── 7. Write manifest (all images kept with Florence-2 labels) ──
    manifest_lines = []
    for item in stats["per_image"]:
        dest_img = image_dir / f"{item['img_id']}.jpg"
        manifest_lines.append(str(dest_img.resolve()))
    manifest_out = manifest_dir / "novel_finetune.txt"
    manifest_out.write_text("\n".join(manifest_lines) + "\n")

    # ── 8. Copy other files to make a valid data dir ──
    # Copy voc_fsod_finetune.yaml, voc_fsod_eval_all.yaml, etc.
    for yaml_file in output_root.glob("voc_fsod_*.yaml"):
        dest_yaml = reviewed_dir / yaml_file.name
        if not dest_yaml.exists():
            content = yaml_file.read_text()
            # Update path in yaml
            content = content.replace(str(output_root), str(reviewed_dir))
            dest_yaml.write_text(content)

    # Copy eval manifests (test, test_novel — unchanged)
    src_test_manifest = output_root / "manifests" / "test.txt"
    src_test_novel_manifest = output_root / "manifests" / "test_novel.txt"
    dest_test_manifest = manifest_dir / "test.txt"
    dest_test_novel_manifest = manifest_dir / "test_novel.txt"
    # Copy eval label and image dirs (these are full, same for original and reviewed)
    if not (reviewed_dir / "labels" / "test").exists():
        src_labels_test = output_root / "labels" / "test"
        if src_labels_test.exists():
            shutil.copytree(src_labels_test, reviewed_dir / "labels" / "test",
                           dirs_exist_ok=True)
    if not (reviewed_dir / "labels" / "test_novel").exists():
        src_labels_tn = output_root / "labels" / "test_novel"
        if src_labels_tn.exists():
            shutil.copytree(src_labels_tn, reviewed_dir / "labels" / "test_novel",
                           dirs_exist_ok=True)
    if not (reviewed_dir / "labels" / "base_train").exists():
        src_labels_base = output_root / "labels" / "base_train"
        if src_labels_base.exists():
            shutil.copytree(src_labels_base, reviewed_dir / "labels" / "base_train",
                           dirs_exist_ok=True)

    # Copy eval images if they don't exist
    for split_name in ["test", "test_novel"]:
        src_imgs = output_root / "images" / split_name
        dst_imgs = reviewed_dir / "images" / split_name
        if src_imgs.exists() and not dst_imgs.exists():
            dst_imgs.mkdir(parents=True, exist_ok=True)
            for img_file in src_imgs.iterdir():
                if not (dst_imgs / img_file.name).exists():
                    shutil.copy2(img_file, dst_imgs / img_file.name)

    # ── 9. Save report ──
    report_path = reviewed_dir / "review_report.json"
    json.dump(stats, open(report_path, "w", encoding="utf-8"), indent=2)

    # ── 10. Summary ──
    print(f"\n{'='*60}")
    print(f"  VLM Review Complete")
    print(f"{'='*60}")
    print(f"  Total images:   {stats['total']}")
    print(f"  Easy (removed): {stats['easy']} ({stats['easy']/max(stats['total'],1)*100:.0f}%)")
    print(f"  Hard (kept):    {stats['hard']} ({stats['hard']/max(stats['total'],1)*100:.0f}%)")
    print(f"  Bboxes corrected: {stats['bboxes_corrected']}/{stats['bboxes_total']}")
    print(f"  FP removed:       {stats['bboxes_removed_fp']}")
    print(f"  Missed added:     {stats['bboxes_added_missed']}")
    print(f"\n  Per class:")
    for cls_name, cstats in sorted(stats["per_class"].items()):
        total_cls = cstats["easy"] + cstats["hard"]
        print(f"    {cls_name:<12}: {cstats['easy']}E / {cstats['hard']}H "
              f"({cstats['easy']/max(total_cls,1)*100:.0f}% easy)")
    print(f"\n  Output: {reviewed_dir}")
    print(f"  Report: {report_path}")


def match_result_summary(yolo_boxes, fl2_boxes):
    """Short text summary for per-image logging."""
    matched = sum(1 for yb in yolo_boxes if yb["matched"])
    correct = sum(1 for yb in yolo_boxes
                   if yb["matched"] and yb["class_name"] == yb["match_label"])
    return f"{len(yolo_boxes)}Y/{len(fl2_boxes)}F, {correct}c"


if __name__ == "__main__":
    main()
