"""COCO few-shot data utilities.

Mirrors the VOC pipeline in ``fsod.voc`` but reads COCO-format JSON
annotations via *pycocotools*.

COCO few-shot benchmark (TFA / DeFRCN convention):
  - 80 total classes → 20 novel (same semantics as VOC-20) + 60 base.
  - Novel class **COCO IDs**: {1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72}
    mapped to COCO category names.  The remaining 60 are base classes.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── COCO 80-class list in the *detection* order (0-indexed) used by Ultralytics / YOLO ──
# This matches the `coco.yaml` shipped with Ultralytics.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

if len(COCO_CLASSES) != 80:
    raise ValueError(f"COCO_CLASSES must contain 80 classes, got {len(COCO_CLASSES)}")

# The 20 novel classes that correspond to VOC-20 categories (Ultralytics 0-idx).
COCO_NOVEL_INDICES = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    6,   # train
    8,   # boat
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    39,  # bottle
    56,  # bed -> actually "dining table" ... (see note)
    # ── Actual mapping from TFA codebase ──
    # VOC names → COCO 80-class 0-idx:
    #   person=0, bicycle=1, car=2, motorcycle=3, airplane=4, bus=5,
    #   train=6, boat=8, bird=14, cat=15, dog=16, horse=17, sheep=18,
    #   cow=19, bottle=39, chair=56(?), couch/sofa=57, potted plant=58,
    #   dining table=60, tv=62(?)
    # NOTE: the index depends on the *Ultralytics* ordering.
]

# ──────────────────────────────────────────────────────────────
#  Safer: define novel classes by *name* (unambiguous across datasets).
# ──────────────────────────────────────────────────────────────
COCO_NOVEL_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "boat", "bird", "cat",
    "dog", "horse", "sheep", "cow", "bottle",
    "chair", "couch", "potted plant", "dining table", "tv",
]

# Map *COCO category_id* (1-indexed, sparse) → Ultralytics 0-indexed class id.
# COCO has 80 categories but their IDs are sparse (1..90).
# This mapping is the standard one baked into Ultralytics `coco.yaml`.
COCO_CATID_TO_IDX: dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}

COCO_IDX_TO_CATID: dict[int, int] = {v: k for k, v in COCO_CATID_TO_IDX.items()}


# ── data classes ──────────────────────────────────────────────

@dataclass(frozen=True)
class CocoObject:
    name: str           # class name (e.g. "person")
    class_idx: int      # 0-indexed class id in Ultralytics order
    bbox_xywh: tuple[float, float, float, float]  # COCO absolute [x, y, w, h]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CocoRecord:
    image_id: int
    file_name: str
    image_path: Path
    width: int
    height: int
    objects: tuple[CocoObject, ...]

    @property
    def key(self) -> int:
        return self.image_id


# ── loaders ───────────────────────────────────────────────────

def load_coco_records(
    coco_root: Path,
    split: str = "train2017",
    annotation_file: Optional[str] = None,
) -> list[CocoRecord]:
    """Load all image records from a COCO JSON annotation file.

    Parameters
    ----------
    coco_root : Path
        Root that contains ``images/train2017/`` and ``annotations/``.
    split : str
        Image sub-folder name (``train2017`` or ``val2017``).
    annotation_file : str | None
        Annotation JSON filename under ``annotations/``.  Defaults to
        ``instances_{split}.json``.
    """
    ann_dir = coco_root / "annotations"
    if annotation_file is None:
        annotation_file = f"instances_{split}.json"
    ann_path = ann_dir / annotation_file

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    catid_to_name: dict[int, str] = {}
    for cat in data["categories"]:
        idx = COCO_CATID_TO_IDX.get(cat["id"])
        if idx is not None:
            catid_to_name[cat["id"]] = COCO_CLASSES[idx]

    img_dict: dict[int, dict] = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    records: list[CocoRecord] = []
    images_dir = coco_root / "images" / split
    # Fallback: some dataset layouts put images directly under split name
    if not images_dir.exists():
        images_dir = coco_root / split

    for img_id, img_info in img_dict.items():
        objects = []
        for ann in ann_by_img.get(img_id, []):
            cat_id = ann["category_id"]
            idx = COCO_CATID_TO_IDX.get(cat_id)
            if idx is None:
                continue  # not one of the 80 detection classes
            name = COCO_CLASSES[idx]
            objects.append(CocoObject(
                name=name,
                class_idx=idx,
                bbox_xywh=tuple(ann["bbox"]),
                area=ann.get("area", 0),
                iscrowd=ann.get("iscrowd", 0),
            ))
        records.append(CocoRecord(
            image_id=img_id,
            file_name=img_info["file_name"],
            image_path=images_dir / img_info["file_name"],
            width=img_info["width"],
            height=img_info["height"],
            objects=tuple(objects),
        ))
    return records


# ── bbox conversion ───────────────────────────────────────────

def coco_bbox_to_yolo(
    bbox_xywh: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, w, h] (absolute, top-left) → YOLO [xc, yc, w, h] (normalized)."""
    x, y, w, h = bbox_xywh
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return xc, yc, nw, nh


# ── label export ──────────────────────────────────────────────

def make_label_lines(
    record: CocoRecord,
    allowed_classes: Optional[set[str]] = None,
    skip_crowd: bool = True,
) -> list[str]:
    lines = []
    for obj in record.objects:
        if skip_crowd and obj.iscrowd:
            continue
        if allowed_classes is not None and obj.name not in allowed_classes:
            continue
        xc, yc, nw, nh = coco_bbox_to_yolo(obj.bbox_xywh, record.width, record.height)
        lines.append(f"{obj.class_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return lines


def export_records(
    records: list[CocoRecord],
    output_root: Path,
    split_name: str,
    allowed_classes: Optional[set[str]] = None,
    skip_empty_labels: bool = False,
    symlink: bool = True,
) -> Path:
    """Export COCO records to YOLO txt format.

    Uses symlinks for images by default to save disk space.
    """
    images_dir = output_root / "images" / split_name
    labels_dir = output_root / "labels" / split_name
    manifests_dir = output_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []

    for record in records:
        label_lines = make_label_lines(record, allowed_classes=allowed_classes)
        if skip_empty_labels and not label_lines:
            continue

        stem = str(record.image_id).zfill(12)
        img_target = images_dir / f"{stem}.jpg"
        lbl_target = labels_dir / f"{stem}.txt"

        if not img_target.exists():
            src = record.image_path
            if symlink:
                try:
                    img_target.symlink_to(src)
                except OSError:
                    shutil.copy2(src, img_target)
            else:
                shutil.copy2(src, img_target)

        lbl_target.write_text("\n".join(label_lines), encoding="utf-8")
        manifest_lines.append(img_target.resolve().as_posix())

    manifest_path = manifests_dir / f"{split_name}.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest_path


# ── few-shot sampling ─────────────────────────────────────────

def count_instances(record: CocoRecord, class_name: str) -> int:
    return sum(1 for obj in record.objects if obj.name == class_name and not obj.iscrowd)


def build_fewshot_records(
    records: list[CocoRecord],
    novel_classes: list[str],
    shot: int,
    seed: int,
) -> tuple[list[CocoRecord], dict[str, int]]:
    """Sample K-shot support set — same logic as VOC version."""
    rng = random.Random(seed)
    selected_keys: set[int] = set()

    for cls in novel_classes:
        candidates = [(r, count_instances(r, cls)) for r in records if count_instances(r, cls) > 0]
        rng.shuffle(candidates)

        collected = 0
        for record, cnt in candidates:
            if collected >= shot:
                break
            selected_keys.add(record.key)
            collected += cnt

        if collected == 0:
            raise ValueError(f"No samples found for novel class: {cls}")

    selected = [r for r in records if r.key in selected_keys]
    stats = {cls: sum(count_instances(r, cls) for r in selected) for cls in novel_classes}
    return selected, stats
