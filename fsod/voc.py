from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass(frozen=True)
class VocObject:
    name: str
    bbox: tuple[int, int, int, int]
    difficult: int = 0


@dataclass(frozen=True)
class VocRecord:
    year: str
    image_id: str
    image_path: Path
    width: int
    height: int
    objects: tuple[VocObject, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.year, self.image_id)


def load_image_ids(voc_root: Path, year: str, split: str) -> list[str]:
    split_file = voc_root / year / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation(voc_root: Path, year: str, image_id: str) -> VocRecord:
    xml_path = voc_root / year / "Annotations" / f"{image_id}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    image_path = voc_root / year / "JPEGImages" / f"{image_id}.jpg"

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", default="0"))
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.findtext("xmin")))
        ymin = int(float(bndbox.findtext("ymin")))
        xmax = int(float(bndbox.findtext("xmax")))
        ymax = int(float(bndbox.findtext("ymax")))
        objects.append(VocObject(name=name, bbox=(xmin, ymin, xmax, ymax), difficult=difficult))

    return VocRecord(
        year=year,
        image_id=image_id,
        image_path=image_path,
        width=width,
        height=height,
        objects=tuple(objects),
    )


def collect_records(voc_root: Path, year_splits: list[tuple[str, str]]) -> list[VocRecord]:
    records = []
    for year, split in year_splits:
        for image_id in load_image_ids(voc_root, year, split):
            records.append(parse_annotation(voc_root, year, image_id))
    return records


def count_instances(record: VocRecord, class_name: str, include_difficult: bool = False) -> int:
    count = 0
    for obj in record.objects:
        if obj.name != class_name:
            continue
        if obj.difficult and not include_difficult:
            continue
        count += 1
    return count


def bbox_to_yolo(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def make_label_lines(
    record: VocRecord,
    class_to_idx: dict[str, int],
    allowed_classes: set[str] | None = None,
    include_difficult: bool = False,
) -> list[str]:
    lines = []
    for obj in record.objects:
        if obj.difficult and not include_difficult:
            continue
        if allowed_classes is not None and obj.name not in allowed_classes:
            continue
        if obj.name not in class_to_idx:
            continue
        xc, yc, bw, bh = bbox_to_yolo(obj.bbox, record.width, record.height)
        lines.append(f"{class_to_idx[obj.name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def export_records(
    records: list[VocRecord],
    output_root: Path,
    split_name: str,
    class_names: list[str],
    allowed_classes: set[str] | None = None,
    include_difficult: bool = False,
) -> Path:
    images_dir = output_root / "images" / split_name
    labels_dir = output_root / "labels" / split_name
    manifests_dir = output_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    manifest_lines = []

    for record in records:
        image_target = images_dir / f"{record.year}_{record.image_id}.jpg"
        label_target = labels_dir / f"{record.year}_{record.image_id}.txt"

        if not image_target.exists():
            shutil.copy2(record.image_path, image_target)

        label_lines = make_label_lines(
            record=record,
            class_to_idx=class_to_idx,
            allowed_classes=allowed_classes,
            include_difficult=include_difficult,
        )
        label_target.write_text("\n".join(label_lines), encoding="utf-8")
        manifest_lines.append(image_target.resolve().as_posix())

    manifest_path = manifests_dir / f"{split_name}.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest_path


def build_fewshot_records(
    records: list[VocRecord],
    novel_classes: list[str],
    shot: int,
    seed: int,
) -> tuple[list[VocRecord], dict[str, int]]:
    random_generator = random.Random(seed)
    selected_keys: set[tuple[str, str]] = set()

    for class_name in novel_classes:
        candidates = [(record, count_instances(record, class_name)) for record in records]
        candidates = [(record, count) for record, count in candidates if count > 0]
        random_generator.shuffle(candidates)

        collected = 0
        for record, instance_count in candidates:
            if collected >= shot:
                break
            selected_keys.add(record.key)
            collected += instance_count

        if collected == 0:
            raise ValueError(f"No samples found for novel class: {class_name}")

    selected_records = [record for record in records if record.key in selected_keys]

    final_stats = {}
    for class_name in novel_classes:
        final_stats[class_name] = sum(count_instances(record, class_name) for record in selected_records)

    return selected_records, final_stats


def sample_base_replay_records(
    records: list[VocRecord],
    base_classes: list[str],
    excluded_keys: set[tuple[str, str]],
    sample_size: int,
    seed: int,
) -> list[VocRecord]:
    base_class_set = set(base_classes)
    candidates = []
    for record in records:
        if record.key in excluded_keys:
            continue
        if any(obj.name in base_class_set and obj.difficult == 0 for obj in record.objects):
            candidates.append(record)

    random_generator = random.Random(seed)
    random_generator.shuffle(candidates)
    return candidates[:sample_size]


def deduplicate_records(records: list[VocRecord]) -> list[VocRecord]:
    deduplicated = {}
    for record in records:
        deduplicated[record.key] = record
    return list(deduplicated.values())