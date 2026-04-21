"""CD-FSOD-Bench dataset registry.

Provides class names and metadata for the 6 target domains in the
CD-FSOD-Bench (CVPR 2024 NTIRE Challenge). Layout assumes the
`lovelyqian/CDFSOD-benchmark` directory structure:

    <CDFSOD_ROOT>/
        ArTaxOr/{train,val,test}/{images,annotations.json}
        clipart1k/...
        DIOR/...
        DeepFish/...
        NEU-DET/...
        UODD/...

This module only declares the spec; actual conversion to YOLO format is
done by ``scripts/prepare_cdfsod.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    classes: tuple[str, ...]
    description: str
    expected_dir: str

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# Class lists below follow the standard CD-FSOD-Bench release.
CDFSOD_DATASETS: dict[str, DatasetSpec] = {
    "ArTaxOr": DatasetSpec(
        name="ArTaxOr",
        classes=(
            "Araneae", "Coleoptera", "Diptera", "Hemiptera",
            "Hymenoptera", "Lepidoptera", "Odonata",
        ),
        description="Arthropod taxonomy ordered fine-grained classification.",
        expected_dir="ArTaxOr",
    ),
    "Clipart1k": DatasetSpec(
        name="Clipart1k",
        classes=(
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ),
        description="VOC-style clipart drawings (large semantic gap, photo→clipart).",
        expected_dir="clipart1k",
    ),
    "DIOR": DatasetSpec(
        name="DIOR",
        classes=(
            "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
            "chimney", "dam", "expressway-service-area", "expressway-toll-station",
            "golffield", "groundtrackfield", "harbor", "overpass", "ship",
            "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
        ),
        description="Aerial / remote-sensing object detection.",
        expected_dir="DIOR",
    ),
    "DeepFish": DatasetSpec(
        name="DeepFish",
        classes=("fish",),
        description="Underwater single-class fish detection.",
        expected_dir="DeepFish",
    ),
    "NEU-DET": DatasetSpec(
        name="NEU-DET",
        classes=(
            "crazing", "inclusion", "patches", "pitted_surface",
            "rolled-in_scale", "scratches",
        ),
        description="Steel-surface defect inspection.",
        expected_dir="NEU-DET",
    ),
    "UODD": DatasetSpec(
        name="UODD",
        classes=("seacucumber", "seaurchin", "scallop"),
        description="Underwater object detection (sea creatures).",
        expected_dir="UODD",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Look up a dataset spec by name (case-insensitive)."""
    key = name.strip()
    if key in CDFSOD_DATASETS:
        return CDFSOD_DATASETS[key]
    lowered = {k.lower(): k for k in CDFSOD_DATASETS}
    if key.lower() in lowered:
        return CDFSOD_DATASETS[lowered[key.lower()]]
    raise KeyError(
        f"Unknown CD-FSOD dataset '{name}'. "
        f"Available: {sorted(CDFSOD_DATASETS)}"
    )
