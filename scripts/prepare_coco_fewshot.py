"""Prepare COCO few-shot dataset for YOLO training.

Usage::

    python scripts/prepare_coco_fewshot.py --config configs/coco_10shot.yaml

Analogous to ``prepare_voc_fewshot.py`` but reads COCO JSON annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsod.coco import (
    COCO_CLASSES,
    COCO_NOVEL_NAMES,
    CocoRecord,
    build_fewshot_records,
    count_instances,
    export_records,
    load_coco_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COCO few-shot dataset for YOLO.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def write_dataset_yaml(
    yaml_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    class_names: list[str],
) -> None:
    doc = {
        "path": yaml_path.parent.resolve().as_posix(),
        "train": train_manifest.resolve().as_posix(),
        "val": val_manifest.resolve().as_posix(),
        "test": test_manifest.resolve().as_posix(),
        "names": class_names,
    }
    yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_config(config_path)

    coco_root = resolve_repo_path(config["coco_root"])
    output_root = resolve_repo_path(config["output_root"])
    shot = int(config["shot"])
    seed = int(config["seed"])
    novel_classes: list[str] = config.get("novel_classes", COCO_NOVEL_NAMES)
    base_classes = [c for c in COCO_CLASSES if c not in novel_classes]

    print(f"COCO root   : {coco_root}")
    print(f"Output root : {output_root}")
    print(f"Shot        : {shot}")
    print(f"Seed        : {seed}")
    print(f"Novel classes ({len(novel_classes)}): {novel_classes}")
    print(f"Base  classes ({len(base_classes)}): {base_classes[:5]}... (+{len(base_classes)-5})")

    # ── Load annotations ──────────────────────────────────────
    print("Loading train annotations …")
    train_records = load_coco_records(coco_root, split="train2017")
    print(f"  → {len(train_records)} train images")

    print("Loading val annotations …")
    val_records = load_coco_records(coco_root, split="val2017")
    print(f"  → {len(val_records)} val images")

    # ── Few-shot sampling ─────────────────────────────────────
    print(f"Sampling {shot}-shot support set …")
    support_records, support_stats = build_fewshot_records(
        records=train_records,
        novel_classes=novel_classes,
        shot=shot,
        seed=seed,
    )
    print("  Support set stats:")
    for cls, cnt in sorted(support_stats.items()):
        print(f"    {cls:20s} → {cnt} instances")

    # ── Export to YOLO format ─────────────────────────────────
    output_root.mkdir(parents=True, exist_ok=True)

    print("Exporting base train split (60 base classes, full data) …")
    base_manifest = export_records(
        records=train_records,
        output_root=output_root,
        split_name="base_train",
        allowed_classes=set(base_classes),
        skip_empty_labels=True,
    )

    print("Exporting novel finetune split (K-shot, novel classes only) …")
    finetune_manifest = export_records(
        records=support_records,
        output_root=output_root,
        split_name="novel_finetune",
        allowed_classes=set(novel_classes),
    )

    print("Exporting val split (all 80 classes) …")
    test_all_manifest = export_records(
        records=val_records,
        output_root=output_root,
        split_name="val",
        allowed_classes=None,
    )

    print("Exporting val split (novel 20 classes only) …")
    test_novel_manifest = export_records(
        records=val_records,
        output_root=output_root,
        split_name="val_novel",
        allowed_classes=set(novel_classes),
        skip_empty_labels=True,
    )

    # ── Write dataset YAMLs ───────────────────────────────────
    write_dataset_yaml(
        output_root / "coco_fsod_base.yaml",
        base_manifest, test_all_manifest, test_all_manifest,
        class_names=COCO_CLASSES,
    )
    write_dataset_yaml(
        output_root / "coco_fsod_finetune.yaml",
        finetune_manifest, test_novel_manifest, test_novel_manifest,
        class_names=COCO_CLASSES,
    )
    write_dataset_yaml(
        output_root / "coco_fsod_eval_all.yaml",
        finetune_manifest, test_all_manifest, test_all_manifest,
        class_names=COCO_CLASSES,
    )

    # ── Metadata ──────────────────────────────────────────────
    metadata = {
        "seed": seed,
        "shot": shot,
        "novel_classes": novel_classes,
        "base_classes": base_classes,
        "support_stats": support_stats,
        "train_images": len(train_records),
        "val_images": len(val_records),
    }
    meta_path = output_root / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadata saved to {meta_path}")
    print("Done!")


if __name__ == "__main__":
    main()
