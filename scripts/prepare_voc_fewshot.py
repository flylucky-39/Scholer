from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsod.voc import VOC_CLASSES, build_fewshot_records, collect_records, deduplicate_records, export_records, sample_base_replay_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VOC few-shot dataset for YOLO baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def write_dataset_yaml(dataset_yaml_path: Path, train_manifest: Path, val_manifest: Path, test_manifest: Path) -> None:
    dataset_yaml = {
        "path": dataset_yaml_path.parent.resolve().as_posix(),
        "train": train_manifest.resolve().as_posix(),
        "val": val_manifest.resolve().as_posix(),
        "test": test_manifest.resolve().as_posix(),
        "names": VOC_CLASSES,
    }
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_config(config_path)

    voc_root = resolve_repo_path(config["voc_root"])
    output_root = resolve_repo_path(config["output_root"])
    shot = int(config["shot"])
    seed = int(config["seed"])
    novel_classes = config["novel_classes"]

    if not (voc_root / "VOC2007").exists() or not (voc_root / "VOC2012").exists():
        raise FileNotFoundError(
            f"VOC root is invalid: {voc_root}. Expected VOC2007/ and VOC2012/ under this directory."
        )

    unknown_classes = [class_name for class_name in novel_classes if class_name not in VOC_CLASSES]
    if unknown_classes:
        raise ValueError(f"Unknown VOC classes in novel_classes: {unknown_classes}")

    base_classes = [class_name for class_name in VOC_CLASSES if class_name not in novel_classes]

    train_records = collect_records(voc_root, [("VOC2007", "trainval"), ("VOC2012", "trainval")])
    test_records = collect_records(voc_root, [("VOC2007", "test")])

    support_records, support_stats = build_fewshot_records(
        records=train_records,
        novel_classes=novel_classes,
        shot=shot,
        seed=seed,
    )

    replay_records = []
    replay_config = config.get("replay", {})
    if replay_config.get("enabled", False):
        replay_records = sample_base_replay_records(
            records=train_records,
            base_classes=base_classes,
            excluded_keys={record.key for record in support_records},
            sample_size=int(replay_config.get("base_images", 0)),
            seed=seed + 1,
        )

    finetune_records = deduplicate_records(support_records + replay_records)

    output_root.mkdir(parents=True, exist_ok=True)

    base_manifest = export_records(
        records=train_records,
        output_root=output_root,
        split_name="base_train",
        class_names=VOC_CLASSES,
        allowed_classes=set(base_classes),
    )
    finetune_manifest = export_records(
        records=finetune_records,
        output_root=output_root,
        split_name="novel_finetune",
        class_names=VOC_CLASSES,
        allowed_classes=set(novel_classes),
    )
    test_manifest = export_records(
        records=test_records,
        output_root=output_root,
        split_name="test",
        class_names=VOC_CLASSES,
        allowed_classes=set(VOC_CLASSES),
    )
    test_novel_manifest = export_records(
        records=test_records,
        output_root=output_root,
        split_name="test_novel",
        class_names=VOC_CLASSES,
        allowed_classes=set(novel_classes),
    )

    write_dataset_yaml(output_root / "voc_fsod_base.yaml", base_manifest, test_manifest, test_manifest)
    write_dataset_yaml(output_root / "voc_fsod_finetune.yaml", finetune_manifest, test_novel_manifest, test_novel_manifest)
    write_dataset_yaml(output_root / "voc_fsod_eval_all.yaml", finetune_manifest, test_manifest, test_manifest)

    metadata = {
        "seed": seed,
        "shot": shot,
        "novel_classes": novel_classes,
        "base_classes": base_classes,
        "finetune_label_scope": "novel_only",
        "train_records": len(train_records),
        "test_records": len(test_records),
        "support_records": len(support_records),
        "replay_records": len(replay_records),
        "finetune_records": len(finetune_records),
        "test_label_scope": "all",
        "finetune_val_scope": "novel_only",
        "selected_novel_instances": support_stats,
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved dataset files to: {output_root.resolve()}")


if __name__ == "__main__":
    main()