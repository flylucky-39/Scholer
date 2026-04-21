"""Prepare a CD-FSOD-Bench target domain for YOLO few-shot training.

CD-FSOD-Bench layout (lovelyqian/CDFSOD-benchmark):
    <ROOT>/<DATASET>/
        annotations/{train,val,test}.json   (COCO format)
        images/{train,val,test}/*.jpg

This script converts a single dataset into the YOLO few-shot layout used by
``scripts/train_fsod.py``::

    <output_root>/
        images/
            base_train/   (target-domain train pool, optional fine-tune-base)
            novel_finetune/ (K-shot subset, the actual FSOD support)
            val/
            test/
        labels/...
        manifests/{base_train,novel_finetune,val,test}.txt
        cdfsod_bench_finetune.yaml
        cdfsod_bench_base.yaml         (only if --include-base-pool)

Notes
-----
* CD-FSOD treats the target domain's *train* split as the few-shot support
  pool. We sample K instances per class, deterministically by seed.
* The source domain (e.g. COCO base pretrain) is NOT touched here; it is
  produced separately by ``scripts/train_baseline.py`` / ``train_fsod.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsod.cdfsod import get_dataset_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a CD-FSOD-Bench dataset for YOLO few-shot training.")
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_coco_split(json_path: Path) -> tuple[list[dict], list[dict], dict[int, str]]:
    """Return (images, annotations, cat_id→name) from a COCO-style json."""
    if not json_path.exists():
        raise FileNotFoundError(f"COCO annotation json not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return coco.get("images", []), coco.get("annotations", []), cats


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Convert COCO [x, y, w, h] (top-left, abs) to YOLO [cx, cy, w, h] (norm)."""
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        return None
    return cx, cy, nw, nh


def write_split(
    *,
    split_name: str,
    images: list[dict],
    annotations: list[dict],
    cat_id_to_name: dict[int, str],
    name_to_idx: dict[str, int],
    src_image_dir: Path,
    out_root: Path,
    image_subset: set[int] | None = None,
) -> list[Path]:
    """Copy images + write YOLO labels for a split. Returns list of relative image paths."""
    img_dir = out_root / "images" / split_name
    lbl_dir = out_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img_lookup = {im["id"]: im for im in images}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    written: list[Path] = []
    target_ids = image_subset if image_subset is not None else set(img_lookup)

    for img_id in target_ids:
        im = img_lookup.get(img_id)
        if im is None:
            continue
        file_name = im["file_name"]
        src_path = src_image_dir / file_name
        if not src_path.exists():
            continue
        dst_path = img_dir / Path(file_name).name
        if not dst_path.exists():
            shutil.copyfile(src_path, dst_path)

        lbl_lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name not in name_to_idx:
                continue
            yolo_box = coco_bbox_to_yolo(ann["bbox"], im["width"], im["height"])
            if yolo_box is None:
                continue
            cx, cy, w, h = yolo_box
            lbl_lines.append(f"{name_to_idx[cat_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (lbl_dir / (Path(file_name).stem + ".txt")).write_text(
            "\n".join(lbl_lines) + ("\n" if lbl_lines else ""),
            encoding="utf-8",
        )
        written.append(dst_path.resolve())

    return written


def sample_kshot_image_ids(
    annotations: list[dict],
    cat_id_to_name: dict[int, str],
    target_classes: set[str],
    shots: int,
    seed: int,
) -> set[int]:
    """Sample image IDs that cover at least `shots` instances per class.

    Greedy — sort images by # target instances they contain, pick until each
    class hits the quota. This mirrors the standard CD-FSOD-Bench protocol.
    """
    rng = Random(seed)

    img_to_classes: dict[int, list[str]] = defaultdict(list)
    for ann in annotations:
        name = cat_id_to_name.get(ann["category_id"])
        if name in target_classes:
            img_to_classes[ann["image_id"]].append(name)

    candidate_ids = list(img_to_classes)
    rng.shuffle(candidate_ids)

    counts = {c: 0 for c in target_classes}
    chosen: set[int] = set()
    for img_id in candidate_ids:
        if all(counts[c] >= shots for c in target_classes):
            break
        adds_useful = any(counts[c] < shots for c in img_to_classes[img_id])
        if not adds_useful:
            continue
        chosen.add(img_id)
        for c in img_to_classes[img_id]:
            counts[c] += 1

    missing = [c for c, v in counts.items() if v < shots]
    if missing:
        print(f"  WARNING: insufficient instances to satisfy {shots}-shot for: {missing} (got {[counts[c] for c in missing]})")
    return chosen


def write_dataset_yaml(yaml_path: Path, out_root: Path, manifests: dict[str, Path], names: list[str]) -> None:
    payload = {
        "path": out_root.resolve().as_posix(),
        "train": manifests["train"].resolve().as_posix(),
        "val": manifests["val"].resolve().as_posix(),
        "test": manifests["test"].resolve().as_posix(),
        "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_manifest(manifest_path: Path, paths: list[Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bench_root = resolve_repo_path(cfg["cdfsod_root"])
    dataset_name = cfg["cdfsod_dataset"]
    spec = get_dataset_spec(dataset_name)
    out_root = resolve_repo_path(cfg["output_root"])
    shots = int(cfg["shot"])
    seed = int(cfg["seed"])
    include_base_pool = bool(cfg.get("include_base_pool", False))

    ds_root = bench_root / spec.expected_dir
    if not ds_root.exists():
        raise FileNotFoundError(f"CD-FSOD dataset directory not found: {ds_root}")

    name_to_idx = {n: i for i, n in enumerate(spec.classes)}

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Preparing CD-FSOD dataset {spec.name} ({spec.num_classes} classes) → {out_root}")

    # ----- TRAIN POOL → sample K-shot --------------------------------------
    train_json = ds_root / "annotations" / "train.json"
    train_imgs_dir = _resolve_split_image_dir(ds_root, "train")
    images_train, anns_train, cats_train = load_coco_split(train_json)

    target_class_set = set(spec.classes)
    if not (set(cats_train.values()) & target_class_set):
        raise RuntimeError(
            f"None of the spec classes {spec.classes} appeared in {train_json}. "
            f"Found: {sorted(set(cats_train.values()))}"
        )

    chosen_ids = sample_kshot_image_ids(anns_train, cats_train, target_class_set, shots, seed)
    print(f"  K={shots}-shot support: {len(chosen_ids)} images")

    novel_paths = write_split(
        split_name="novel_finetune",
        images=images_train,
        annotations=anns_train,
        cat_id_to_name=cats_train,
        name_to_idx=name_to_idx,
        src_image_dir=train_imgs_dir,
        out_root=out_root,
        image_subset=chosen_ids,
    )

    base_paths: list[Path] = []
    if include_base_pool:
        # Optional: keep the full train pool as base_train (for in-domain pretrain experiments).
        base_paths = write_split(
            split_name="base_train",
            images=images_train,
            annotations=anns_train,
            cat_id_to_name=cats_train,
            name_to_idx=name_to_idx,
            src_image_dir=train_imgs_dir,
            out_root=out_root,
        )
        print(f"  base_train pool: {len(base_paths)} images")

    # ----- VAL --------------------------------------------------------------
    val_json = ds_root / "annotations" / "val.json"
    val_imgs_dir = _resolve_split_image_dir(ds_root, "val")
    if val_json.exists():
        images_val, anns_val, cats_val = load_coco_split(val_json)
        val_paths = write_split(
            split_name="val",
            images=images_val,
            annotations=anns_val,
            cat_id_to_name=cats_val,
            name_to_idx=name_to_idx,
            src_image_dir=val_imgs_dir,
            out_root=out_root,
        )
    else:
        print("  No val.json — re-using novel_finetune as val (limited evaluation)")
        val_paths = novel_paths

    # ----- TEST -------------------------------------------------------------
    test_json = ds_root / "annotations" / "test.json"
    test_imgs_dir = _resolve_split_image_dir(ds_root, "test")
    if test_json.exists():
        images_test, anns_test, cats_test = load_coco_split(test_json)
        test_paths = write_split(
            split_name="test",
            images=images_test,
            annotations=anns_test,
            cat_id_to_name=cats_test,
            name_to_idx=name_to_idx,
            src_image_dir=test_imgs_dir,
            out_root=out_root,
        )
    else:
        print("  No test.json — re-using val as test")
        test_paths = val_paths

    # ----- MANIFESTS --------------------------------------------------------
    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifests_dir / "novel_finetune.txt", novel_paths)
    write_manifest(manifests_dir / "val.txt", val_paths)
    write_manifest(manifests_dir / "test.txt", test_paths)
    if base_paths:
        write_manifest(manifests_dir / "base_train.txt", base_paths)

    # ----- DATASET YAMLs ---------------------------------------------------
    finetune_yaml = out_root / "cdfsod_bench_finetune.yaml"
    write_dataset_yaml(
        finetune_yaml,
        out_root,
        manifests={
            "train": manifests_dir / "novel_finetune.txt",
            "val": manifests_dir / "val.txt",
            "test": manifests_dir / "test.txt",
        },
        names=list(spec.classes),
    )
    print(f"  finetune yaml: {finetune_yaml}")

    if base_paths:
        base_yaml = out_root / "cdfsod_bench_base.yaml"
        write_dataset_yaml(
            base_yaml,
            out_root,
            manifests={
                "train": manifests_dir / "base_train.txt",
                "val": manifests_dir / "val.txt",
                "test": manifests_dir / "test.txt",
            },
            names=list(spec.classes),
        )
        print(f"  base yaml: {base_yaml}")

    print("Done.")


def _resolve_split_image_dir(ds_root: Path, split: str) -> Path:
    """CD-FSOD-Bench releases vary slightly. Try common layouts."""
    candidates = [
        ds_root / "images" / split,
        ds_root / split,
        ds_root / split / "images",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the first guess; write_split() will gracefully skip missing files.
    return candidates[0]


if __name__ == "__main__":
    main()
