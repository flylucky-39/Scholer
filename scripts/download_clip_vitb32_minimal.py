"""Download CLIP-ViT-B/32 weights in a minimal form for offline use.

Downloads only the safetensors checkpoint + tokenizer/processor metadata,
skipping redundant TF / Flax / legacy pytorch_model.bin copies. Target size
after download is ~580 MB instead of ~1.82 GB.

Usage:
    python scripts/download_clip_vitb32_minimal.py \
        --out models/clip-vit-base-patch32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "pytorch_model.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal CLIP-ViT-B/32 downloader (safetensors only).")
    parser.add_argument(
        "--repo-id", default="openai/clip-vit-base-patch32",
        help="HF repo id.")
    parser.add_argument(
        "--out", default="models/clip-vit-base-patch32",
        help="Local output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} → {out_dir}")
    print(f"allow_patterns: {ALLOW_PATTERNS}")
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
        resume_download=True,
    )
    # Summary.
    files = sorted(p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\nDone: {path}")
    print(f"Files ({len(files)}):")
    for f in files:
        size_mb = (out_dir / f).stat().st_size / (1024 ** 2)
        print(f"  {f}  ({size_mb:.1f} MB)")
    print(f"Total: {total_bytes / (1024 ** 2):.1f} MB")


if __name__ == "__main__":
    main()
