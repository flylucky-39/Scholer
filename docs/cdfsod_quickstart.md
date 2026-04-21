# CD-FSOD Quickstart

This document walks through running the CD-FSOD experiments described in
[`proposal_cdfsod_realtime.md`](proposal_cdfsod_realtime.md).

## 0. Pre-requisites

* A source-domain pretrain checkpoint (e.g. COCO-base trained earlier with
  `scripts/train_baseline.py` or `scripts/train_fsod.py --stage base`).
* The CD-FSOD-Bench data laid out as
  `<cdfsod_root>/<dataset>/{annotations,images}/...`. See
  https://github.com/lovelyqian/CDFSOD-benchmark for the canonical release.
* (Optional) `open_clip` installed for a real Domain-Gap Estimator backend:
  `pip install open_clip_torch`. Without it, the script falls back to a
  simple colour-histogram + random projection (dev-only).
* (Optional) Florence-2 weights for text prototypes; otherwise pass
  `--no-text` to disable the VLM branch.

## 1. Prepare a target domain

```bash
python scripts/prepare_cdfsod.py --config configs/cdfsod_DIOR_10shot.yaml
```

This writes `data/cdfsod_DIOR_10shot/` with `images/`, `labels/`,
`manifests/`, and a YOLO-compatible `cdfsod_bench_finetune.yaml`.

## 2. Visual-only baseline (no VLM, no DGE, no CDPC)

```bash
python scripts/train_cdfsod.py \
    --config configs/cdfsod_DIOR_10shot.yaml \
    --base-weights runs/coco_fsod_30shot/base_pretrain/weights/best.pt \
    --no-text
```

Output: `runs/cdfsod_DIOR_10shot/cdfsod_DIOR_10shot_visual_only/`

## 3. Full method (DGE + DAF + CDPC + text prototypes)

```bash
python scripts/train_cdfsod.py \
    --config configs/cdfsod_DIOR_10shot.yaml \
    --base-weights runs/coco_fsod_30shot/base_pretrain/weights/best.pt
```

The script logs the estimated domain gap `g`, the resulting fusion weight
`α(g)`, and writes a `cdfsod_summary.yaml` next to the run for reproducibility.

## 4. Suggested ablation

| Variant | Flags |
| --- | --- |
| (a) visual only | `--no-text` |
| (b) text + fixed α=0.5 | `--no-dge` (uses `alpha_default`) |
| (c) text + DGE only | `--no-cdpc` |
| (d) full DAF + CDPC | (no flags) |

## 5. Key paths the scripts expect

| Setting | Where defined | Default |
| --- | --- | --- |
| CD-FSOD-Bench root | `cdfsod_root` in config | `~/epfs/07_FSOD_LLM/datasets/cdfsod_bench` |
| Source images for DGE | `source_image_dir` or `source_image_manifest` | COCO `train2017` |
| Florence-2 weights | `florence2_model` | `~/epfs/07_FSOD_LLM/models/Florence-2-base/` |

## 6. What is *not* implemented yet

* CD-ViTO / DE-ViT replication (external comparisons).
* Multi-seed sweep automation — wrap `train_cdfsod.py` in a shell loop for now.
* mAP table aggregation across the 6 domains.

These are intentionally left out of the first commit; they can be added
without touching the new modules.
