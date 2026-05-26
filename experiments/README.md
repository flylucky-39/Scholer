# Experiments for "Suppressing False Positives in FSOD"

## Overview

Enhance the published baseline (conference paper) with three complementary techniques for false positive suppression, forming a substantial extension for journal submission.

### Techniques

| # | Technique | Status | Component |
|---|-----------|--------|-----------|
| 1 | **Scale Regularization** (v0.1.0) | ✅ Done | L2 penalty on `scale.exp()` in CosineConv2d |
| 2 | **Per-class Temperature** | ✅ Code exists | Per-class temperature instead of global |
| 3 | **Background Suppression Loss** | ❌ To implement | Penalize high-confidence predictions on negative anchors |

### Paper Structure (proposed)

```
1. Introduction
2. Related Work
3. Baseline: Cosine-Classifier FSOD (conference paper summary)
4. Proposed Method
   4.1 Scale Regularization (global temperature)
   4.2 Per-class Temperature
   4.3 Background Suppression Loss
   4.4 Combined Objective
5. Experiments
   5.1 VOC: 3 splits × 1/5/10-shot
   5.2 COCO: 10/30-shot
   5.3 Ablation Studies
   5.4 Analysis (P-R curves, scale evolution, FP reduction)
6. Conclusion
```

---

## Series 1: Scale Regularization (扩展 v0.1.0)

**Purpose**: Validate scale regularization across diverse settings.

### S1.1 VOC 3 splits × 3 shots

| Run ID | Split | Shot | Command |
|--------|-------|------|---------|
| s1.1.1 | split1 | 10-shot | `--config configs/baseline_voc_10shot_scalereg.yaml --stage all --reg-weight 0.05` |
| s1.1.2 | split1 | 5-shot | (create config from `baseline_voc_5shot.yaml`) |
| s1.1.3 | split1 | 1-shot | (create config from `baseline_voc_1shot.yaml`) |
| s1.1.4 | split2 | 10-shot | (create config from `baseline_voc_10shot_split2_*.yaml`) |
| s1.1.5 | split2 | 5-shot | — |
| s1.1.6 | split3 | 10-shot | — |
| s1.1.7 | split3 | 5-shot | — |

### S1.2 reg_weight ablation

| Run ID | reg_weight | Config |
|--------|------------|--------|
| s1.2.1 | 0.01 | split1 10-shot |
| s1.2.2 | 0.05 | ✅ done (v0.1.0) |
| s1.2.3 | 0.10 | split1 10-shot |
| s1.2.4 | 0.50 | split1 10-shot |

### S1.3 Scale evolution logging

Add TensorBoard or CSV logging of `scale.exp()` per epoch. Confirms that regularization actually suppresses temperature.

---

## Series 2: Per-class Temperature

**Purpose**: Let each class learn its own temperature for finer P/R control.

**Code**: Already in `cosine_head.py:39-40` (commit `1f7b258`). `scale` changed from `(1,)` to `(C_out,)`.

### S2.1 Per-class temp + scale reg combined

| Run ID | Config | Notes |
|--------|--------|-------|
| s2.1.1 | split1 10-shot w/ per-class temp + reg=0.05 | Both techniques |
| s2.1.2 | split1 10-shot w/ per-class temp (no reg) | Per-class temp alone |

### S2.2 Per-class scale visualization

Plot the final per-class `scale.exp()` values. Shows which classes need higher/lower temperature.

---

## Series 3: Background Suppression Loss

**Purpose**: Directly penalize high-confidence predictions on background/negative anchors.

### Implementation sketch

In `ScaleRegDetectionLoss.__call__()`, after computing BCE loss, add:

```python
# Identify negative anchors (target_scores == 0 for all classes)
target_scores_sum = target_scores.sum(dim=-1)  # (n_anchors,)
neg_mask = target_scores_sum == 0

if neg_mask.any():
    pred_scores_neg = pred_scores[neg_mask]  # scores on negative anchors
    # Penalize high confidence on negatives
    bg_loss = bg_weight * pred_scores_neg.max(dim=-1).values.mean()
    loss = loss + bg_loss
```

### S3.1 bg_loss weight sweep

| Run ID | bg_weight | Config |
|--------|-----------|--------|
| s3.1.1 | 0.01 | split1 10-shot |
| s3.1.2 | 0.05 | split1 10-shot |
| s3.1.3 | 0.10 | split1 10-shot |

### S3.2 Combined: bg_loss + scale_reg

| Run ID | Components | Config |
|--------|-----------|--------|
| s3.2.1 | bg=0.05 + reg=0.05 | split1 10-shot |

---

## Series 4: Full Method (Final)

Combine all three techniques + best hyper-parameters from ablation.

| Run ID | Split | Shot | reg | per-class | bg |
|--------|-------|------|-----|-----------|----|
| s4.1 | split1 | 10-shot | ✓ | ✓ | ✓ |
| s4.2 | split2 | 10-shot | ✓ | ✓ | ✓ |
| s4.3 | split3 | 10-shot | ✓ | ✓ | ✓ |
| s4.4 | split1 | 5-shot | ✓ | ✓ | ✓ |
| s4.5 | split2 | 5-shot | ✓ | ✓ | ✓ |
| s4.6 | split3 | 5-shot | ✓ | ✓ | ✓ |
| s4.7 | split1 | 1-shot | ✓ | ✓ | ✓ |
| s4.8 | COCO | 10-shot | ✓ | ✓ | ✓ |
| s4.9 | COCO | 30-shot | ✓ | ✓ | ✓ |

---

## Summary Table

| Series | Runs | Purpose | Est. Time |
|--------|------|---------|-----------|
| S1.1 | 7 | VOC 3 splits × shots validation | ~14h |
| S1.2 | 3 | reg_weight ablation | ~6h |
| S2.1 | 2 | Per-class temp eval | ~4h |
| S3.1 | 3 | bg_loss sweep | ~6h |
| S3.2 | 1 | Combined bg + reg | ~2h |
| S4 | 9 | Full method all settings | ~20h |
| **Total** | **25** | | **~52h (2d)** |

### Priority order

```
Phase 1 (foundation):
  ├── S1.1 VOC 3 splits × 10-shot (6 runs, ~12h)
  ├── S2.1 Per-class temp (2 runs, ~4h)
  └── S3.1 bg_loss sweep (3 runs, ~6h)

Phase 2 (ablation):
  ├── S1.2 reg_weight sweep (3 runs, ~6h)
  └── S3.2 bg + reg combined (1 run, ~2h)

Phase 3 (final + paper):
  └── S4 Full method (9 runs, ~20h)
```

---

## Run Template

```bash
# Standard run command template
python scripts/train_fsod_scalereg.py \
  --config experiments/<config>.yaml \
  --stage finetune \
  --reg-weight <REG> \
  --base-weights base_pretrain/weights/best.pt \
  --epochs 200 \
  --prototype \
  --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/ \
  > experiments/series_XX/<run_name>.log 2>&1
```
