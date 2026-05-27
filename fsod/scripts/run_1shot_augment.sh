#!/bin/bash
# Run 1-shot experiment with support set augmentation
# Usage: bash scripts/run_1shot_augment.sh

python scripts/train_fsod.py \
  --config configs/baseline_voc_1shot_augment.yaml \
  --stage finetune \
  --prototype \
  --augment \
  --augment-k 4 \
  --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/
