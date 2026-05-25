#!/bin/bash
# Run VOC FSOD experiments: Cosine + Fuse (Florence-2 FiLM) + CV2 frozen + mosaic disabled
# Usage:
#   bash scripts/run_voc_freeze_cv2_fused.sh          # run both 1-shot and 10-shot
#   bash scripts/run_voc_freeze_cv2_fused.sh 1         # run only 1-shot
#   bash scripts/run_voc_freeze_cv2_fused.sh 10        # run only 10-shot
#   nohup bash scripts/run_voc_freeze_cv2_fused.sh > logs/run_freeze_cv2_fused.log 2>&1 &  # background
#
# Prerequisites:
#   - VOC data prepared (scripts/prepare_voc_fewshot.py already ran)
#   - yolo11s.pt exists in project root
#   - Florence-2 model at ~/epfs/07_FSOD_LLM/models/Florence-2-base/
#   - cd to project root (/root/epfs/07_FSOD_LLM/fsod)

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
LOG_DIR=logs
mkdir -p "$LOG_DIR"

FLORENCE2_PATH=~/epfs/07_FSOD_LLM/models/Florence-2-base/
FUSE_RUN_NAME="novel_finetune_cosine_fused_cv2freeze_nomosaic"

echo "=============================================="
echo "VOC FSOD: Cosine+Fuse + CV2 Frozen + Mosaic Disabled"
echo "Project root: $PROJECT_ROOT"
echo "Florence-2: $FLORENCE2_PATH"
echo "Start time: $(date)"
echo "=============================================="

# Determine which shots to run
RUN_ALL=true
if [ $# -ge 1 ]; then
    RUN_ALL=false
fi

run_experiment() {
    local SHOT=$1
    local CFG="configs/baseline_voc_${SHOT}shot_freeze_cv2_nomosaic.yaml"
    local RUN_DIR="runs/voc_fsod_freeze_cv2_${SHOT}shot"
    local LOG="${LOG_DIR}/freeze_cv2_fused_${SHOT}shot.log"

    echo ""
    echo "========== VOC ${SHOT}-shot (Cosine+Fuse+FreezeCV2+NoMosaic) =========="
    echo "Config: $CFG"
    echo "Run dir: $RUN_DIR"
    echo "Log: $LOG"
    echo ""

    # Check if finetune already done
    if [ -f "$RUN_DIR/${FUSE_RUN_NAME}/weights/best.pt" ]; then
        echo "[SKIP] Finetune already completed for ${SHOT}-shot"
        echo "Weights: $RUN_DIR/${FUSE_RUN_NAME}/weights/best.pt"
        return 0
    fi

    # Step 1: Finetune with cosine + fuse + cv2 frozen + mosaic disabled
    echo "--- Step 1: Finetune (cosine+fuse, cv2 frozen, mosaic=0.0, base_weights=base_pretrain/weights/best.pt, ${SHOT}-shot) ---"
    PYTHONUNBUFFERED=1 python scripts/train_fsod_freeze_cv2_patched.py \
        --config "$CFG" \
        --stage finetune \
        --base-weights base_pretrain/weights/best.pt \
        --prototype \
        --florence2 "$FLORENCE2_PATH" \
        2>&1 | tee -a "$LOG"
    echo "[done] Finetune complete"

    # Step 2: Evaluate
    echo "--- Step 2: Evaluate (${SHOT}-shot) ---"
    python scripts/eval_fsod.py \
        --config "$CFG" \
        --run-name "${FUSE_RUN_NAME}" \
        2>&1 | tee -a "$LOG"

    echo "[done] Evaluation complete"
    echo "========== VOC ${SHOT}-shot done =========="
}

# Run experiments
if [ "$RUN_ALL" = true ]; then
    run_experiment 1
    run_experiment 10
else
    for SHOT in "$@"; do
        run_experiment "$SHOT"
    done
fi

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "End time: $(date)"
echo "=============================================="
