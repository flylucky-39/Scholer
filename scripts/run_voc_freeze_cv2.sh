#!/bin/bash
# Run VOC FSOD experiments with CV2 frozen + mosaic disabled
# Usage:
#   bash scripts/run_voc_freeze_cv2.sh          # run both 1-shot and 10-shot
#   bash scripts/run_voc_freeze_cv2.sh 1         # run only 1-shot
#   bash scripts/run_voc_freeze_cv2.sh 10        # run only 10-shot
#   nohup bash scripts/run_voc_freeze_cv2.sh > logs/run_freeze_cv2.log 2>&1 &  # background
#
# Prerequisites:
#   - VOC data prepared (scripts/prepare_voc_fewshot.py already ran)
#   - yolo11s.pt exists in project root
#   - cd to project root (/root/epfs/07_FSOD_LLM/fsod)

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
LOG_DIR=logs
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "VOC FSOD: CV2 Frozen + Mosaic Disabled"
echo "Project root: $PROJECT_ROOT"
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
    local LOG="${LOG_DIR}/freeze_cv2_${SHOT}shot.log"

    echo ""
    echo "========== VOC ${SHOT}-shot =========="
    echo "Config: $CFG"
    echo "Run dir: $RUN_DIR"
    echo "Log: $LOG"
    echo ""

    # Check if finetune already done
    if [ -f "$RUN_DIR/novel_finetune_cosine_cv2freeze_nomosaic/weights/best.pt" ]; then
        echo "[SKIP] Finetune already completed for ${SHOT}-shot"
        echo "Weights: $RUN_DIR/novel_finetune_cosine_cv2freeze_nomosaic/weights/best.pt"
        return 0
    fi

    # Step 1: Base pretrain (if not exists)
    BASE_WEIGHTS="$RUN_DIR/base_pretrain/weights/best.pt"
    if [ ! -f "$BASE_WEIGHTS" ]; then
        echo "--- Step 1: Base pretrain (${SHOT}-shot) ---"
        python scripts/train_fsod_freeze_cv2.py \
            --config "$CFG" \
            --stage base \
            2>&1 | tee -a "$LOG"
        echo "[done] Base pretrain complete: $BASE_WEIGHTS"
    else
        echo "[skip] Base pretrain already exists: $BASE_WEIGHTS"
    fi

    # Step 2: Finetune with cv2 frozen + mosaic disabled
    echo "--- Step 2: Finetune (cv2 frozen, mosaic=0.0, ${SHOT}-shot) ---"
    python scripts/train_fsod_freeze_cv2.py \
        --config "$CFG" \
        --stage finetune \
        2>&1 | tee -a "$LOG"
    echo "[done] Finetune complete"

    # Step 3: Evaluate
    echo "--- Step 3: Evaluate (${SHOT}-shot) ---"
    python scripts/eval_fsod.py \
        --config "$CFG" \
        --run-name novel_finetune_cosine_cv2freeze_nomosaic \
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
