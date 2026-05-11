#!/bin/bash
# Phase 1: Cross-Dataset Validation Experiments
# =============================================
# Runs Clipart1k / NEU-DET / UODD at 1-shot and 10-shot
# Each dataset: prepare → train (visual-only + DAF) → eval (baseline + dual-path)
#
# Usage:
#   bash scripts/run_phase1.sh              # run all 3 datasets
#   bash scripts/run_phase1.sh Clipart1k    # run specific dataset
#   nohup bash scripts/run_phase1.sh > logs/phase1.log 2>&1 &   # background
#
# Assumes:
#   - cd to project root (/root/epfs/07_FSOD_LLM/fsod)
#   - COCO base weights at runs/coco_fsod_10shot_exp/base_pretrain/weights/best.pt
#   - CLIP model at models/clip-vit-base-patch32 (local files)
#   - conda env FSOD_LLM activated
#   - No internet (local_files_only for all HF models)
# =============================================

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_WEIGHTS=runs/coco_fsod_10shot_exp/base_pretrain/weights/best.pt
CLIP_MODEL=models/clip-vit-base-patch32
VLM_TEMP=5
GAMMA=0.5
LOG_DIR=logs

mkdir -p "$LOG_DIR" runs/vlm_assets

# Datasets to process (default: all 3 Phase 1 datasets)
DATASETS=("${@:-Clipart1k NEU-DET UODD}")

# Validate base weights exist
if [ ! -f "$BASE_WEIGHTS" ]; then
    echo "[ERROR] Base weights not found: $BASE_WEIGHTS"
    echo "Update BASE_WEIGHTS in this script to the correct path."
    exit 1
fi

echo "=============================================="
echo "Phase 1: Cross-Dataset Validation"
echo "Datasets: ${DATASETS[*]}"
echo "Base weights: $BASE_WEIGHTS"
echo "CLIP model: $CLIP_MODEL"
echo "=============================================="
echo ""
START_TS=$(date +%s)

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
for DS in "${DATASETS[@]}"; do
    # Lowercase for prompt filenames and tags
    DS_LOWER=$(echo "$DS" | tr '[:upper:]' '[:lower:]')
    PROTO=runs/vlm_assets/${DS_LOWER}_clip_vitb32_text_proto.pt

    echo ""
    echo "========== $DS =========="
    echo "Prompt file: prompts/${DS_LOWER}.json"
    echo "Text proto:  $PROTO"

    # ------------------------------------------------------------------
    # Step 1: Extract CLIP text prototypes (one-time per dataset)
    # ------------------------------------------------------------------
    if [ ! -f "$PROTO" ]; then
        echo "--- Step 1: Extracting text prototypes ---"
        python scripts/extract_vlm_text_proto.py \
            --dataset "$DS" \
            --prompts prompts/${DS_LOWER}.json \
            --model "$CLIP_MODEL" \
            --out "$PROTO" \
            2>&1 | tee -a "$LOG_DIR/step1_${DS_LOWER}.log"
        echo "[done] Text prototypes saved to $PROTO"
    else
        echo "[skip] Text prototypes already exist: $PROTO"
    fi

    # ------------------------------------------------------------------
    # Process 1-shot and 10-shot
    # ------------------------------------------------------------------
    for SHOT in 1 10; do
        CFG=configs/cdfsod_${DS}_${SHOT}shot.yaml
        TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

        echo ""
        echo "--- $DS $SHOT-shot ---"

        # ------------------------------------------------------------------
        # Step 2: Prepare data
        # ------------------------------------------------------------------
        echo "--- Step 2: Preparing $SHOT-shot data ---"
        python scripts/prepare_cdfsod.py --config "$CFG" \
            2>&1 | tee -a "$LOG_DIR/prepare_${DS_LOWER}_${SHOT}shot.log"
        echo "[done] Data prepared for $DS $SHOT-shot"

        # ------------------------------------------------------------------
        # Step 3a: Train Visual-Only
        # ------------------------------------------------------------------
        VIS_RUN=runs/cdfsod_${DS}_${SHOT}shot/cdfsod_${DS}_${SHOT}shot_visual_only
        VIS_WEIGHTS=$VIS_RUN/weights/best.pt

        if [ ! -f "$VIS_WEIGHTS" ]; then
            echo "--- Step 3a: Training Visual-Only ($SHOT-shot) ---"
            python scripts/train_cdfsod.py \
                --config "$CFG" \
                --base-weights "$BASE_WEIGHTS" \
                --no-text \
                2>&1 | tee -a "$LOG_DIR/train_${DS_LOWER}_${SHOT}shot_visualonly.log"
            echo "[done] Visual-Only training complete: $VIS_RUN"
        else
            echo "[skip] Visual-Only already trained: $VIS_WEIGHTS"
        fi

        # ------------------------------------------------------------------
        # Step 3b: Train DAF (with full DGE + DAF + CDPC pipeline)
        # ------------------------------------------------------------------
        # DAF run name is dynamic: cdfsod_{DS}_{K}shot_daf_a{alpha:.2f}
        # Use glob to find existing DAF run
        DAF_RUN_DIR=$(ls -d runs/cdfsod_${DS}_${SHOT}shot/cdfsod_${DS}_${SHOT}shot_daf_a* 2>/dev/null | head -1 || true)
        if [ -z "$DAF_RUN_DIR" ] || [ ! -f "$DAF_RUN_DIR/weights/best.pt" ]; then
            echo "--- Step 3b: Training DAF ($SHOT-shot) ---"
            python scripts/train_cdfsod.py \
                --config "$CFG" \
                --base-weights "$BASE_WEIGHTS" \
                2>&1 | tee -a "$LOG_DIR/train_${DS_LOWER}_${SHOT}shot_daf.log"
            DAF_RUN_DIR=$(ls -d runs/cdfsod_${DS}_${SHOT}shot/cdfsod_${DS}_${SHOT}shot_daf_a* 2>/dev/null | head -1)
            echo "[done] DAF training complete: $DAF_RUN_DIR"
        else
            echo "[skip] DAF already trained: $DAF_RUN_DIR"
        fi
        DAF_WEIGHTS=$DAF_RUN_DIR/weights/best.pt

        # ------------------------------------------------------------------
        # Step 4: Evaluate (4 combos per shot)
        # ------------------------------------------------------------------
        echo "--- Step 4: Evaluating ---"

        # 4a: Visual-Only baseline
        echo "[eval] Visual-Only baseline $SHOT-shot..."
        python scripts/eval_fsod_dualpath.py \
            --config "$CFG" \
            --weights "$VIS_WEIGHTS" \
            --text-proto "$PROTO" \
            --clip-model "$CLIP_MODEL" \
            --fusion-mode none \
            --split val \
            --run-tag "A1_${DS_LOWER}_${SHOT}shot_visualonly" \
            2>&1 | tee -a "$LOG_DIR/eval_${DS_LOWER}_${SHOT}shot_visualonly_none.log"

        # 4b: Visual-Only + Dual-Path (multiplicative, T=5)
        echo "[eval] Visual-Only + Dual-Path $SHOT-shot..."
        python scripts/eval_fsod_dualpath.py \
            --config "$CFG" \
            --weights "$VIS_WEIGHTS" \
            --text-proto "$PROTO" \
            --clip-model "$CLIP_MODEL" \
            --fusion-mode multiplicative --gamma $GAMMA --vlm-temperature $VLM_TEMP \
            --split val \
            --run-tag "A2_${DS_LOWER}_${SHOT}shot_visualonly_dualpath" \
            2>&1 | tee -a "$LOG_DIR/eval_${DS_LOWER}_${SHOT}shot_visualonly_mult_T5.log"

        # 4c: DAF baseline
        echo "[eval] DAF baseline $SHOT-shot..."
        python scripts/eval_fsod_dualpath.py \
            --config "$CFG" \
            --weights "$DAF_WEIGHTS" \
            --text-proto "$PROTO" \
            --clip-model "$CLIP_MODEL" \
            --fusion-mode none \
            --split val \
            --run-tag "A0_${DS_LOWER}_${SHOT}shot_daf_baseline" \
            2>&1 | tee -a "$LOG_DIR/eval_${DS_LOWER}_${SHOT}shot_daf_none.log"

        # 4d: DAF + Dual-Path (multiplicative, T=5)
        echo "[eval] DAF + Dual-Path $SHOT-shot..."
        python scripts/eval_fsod_dualpath.py \
            --config "$CFG" \
            --weights "$DAF_WEIGHTS" \
            --text-proto "$PROTO" \
            --clip-model "$CLIP_MODEL" \
            --fusion-mode multiplicative --gamma $GAMMA --vlm-temperature $VLM_TEMP \
            --split val \
            --run-tag "A3_${DS_LOWER}_${SHOT}shot_daf_dualpath" \
            2>&1 | tee -a "$LOG_DIR/eval_${DS_LOWER}_${SHOT}shot_daf_mult_T5.log"

        echo "[done] $DS $SHOT-shot complete"
    done
done

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))

echo ""
echo "=============================================="
echo "Phase 1 Complete!"
echo "Total duration: ${HOURS}h ${MINS}m"
echo "Results appended to: runs/direction_c/dualpath_results.csv"
echo "Logs: $LOG_DIR/"
echo ""
echo "To view results quickly:"
echo "  column -ts, runs/direction_c/dualpath_results.csv | grep -E 'A0_|A1_|A2_|A3_'"
echo "=============================================="
