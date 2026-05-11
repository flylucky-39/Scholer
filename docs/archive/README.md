# 历史方向归档说明

更新时间：2026-05-07

本目录用于说明旧研究方向和历史实验资产。第一轮清理只做文档归档，不删除、不移动任何已有代码或结果文件。

## 1. 已转为历史探索的方向

### YOLO-FSOD / VOC / COCO baseline

早期仓库以 YOLO11s 为基础，围绕 VOC / COCO few-shot detection 做 baseline、cosine classifier、prototype、Florence-2 文本融合等探索。

相关资产仍保留在原路径：

- `scripts/prepare_voc_fewshot.py`
- `scripts/prepare_coco_fewshot.py`
- `scripts/train_baseline.py`
- `scripts/train_fsod.py`
- `scripts/eval_baseline.py`
- `scripts/eval_fsod.py`
- `fsod/voc.py`
- `fsod/coco.py`
- `configs/baseline_voc_*.yaml`
- `configs/coco_*.yaml`
- `runs/voc_fsod_*`
- `runs/coco_fsod_*`

这些文件暂不作为新主线入口，但保留作对照和代码参考。

### DAF / Dual-Path VLM Verifier

之后仓库推进到 CD-FSOD 上的 DAF 和 Dual-Path 推理融合方向。该方向已经在 DIOR 上形成若干结果，但整体增益不够稳定，因此不再作为主线继续推进。

相关资产仍保留在原路径：

- `scripts/train_cdfsod.py`
- `scripts/eval_fsod_dualpath.py`
- `scripts/extract_vlm_text_proto.py`
- `fsod/modules/dual_path_fusion.py`
- `fsod/modules/vlm_verifier.py`
- `fsod/modules/tests/test_dual_path_fusion.py`
- `runs/direction_c/`
- `HANDOFF.md`

其中 `train_cdfsod.py`、`prototype.py`、`calibration.py`、`prompts/*.json` 仍可能被新方向复用，暂不归为删除候选。

## 2. 已有历史结果

重要历史结果文件：

- `runs/direction_c/visualonly_vs_daf_comparison.md`
- `runs/direction_c/dualpath_results_summary.md`
- `runs/direction_c/dualpath_results.csv`
- `coco_all_experiments_summary.md`
- `coco_30shot_results_summary.md`
- `coco_all_experiments_results.csv`
- `coco_30shot_results.csv`
- `3shotresult.txt`
- `5shotresult.txt`

这些结果目前不删除，因为它们可以解释为什么主线切换到外部论文 baseline 复现。

## 3. 清理原则

后续清理按以下顺序进行：

1. 先确认新 baseline 复现能跑通。
2. 再移动旧结果到 `docs/archive` 或专门的 `archive/legacy_results`。
3. 最后才考虑删除旧代码。

删除任何文件前都需要单独确认。

## 4. 当前主线入口

新方向入口见：

- `README.md`
- `docs/baseline_reproduction_plan.md`
- `docs/restart_baseline_cleanup_plan.md`
