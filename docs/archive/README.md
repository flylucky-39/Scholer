# 归档索引（2026-10-08 整理）

本目录与 `runs/archive/` 存放**数据泄漏协议时代（≤2026-05）**的历史资产。
**所有归档数字不可用于论文或实验对比**；唯一可信数据源：[`docs/CLEAN_PROTOCOL_RESULTS.md`](../CLEAN_PROTOCOL_RESULTS.md)（干净协议，2026-09-22）。

## 为什么归档

原协议的 base 训练从官方 `yolo11s.pt`（COCO 80 类、118K 图、600 epoch）初始化，而 VOC novel 类（bird/bus/cow/motorbike/sofa）在 COCO 中同类同图，构成数据泄漏——官方权重在 VOC novel 上零样本 mAP50=0.9282。2026-09 协议重建后，全部实验改为从零训练 base（`*.yaml` 从零初始化、epoch 50→120/200），主表、消融、敏感性全部重跑。

## legacy_2026-05/（旧文档，md/tex 均带弃用横幅）

| 文件 | 说明 |
|---|---|
| `legacy_direction_README.md` | 旧版仓库 README（CD-FSOD baseline 复现方向，已废弃） |
| `HANDOFF.md` | 旧方向交接文档（YOLO-FSOD → DAF / Dual-Path VLM） |
| `legacy_experiment_plan.md` | 旧期刊扩展实验计划（scale reg / per-class temp / bg loss） |
| `EXPERIMENT_v0.1.0.md` | v0.1.0 Scale Regularization 实验记录 |
| `cdfssod_banchmarkRESULT.md` | CD-ViTO DIOR 复现记录 |
| `coco_all_experiments_summary.md`、`coco_30shot_results_summary.md` | 旧协议 COCO 结果汇总（**数字作废**） |
| `*.csv`、`FSOD_results.xlsx`、`3shotresult.txt`、`5shotresult.txt` | 旧协议结果数据 |
| `vcp_paper_*.{md,tex,pdf}`、`conference_101719.tex`、`IEEEtran.cls` | 会议论文稿（含泄漏协议数字） |
| `221349391900.doc`、`extract_doc.py`、`doc_output.txt` | CAC 2025 投稿模板及提取工具 |
| `baseline_reproduction_plan.md`、`restart_baseline_cleanup_plan.md` | 2026-05 方向切换与清理计划（已执行完毕并被干净协议超越） |
| `cdfsod_quickstart.md`、`proposal_cdfsod_realtime.md` | CD-FSOD 方向文档 |
| `fig1-3`（jpg/pdf/png）、`paper_figures/`、`generate_figures.py` | 旧数字生成的论文图 |
| `test_*.txt`、`output_log.txt`、`full_output.txt`、`conf_sweep_test_novel_results.json` | 旧训练日志与评测输出 |
| `create_excel.py`、`plot_experiment_results.py`、`test_adapt.py` | 旧结果处理脚本 |

## runs/archive/（旧协议 runs，31 个目录）

| 目录 | 内容 |
|---|---|
| `voc_fsod_*`（含 split2/3、freeze_cv2 变体） | 旧泄漏协议 VOC 实验（旧主表数字来源，作废） |
| `coco_fsod_*`（exp/OLD/New/Vlm/baseline） | 旧协议 COCO 实验 |
| `cdfsod_DIOR_{1,5,10}shot` | CD-FSOD / DAF 方向 |
| `direction_c/` | Dual-Path VLM 融合 |
| `fsod_{1,3,5}shot`、`fsod_baseline`、`fsod_cap_1shot`、`fsod_scalereg` | v0.1.0 scale-reg 时代 |
| `base_pretrain/` | 旧 base 训练输出（yolo11s.pt 初始化，2026-04-09） |

## 移出仓库的旧检出（2026-10-08）

`fsod/` 目录内曾混入一份完整的旧项目检出（嵌套 `.git` 955MB + 旧 runs 297MB，共 1.3GB，未被跟踪），已整体移至 `E:\Study\FSOD_VLM_legacy\fsod_dump\`；`.backup_conflicts\` 移至同目录 `backup_conflicts\`。仓库内 `fsod/` 现仅保留当前 Python 包（`voc.py`、`coco.py`、`florence2.py`、`modules/`、`cdfsod/`）。

另清除了误提交的嵌套副本：`fsod/fsod/`、`fsod/configs/baseline_voc_1shot_augment.yaml`、`fsod/scripts/`（git 历史可找回）。

## 仍在原位、未归档的旧资产（超出本次文档整理范围）

- `configs/` 中旧方向配置（`baseline_voc_*`、`cdfsod_*`、旧版 `coco_*.yaml` 等）
- `data/florence2_outputs/`、`prompts/`、`datasets/`、`third_party/ultralytics/`
- `docs/server_ai_runbook_no_changes.md`、`docs/gitlab_server_workflow.md`（服务器运维文档，仍有效）

如需进一步清理，逐项确认后再动。
