> ⚠️ **已归档 · 旧协议数据（≤2026-05）**
> 本文件产生于**数据泄漏协议**（base 从 yolo11s.pt——COCO 80 类 600 epoch 官方权重——初始化，VOC novel 类零样本 mAP50=0.93），文中全部数字**不可用于论文或实验对比**。
> 唯一可信数据源：[`docs/CLEAN_PROTOCOL_RESULTS.md`](../../CLEAN_PROTOCOL_RESULTS.md)（干净协议，2026-09-22）。归档：2026-10-08。

# 重新开题代码清理与复现基线改造计划

更新时间：2026-05-07

目标：放弃当前 YOLO + DAF + Dual-Path 主线，把仓库整理成“复现外部论文作为 baseline，再做小模块改进”的研究工作区。

> 本文档只做清理规划，不删除、不移动、不修改任何代码文件。实际清理前需要逐项确认。

## 1. 新方向建议

建议主线：先复现 CD-ViTO / DE-ViT 等跨域少样本目标检测论文，再在其 prototype / prompt / support quality / background calibration 模块上做改进。

更稳的第一版题目：

**Quality-Aware Prototype Calibration for Cross-Domain Few-Shot Object Detection**

中文：面向跨域少样本目标检测的质量感知原型校准方法。

## 2. 当前仓库资产分层

### A. 应保留的资产

这些和新方向强相关，建议保留：

- `configs/cdfsod_*shot.yaml`：6 个 CD-FSOD 数据集 × 1/5/10-shot 配置已齐。
- `prompts/*.json`：6 个数据集的 prompt 文件已齐，可用于文本原型或 prompt ensemble。
- `fsod/cdfsod/datasets.py`：CD-FSOD-Bench 数据集注册与类别定义。
- `scripts/prepare_cdfsod.py`：CD-FSOD 数据转换工具，可继续服务 baseline 复现。
- `cdfssod_banchmarkRESULT.md`：CD-ViTO DIOR 10-shot 复现记录，是新方向的重要起点。
- `docs/cdfsod_quickstart.md`：CD-FSOD 运行说明，可更新为新主线 quickstart。

### B. 暂时保留但不作为主线的资产

这些可能还可作为参考或辅助 baseline，不建议第一轮删除：

- `fsod/modules/cosine_head.py`
- `fsod/modules/prototype.py`
- `fsod/modules/adaptation.py`
- `fsod/cdfsod/calibration.py`
- `fsod/cdfsod/domain_gap.py`
- `scripts/train_cdfsod.py`
- `scripts/extract_vlm_text_proto.py`
- `third_party/ultralytics/`

原因：虽然它们属于旧 YOLO-FSOD/DAF 线，但 prototype、calibration、文本原型等概念能迁移到新方法设计里。

### C. 旧方向归档候选

这些主要服务当前准备放弃的 Dual-Path 方向，建议从主线文档和入口里移走，先归档，确认无用后再删除：

- `scripts/eval_fsod_dualpath.py`
- `fsod/modules/dual_path_fusion.py`
- `fsod/modules/vlm_verifier.py`
- `fsod/modules/tests/test_dual_path_fusion.py`
- `scripts/run_phase1.sh`
- `scripts/summarize_phase1.py`
- `runs/direction_c/`
- `HANDOFF.md` 中关于 Direction C 的主线描述

注意：这些文件当前有未提交改动，不能直接回滚或删除。

### D. 早期 VOC / COCO 实验归档候选

这些是早期 YOLO baseline 探索，和“复现外部论文作为 baseline”关系较弱：

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
- 顶层 `3shotresult.txt`、`5shotresult.txt`、`test_result.txt`、`test_proto_result.txt`、`test_run_output.txt`、`output_log.txt`、`full_output.txt`
- 顶层 `coco_*results*.csv/md`

第一轮建议只归档结果和文档，不急着删脚本，因为 COCO 预训练权重/数据转换经验可能还会被用来做对比。

### E. 工具生成文件候选

这些不是科研代码，建议单独确认是否加入 `.gitignore` 或删除：

- `.omc/`
- `.claude/settings.local.json`
- `tmpclaude-98e5-cwd`

## 3. 推荐清理顺序

### 第 0 步：保护现场

在任何清理前先做三件事：

1. 查看 git 状态。
2. 把当前未提交改动分成“要保留”和“要放弃”。
3. 对要删除的文件逐项确认。

当前已知未提交改动包括：

- `fsod/modules/dual_path_fusion.py`
- `fsod/modules/tests/test_dual_path_fusion.py`
- `scripts/eval_fsod_dualpath.py`
- `scripts/run_phase1.sh`
- `scripts/summarize_phase1.py`
- `.omc/**`
- `.claude/settings.local.json`

### 第 1 步：只整理文档入口

推荐先更新文档，不改代码：

- 将 `README.md` 从 VOC baseline 说明改成“CD-FSOD baseline replication workspace”。
- 新增或更新 `docs/baseline_reproduction_plan.md`，列出 CD-ViTO / DE-ViT 复现步骤。
- 将旧方向结果集中到 `docs/archive/legacy_yolo_dualpath/`。

### 第 2 步：建立新方向目录边界

建议后续采用如下结构：

```text
baselines/
  cdvito/
    README.md
    configs/
    scripts/
    results/
  devit/
    README.md
    configs/
    scripts/
    results/

fsod/
  cdfsod/              # 数据集定义/转换工具，保留
  modules/             # 只保留跨方法可复用的小模块

scripts/
  prepare_cdfsod.py    # 保留
  reproduce_cdvito.py  # 后续新增，封装复现命令/结果抽取
  summarize_baselines.py
```

原则：外部论文复现代码不要揉进原来的 YOLO-FSOD 主包里，避免旧线和新线互相污染。

### 第 3 步：代码归档而非直接删除

第一轮建议做“软清理”：

- 把旧线脚本移动到 `archive/legacy_yolo_fsod/scripts/`。
- 把旧线模块移动到 `archive/legacy_yolo_fsod/modules/` 或保留在原处但从 README 主流程移除。
- 把旧结果移动到 `archive/legacy_results/`。

等新 baseline 复现跑通后，再决定是否删除 archive。

### 第 4 步：最小代码改造

新方向第一批真正需要写的代码不多：

1. `scripts/summarize_baseline_results.py`：统一抽取 CD-ViTO/DE-ViT 的 AP、AP50、AP75。
2. `scripts/prepare_cdfsod.py`：如外部 baseline 需要特定格式，可增加导出 Detectron2/COCO split 的选项。
3. `baselines/cdvito/README.md`：记录环境、命令、权重、结果。
4. `baselines/cdvito/configs/`：保存每个数据集/shot 的配置副本。

不要一开始改 detection head 或训练框架。

## 4. 建议第一轮清理执行清单

如果要我接着执行，建议第一轮只做这些安全动作：

1. 新建 `docs/baseline_reproduction_plan.md`。
2. 更新 `README.md` 的项目定位。
3. 新建 `archive/legacy_results/README.md`，说明旧结果来源。
4. 只移动结果文档/日志，不移动 Python 代码。
5. 生成一份待删除清单，但不删除。

## 5. 需要你确认的危险动作

以下动作必须逐项确认后才能做：

- 删除任何文件。
- 移动或重命名 Python 代码文件。
- 修改 `fsod/modules/__init__.py` 的导出项。
- 修改 `.gitignore`。
- 改写 `README.md` 主叙事。
- 回滚当前未提交的 Dual-Path 改动。

## 6. 我建议的下一步

先执行“只文档化 + 软归档”的清理，不删代码。等 CD-ViTO 或 DE-ViT 至少跑通 1 个数据集后，再删除旧 YOLO/VOC/COCO 线。

推荐确认语：

> 同意先做第一轮安全清理：只更新文档和归档旧结果，不删除、不移动 Python 代码。
