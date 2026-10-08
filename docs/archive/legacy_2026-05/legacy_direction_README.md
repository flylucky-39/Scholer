> ⚠️ **已归档 · 旧协议数据（≤2026-05）**
> 本文件产生于**数据泄漏协议**（base 从 yolo11s.pt——COCO 80 类 600 epoch 官方权重——初始化，VOC novel 类零样本 mAP50=0.93），文中全部数字**不可用于论文或实验对比**。
> 唯一可信数据源：[`docs/CLEAN_PROTOCOL_RESULTS.md`](../../CLEAN_PROTOCOL_RESULTS.md)（干净协议，2026-09-22）。归档：2026-10-08。

# FSOD VLM Baseline Reproduction Workspace

这个仓库现在作为跨域少样本目标检测（CD-FSOD）的 baseline 复现与改进工作区使用。当前主线从自研 YOLO + DAF + Dual-Path 探索，切换为：

1. 先复现外部强 baseline 论文，例如 CD-ViTO / DE-ViT。
2. 在统一数据集、统一 split、统一指标口径下建立可靠对照。
3. 再围绕 prototype、prompt、support quality 或 background calibration 做小而明确的改进。

旧 VOC / COCO / YOLO-FSOD / Dual-Path 代码和结果暂时保留为历史资产，第一轮清理不删除、不移动代码文件。

## 当前定位

推荐新方向：**Quality-Aware Prototype Calibration for Cross-Domain Few-Shot Object Detection**。

核心思路是：以 CD-ViTO / DE-ViT 等已发表方法作为主 baseline，优先复现 CD-FSOD-Bench 上的 1/5/10-shot 结果，然后加入质量感知 prototype 校准、领域 prompt 或背景原型校准模块。

## 重要文档

- [docs/baseline_reproduction_plan.md](docs/baseline_reproduction_plan.md)：新 baseline 复现路线。
- [docs/restart_baseline_cleanup_plan.md](docs/restart_baseline_cleanup_plan.md)：第一轮清理边界和后续清理建议。
- [docs/archive/README.md](docs/archive/README.md)：旧方向资产说明。
- [cdfssod_banchmarkRESULT.md](cdfssod_banchmarkRESULT.md)：已有 CD-ViTO DIOR 10-shot 复现记录。
- [docs/cdfsod_quickstart.md](docs/cdfsod_quickstart.md)：当前 CD-FSOD 数据准备和旧 YOLO-FSOD 流程说明。

## 建议保留资产

```text
configs/cdfsod_*shot.yaml       # 6 个 CD-FSOD 数据集的 1/5/10-shot 配置
prompts/*.json                  # 每个数据集的类别 prompt 文件
fsod/cdfsod/                    # CD-FSOD 数据集注册、domain gap、calibration 工具
scripts/prepare_cdfsod.py       # CD-FSOD 数据转换入口
cdfssod_banchmarkRESULT.md      # CD-ViTO 复现记录
third_party/ultralytics/        # 旧 YOLO-FSOD 线仍依赖的 vendored Ultralytics
```

## 第一阶段目标

1. 选定主 baseline：优先 CD-ViTO，备选 DE-ViT。
2. 复现至少 3 个代表性数据集：DIOR、Clipart1k、DeepFish。
3. 每个数据集先跑 1/5/10-shot，记录 AP、AP50、AP75。
4. 固定环境、配置、权重和 split，形成可重复的 baseline 表。
5. 在 baseline 稳定后，再加入自己的 prototype calibration 改进。

## 当前已有结果

已有一版 CD-ViTO DIOR 10-shot 复现结果：

| Method | Dataset | Shot | AP | AP50 | AP75 |
| --- | --- | ---: | ---: | ---: | ---: |
| CD-ViTO | DIOR | 10 | 30.41 | 46.56 | 32.47 |

结果细节见 [cdfssod_banchmarkRESULT.md](cdfssod_banchmarkRESULT.md)。

## 清理原则

第一轮只做文档层清理：

1. README 改为新方向入口。
2. 新增 baseline 复现计划。
3. 新增旧方向归档说明。
4. 不删除任何文件。
5. 不移动 Python 代码。
6. 不回滚当前未提交改动。

后续如果要删除或移动文件，需要逐项确认。

## 环境提示

本仓库仍保留旧 YOLO-FSOD 依赖：

```bash
pip install -r requirements.txt
```

CD-ViTO / DE-ViT 复现通常需要单独的 Detectron2 / RegionCLIP / DINOv2 环境，建议和本仓库 Python 环境隔离管理，并在 `baselines/` 或文档中记录完整命令。