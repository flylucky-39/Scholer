# Baseline 论文复现计划

更新时间：2026-05-07

## 1. 新主线

当前仓库主线切换为：先复现外部强 baseline，再在此基础上做小模块改进。

优先级：

1. CD-ViTO：已完成 DIOR 10-shot 的一次复现记录，最适合作为第一主线。
2. DE-ViT：作为第二 baseline 或补充 SOTA 对比。
3. Detection-PT / TFA / FSCE / DeFRCN：作为后续表格补充，不作为第一阶段重点。

## 2. 第一阶段目标

第一阶段不设计新模块，只做可靠复现。

目标表格：

| Method | Dataset | Shot | AP | AP50 | AP75 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CD-ViTO | DIOR | 10 | 30.41 | 46.56 | 32.47 | 已复现 |
| CD-ViTO | DIOR | 1 | TBD | TBD | TBD | 待跑 |
| CD-ViTO | DIOR | 5 | TBD | TBD | TBD | 待跑 |
| CD-ViTO | Clipart1k | 1/5/10 | TBD | TBD | TBD | 待跑 |
| CD-ViTO | DeepFish | 1/5/10 | TBD | TBD | TBD | 待跑 |

优先数据集：

1. DIOR：已有 CD-ViTO 10-shot 结果，先补 1/5-shot。
2. Clipart1k：语义和 CLIP 更接近，适合验证视觉语言 baseline。
3. DeepFish：单类水下数据，能观察 domain gap 和背景影响。
4. NEU-DET / UODD / ArTaxOr：第二轮补齐。

## 3. 复现记录要求

每个 baseline 必须记录：

1. 论文名称、代码仓库、commit 或下载时间。
2. Python / CUDA / PyTorch / Detectron2 或其他核心依赖版本。
3. 数据集路径、split、shot 文件来源。
4. 预训练权重路径和下载来源。
5. 完整训练命令。
6. 完整评估命令。
7. AP、AP50、AP75、每类 AP。
8. 输出目录和日志路径。
9. 是否和论文官方指标一致，不一致时记录可能原因。

## 4. 建议目录结构

外部 baseline 不建议直接揉进 `fsod/` 主包，建议单独放在 `baselines/` 下：

```text
baselines/
  cdvito/
    README.md
    configs/
    logs/
    results/
  devit/
    README.md
    configs/
    logs/
    results/
```

如果外部论文代码仓库很大，建议不要复制完整源码进来，而是在文档中记录 clone 路径、commit、环境和运行命令。

## 5. 可复用的本仓库资产

当前仓库可继续服务 baseline 复现的部分：

- `configs/cdfsod_*shot.yaml`：CD-FSOD 数据集和 shot 配置。
- `prompts/*.json`：类别文本描述，可用于 prompt ensemble 或文本原型。
- `fsod/cdfsod/datasets.py`：数据集类别注册。
- `scripts/prepare_cdfsod.py`：YOLO 格式转换。如果外部 baseline 需要 COCO / Detectron2 格式，可在确认后新增导出脚本。
- `cdfssod_banchmarkRESULT.md`：CD-ViTO DIOR 10-shot 已复现记录。

## 6. 后续可改模块

复现稳定后再考虑自己的方法。优先级建议：

1. Support-Quality Weighted Prototype：根据 support crop 质量、类内一致性或 CLIP 相似度加权 prototype。
2. Domain-Aware Prompt Ensemble：按数据集域类型选择或加权 prompt。
3. Background-Aware Prototype Refinement：针对跨域背景混淆做背景原型或 hard negative 校准。
4. Domain-specific CLIP 替换：RemoteCLIP / GeoRSCLIP / underwater-specific encoder，作为替换验证器或原型编码器。

第一版不建议大改检测头或 backbone。

## 7. 下一步执行顺序

1. 整理 CD-ViTO 复现环境和已有 DIOR 10-shot 命令。
2. 补跑 CD-ViTO DIOR 1-shot 和 5-shot。
3. 跑 CD-ViTO Clipart1k 1/5/10-shot。
4. 跑 CD-ViTO DeepFish 1/5/10-shot。
5. 生成统一结果表。
6. 再决定第一个改进模块。
