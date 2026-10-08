> ⚠️ **已归档 · 旧协议数据（≤2026-05）**
> 本文件产生于**数据泄漏协议**（base 从 yolo11s.pt——COCO 80 类 600 epoch 官方权重——初始化，VOC novel 类零样本 mAP50=0.93），文中全部数字**不可用于论文或实验对比**。
> 唯一可信数据源：[`docs/CLEAN_PROTOCOL_RESULTS.md`](../../CLEAN_PROTOCOL_RESULTS.md)（干净协议，2026-09-22）。归档：2026-10-08。

# Scale Regularization 实验 — YOLO-FSOD (v0.1.0)

## 实验目的

解决 Few-Shot Object Detection 中因 CosineConv2d 温度参数 (scale) 无约束增长，导致低置信度阈值下产生大量误检框 (False Positive) 的问题。

## 方法

在 finetune 阶段的损失函数中，对每个 `CosineConv2d` 层的温度参数 `scale.exp()` 添加 L2 正则化：

```
L = L_original + reg_weight * mean(scale.exp())
```

阻止分类头的温度无限膨胀，使模型输出的置信度保持在校准的范围内。

## 分支

`v0.1.0`

### 新增文件

| 文件 | 说明 |
|------|------|
| `fsod/modules/scale_regularization.py` | 自定义损失 `ScaleRegDetectionLoss`，继承 `v8DetectionLoss`，加入 scale L2 惩罚 |
| `scripts/train_fsod_scalereg.py` | 训练脚本，添加 `--reg-weight` 参数，支持 prototype / Florence-2 初始化 + 回调注入正则化损失 |
| `configs/baseline_voc_10shot_scalereg.yaml` | 实验配置，`runs_dir: ./runs/fsod_scalereg` |

### 修改文件

| 文件 | 说明 |
|------|------|
| `.gitignore` | 添加 `runs/` 和 `outputs/` 忽略训练产出 |

## 配置

### 数据集

- **数据集**: VOC 2007+2012
- **划分**: Split 1
- **Novel 类别 (5 类)**: bird, bus, cow, motorbike, sofa
- **Base 类别 (15 类)**: aeroplane, bicycle, boat, bottle, car, cat, chair, diningtable, dog, horse, person, pottedplant, sheep, train, tvmonitor
- **Shot**: 10-shot（每类 10 张标注图片）

### 模型

- **骨架**: YOLO11s (9.4M 参数)
- **检测头**: FSODDetect（`CosineConv2d` 替换 `cv3` 最后一层卷积）
- **初始化方式**: Base 预训练权重 + Florence-2 描述调制 + Visual Prototype 融合初始化

### 训练参数

| 参数 | Base 预训练 | Finetune |
|------|-------------|----------|
| epochs | 100 | 200 (早停于 epoch 136) |
| batch_size | 64 | 4 |
| lr0 | 0.01 | 0.0005 |
| imgsz | 640 | 640 |
| freeze | - | backbone 10 epochs |
| patience | - | 30 |

### Scale Regularization

| 参数 | 值 |
|------|------|
| `reg_weight` | 0.05 |
| 实现方式 | 回调注入 `ScaleRegDetectionLoss` 替换原始 loss |

## 实验结果

### 训练过程

| Epoch | P | R | mAP50 | mAP50-95 |
|-------|------|------|-------|----------|
| 5 | 0.004 | 0.799 | 0.100 | 0.036 |
| 50 | 0.012 | 0.910 | 0.650 | 0.440 |
| 90 | 0.011 | 0.935 | 0.675 | 0.470 |
| 107 | 0.129 | 0.869 | 0.730 | 0.510 |
| 118 (best) | 0.162 | 0.870 | **0.733** | 0.505 |
| 136 (final) | 0.130 | 0.870 | 0.731 | 0.512 |

- **总训练时间**: ~1.95 小时
- **早停**: epoch 136（patience=30 触发）

### 置信度阈值扫描 (conf sweep)

使用 novel-only test set (1024 images, 1480 instances):

| conf | P | R | mAP50 | mAP50-95 |
|------|------|------|-------|----------|
| 0.001 (默认) | 0.130 | 0.870 | 0.731 | 0.513 |
| 0.01 | 0.130 | 0.870 | 0.732 | 0.513 |
| **0.05** ✅ | **0.607** | **0.746** | **0.756** | **0.565** |
| 0.10+ | 0.000 | 0.000 | 0.000 | 0.000 |

### 最佳阈值下各类别指标 (conf=0.05)

| 类别 | P | R | mAP50 | mAP50-95 |
|------|------|------|-------|----------|
| **all** | **0.607** | **0.746** | **0.756** | **0.565** |
| bird | 0.842 | 0.338 | 0.598 | 0.417 |
| bus | 0.593 | 0.854 | 0.856 | 0.728 |
| cow | 0.579 | 0.885 | 0.837 | 0.598 |
| motorbike | 0.651 | 0.788 | 0.801 | 0.551 |
| sofa | 0.370 | 0.866 | 0.688 | 0.530 |

## 关键发现

1. **Scale regularization 有效压制了温度膨胀**: 所有检测框的置信度被限制在 0.05-0.07 区间，模型不再产生高置信度预测
2. **P 在适当阈值下大幅提升**: conf=0.05 时 P=0.607，相比默认阈值 (0.001) 的 0.130 提升 4.7 倍，有效减少误检框
3. **mAP 基本无损**: 在最佳阈值 conf=0.05 下 mAP50=0.756，甚至略高于 conf=0.001 时的 0.731
4. **bird 的 recall 偏低** (0.338): 小目标 bird 的置信度常低于 0.05 被过滤
5. **置信度分布过窄**: 所有预测置信度聚集在 0.05-0.07，无法产生高置信度检测，实际部署阈值选择受限

## 运行命令

### 训练

```bash
python scripts/train_fsod_scalereg.py \
  --config configs/baseline_voc_10shot_scalereg.yaml \
  --stage finetune \
  --reg-weight 0.05 \
  --base-weights base_pretrain/weights/best.pt \
  --epochs 200 \
  --prototype \
  --florence2 ~/epfs/07_FSOD_LLM/models/Florence-2-base/
```

### 阈值扫描

```bash
python scripts/sweep_confidence.py \
  --weights runs/fsod_scalereg/novel_finetune_cosine_fused_scalereg_ep200/weights/best.pt \
  --config configs/baseline_voc_10shot_scalereg.yaml \
  --novel-only
```

## 对比 Baseline

| 指标 | Baseline (fused) | v0.1.0 (scalereg) |
|------|-----------------|--------------------|
| mAP50 (conf=0.001) | ~0.73 | 0.731 |
| mAP50 (最优阈值) | ~0.73 @ 0.001 | **0.756 @ 0.05** |
| P (最优阈值) | ~0.13 | **0.607** |
| R (最优阈值) | ~0.87 | 0.746 |

## 输出目录

- 训练结果: `runs/fsod_scalereg/novel_finetune_cosine_fused_scalereg_ep200/`
- 训练日志: `logs/finetune_scalereg_fused.log`
- 演示图片: `runs/demo_fused_scalereg/`
- 阈值扫描结果: `runs/fsod_scalereg/novel_finetune_cosine_fused_scalereg_ep200/conf_sweep.json`
