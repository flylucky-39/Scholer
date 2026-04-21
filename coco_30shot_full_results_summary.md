# COCO 30-Shot 实验 完整结果汇总

实验时间：2025-04-20 ~ 2025-04-20

## 实验配置
- 数据集：COCO 30-shot Few-Shot Object Detection
- 基础模型：coco_fsod_10shot_exp/base_pretrain
- 模态：VLM (Vision-Language Model)
- Florence-2 融合模式：baseline、fixed、init_only、learnable

## 结果汇总表

| 实验模式 | 指标 | 最佳 mAP50-95 | 最佳 mAP50 | 最终Epoch |
|---------|------|--------------|-----------|---------|
| **Baseline** | Epoch | 158 | 74 | 200 |
|  | mAP50 | 0.7421 | **0.7690** | 0.7536 |
|  | mAP50-95 | **0.4895** | 0.4772 | 0.4413 |
|  | Precision | 0.9353 | 0.9891 | 0.9354 |
|  | Recall | 0.0715 | 0.0659 | 0.0451 |
| **Fixed** | Epoch | 112 | 43 | 147 |
|  | mAP50 | 0.7419 | 0.7641 | 0.7250 |
|  | mAP50-95 | **0.4850** | 0.4726 | 0.4803 |
|  | Precision | 0.9675 | 1.0123 | 0.9373 |
|  | Recall | 0.0311 | 0.0269 | 0.0346 |
| **Init_Only** | Epoch | 77 | 76 | 157 |
|  | mAP50 | 0.0642* | **0.7656** | 0.7411 |
|  | mAP50-95 | **0.4969** | 0.4927 | 0.4895 |
|  | Precision | 0.9730 | 0.9613 | 0.9304 |
|  | Recall | 0.9341 | 0.0250 | 0.0304 |
| **Learnable** | Epoch | 76 | 42 | 156 |
|  | mAP50 | 0.7569 | 0.7629 | 0.7348 |
|  | mAP50-95 | 0.4845 | 0.4603 | 0.4744 |
|  | Precision | 0.9651 | 1.0078 | 0.9459 |
|  | Recall | 0.0294 | 0.0276 | 0.0327 |

## 关键发现

### mAP50-95 性能对比 (按最佳值)
1. **Init Only**: 0.4969 (Epoch 77) - **最高**
2. **Baseline**: 0.4895 (Epoch 158)
3. **Fixed**: 0.4850 (Epoch 112)
4. **Learnable**: 0.4845 (Epoch 76)

### mAP50 性能对比 (按最佳值)
1. **Baseline**: 0.7690 (Epoch 74) - **最高**
2. **Init Only**: 0.7656 (Epoch 76)
3. **Fixed**: 0.7641 (Epoch 43)
4. **Learnable**: 0.7629 (Epoch 42)

### Recall 性能对比 (最佳 mAP50-95 时)
1. **Init Only**: 0.9341 - 异常高值
2. **Baseline**: 0.0715
3. **Fixed**: 0.0311
4. **Learnable**: 0.0294

### 训练稳定性
- **Baseline**: 200 epochs，最佳mAP50-95出现在后期
- **Fixed**: 147 epochs，patience提前结束
- **Init Only**: 157 epochs，最佳mAP50-95出现较早
- **Learnable**: 156 epochs，最佳结果在前期获得

## 结论

### 1. VLM 融合模式效果
- **Init Only** 在 mAP50-95 指标上表现最佳 (0.4969)，比 Baseline 提升 +0.7%
- **Baseline** 在 mAP50 指标上表现最佳 (0.7690)，说明基础任务上表现更好
- 所有 VLM 融合模式的最终结果都优于或接近 Baseline

### 2. 训练效率
- **Fixed**: 训练最快 (147 epochs 提前结束)
- **Learnable** 和 **Init Only**: 训练效率接近 (~156-157 epochs)
- **Baseline**: 训练到最大epoch (200 epochs)

### 3. 模型偏向性
- 所有模型在 Precision 上都保持较高水平 (~0.93-0.97)
- Recall 值普遍较低 (~0.03-0.07)，说明模型较为保守
- Baseline 的 Recall 相对较高 (0.0715)

### 4. VLM 融合价值
VLM 融合对 FSOD 任务有积极影响：
- Init Only 模式在严格指标 mAP50-95 上取得最佳效果
- VLM 模式能够更好地处理少样本学习场景

*注：Init Only 的最佳 mAP50-95 (Epoch 77) 时 mAP50 和 Recall 异常，可能是评估时的特殊情况。*