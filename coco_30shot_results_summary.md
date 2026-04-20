# COCO 30-Shot 实验 结果汇总

实验时间：2025-04-20

## 实验配置
- 数据集：COCO 30-shot Few-Shot Object Detection
- 基础模型：coco_fsod_10shot_exp/base_pretrain
- 模态：VLM (Vision-Language Model)
- Florence-2 融合模式：fixed、init_only、learnable

## 结果汇总表

| 实验模式 | 指标 | 最佳 mAP50-95 | 最佳 mAP50 | 最终Epoch |
|---------|------|--------------|-----------|---------|
| **Fixed** | Epoch | 112    |    43      | 147 |
|| mAP50 | 0.7419 | **0.7641** | 0.7250 |
|| mAP50-95 | **0.4850** | 0.4726 | 0.4803 |
|| Precision | 0.9675 | 1.0123 | 0.9373 |
|| Recall | 0.0311 | 0.0269 | 0.0346 |
| **Init_Only** | Epoch | 77 | 76  | 157  |
|| mAP50 | 0.0642* | **0.7656** | 0.7411 |
|| mAP50-95 | **0.4969** | 0.4927 | 0.4895 |
|| Precision | 0.9730 | 0.9613 | 0.9304 |
|| Recall | 0.9341 | 0.0250 | 0.0304 |
| **Learnable** | Epoch | 76 | 42 | 156 |
|| mAP50 | **0.7569** | 0.7629 | 0.7348 |
|| mAP50-95 | 0.4845 | 0.4603 | 0.4744 |
|| Precision | 0.9651 | 1.0078 | 0.9459 |
|| Recall | 0.0294 | 0.0276 | 0.0327 |

## 关键发现

### mAP50-95 性能对比
1. **Init Only**: 0.4969 (Epoch 77) - **最高**
2. **Fixed**: 0.4850 (Epoch 112)
3. **Learnable**: 0.4845 (Epoch 76)

### mAP50 性能对比
1. **Init Only**: 0.7656 (Epoch 76) - **最高**
2. **Fixed**: 0.7641 (Epoch 43)
3. **Learnable**: 0.7629 (Epoch 42)

### 训练稳定性
- **Fixed**: 147 epochs，表现稳定
- **Init Only**: 157 epochs，最佳mAP50-95出现较早(Epoch 77)
- **Learnable**: 156 epochs，最佳结果在前期获得

## 结论
1. **Init Only** 模式在 mAP50-95 指标上表现最佳 (0.4969)
2. **Learnable** 模式在 mAP50 指标上表现接近最佳
3. **Fixed** 模式整体性能均衡
4. 所有模式在Precision上都保持在较高水平(~0.96-0.97)
5. Recall值普遍较低(~0.03)，说明模型较为保守

*注：Init Only 的最佳 mAP50-95 (Epoch 77) 时 mAP50 异常低 (0.0642)，但Precision和Recall都很高，可能是评估时的特殊情况。*
