# COCO Few-Shot Object Detection 实验结果汇总表

## COCO 10-shot

| Experiment                 |  P(%) |  R(%) | mAP50(%) | mAP50-95(%) | Best Epoch |
|----------------------------|-------|-------|----------|-------------|------------|
| base_pretrain              | 71.09 | 36.96 |   51.42  |    36.65    |     1      |
| novel_finetune             | 89.48 | 27.97 |   58.97  |    41.24    |     5      |
| novel_finetune_cosine      | 13.99 | 66.12 |   47.08  |    31.02    |    98      |
| novel_finetune_cosine_fused| 12.75 | 64.66 |   46.96  |    31.87    |    51      |
| novel_finetune_cosine_proto| 15.84 | 58.43 |   47.15  |    32.41    |   117      |

## COCO 30-shot

### 30shot_OLD

| Experiment                 |  P(%) |  R(%) | mAP50(%) | mAP50-95(%) | Best Epoch |
|----------------------------|-------|-------|----------|-------------|------------|
| novel_finetune             | 28.49 | 69.17 |   60.27  |    42.02    |     1      |
| novel_finetune_cosine      |  7.26 | 75.34 |   49.59  |    32.21    |    99      |
| novel_finetune_cosine_fused| 10.35 | 70.23 |   51.55  |    35.19    |    23      |
| novel_finetune_cosine_proto| 17.30 | 61.38 |   51.29  |    35.77    |    36      |

### 30shot_New

| Experiment                 |  P(%) |  R(%) | mAP50(%) | mAP50-95(%) | Best Epoch |
|----------------------------|-------|-------|----------|-------------|------------|
| novel_finetune             | 28.49 | 69.17 |   60.27  |    42.02    |     1      |
| novel_finetune_cosine      | 11.53 | 73.59 |   49.38  |    32.38    |    39      |
| novel_finetune_cosine_fused| 30.92 | 52.90 |   47.41  |    30.86    |    37      |
| novel_finetune_cosine_proto| 17.99 | 61.46 |   51.37  |    35.86    |    36      |

### 30shot_Vlm

| Experiment                          |  P(%) |  R(%) | mAP50(%) | mAP50-95(%) | Best Epoch |
|-------------------------------------|-------|-------|----------|-------------|------------|
| novel_finetune_cosine_fused         |  2.94 | 75.69 |   48.45  |    31.36    |    76      |
| novel_finetune_cosine_fused_fixed   |  2.92 | 75.43 |   48.51  |    31.82    |    67      |
| novel_finetune_cosine_fused_init_only| 93.44 |  6.43 |   49.69  |    32.54    |    37      |

### 30shot_baseline

| Experiment                 |  P(%) |  R(%) | mAP50(%) | mAP50-95(%) | Best Epoch |
|----------------------------|-------|-------|----------|-------------|------------|
| novel_finetune_cosine      |  7.51 | 73.96 |   48.95  |    31.78    |   161      |

### 30shot_exp

| Experiment                 |  P(%) |  R(%) | mAP50(%) | mAP50-95(%) | Best Epoch |
|----------------------------|-------|-------|----------|-------------|------------|
| novel_finetune_cosine      |  7.20 | 75.97 |   48.81  |    31.51    |   112      |
| novel_finetune_cosine_fused| 12.01 | 70.16 |   52.29  |    35.70    |    24      |
| novel_finetune_cosine_proto| 15.02 | 64.14 |   51.62  |    35.92    |    39      |
