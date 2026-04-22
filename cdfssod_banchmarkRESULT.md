# CD-ViTO DIOR 10-shot 复现结果

## 环境

- GPU: NVIDIA H20-3e
- PyTorch: 2.5.1+cu121
- Detectron2: RegionCLIP (editable install)
- xformers: 0.0.28.post2
- CUDA: 12.1

## 总体指标

| 指标 | 值 |
|------|-----|
| **AP** | **30.41** |
| **AP50** | **46.56** |
| **AP75** | **32.47** |
| APm | 12.37 |
| APl | 40.69 |

## 各类别 AP

| 类别 | AP |
|------|-----|
| stadium | 75.98 |
| chimney | 72.83 |
| airplane | 67.39 |
| storagetank | 55.89 |
| baseballfield | 58.48 |
| groundtrackfield | 46.02 |
| golffield | 37.64 |
| Expressway-toll-station | 32.49 |
| airport | 26.96 |
| tenniscourt | 25.36 |
| dam | 21.11 |
| basketballcourt | 16.20 |
| overpass | 14.63 |
| harbor | 12.34 |
| trainstation | 11.88 |
| Expressway-Service-area | 10.26 |
| windmill | 7.38 |
| bridge | 6.32 |
| vehicle | 6.57 |
| ship | 2.42 |

## 训练配置

| 参数 | 值 |
|------|-----|
| Model | ViT-L/14 + CD-ViTO |
| Backbone | DINOv2 ViT-L/14 |
| Dataset | DIOR 10-shot |
| IMS_PER_BATCH | 8 |
| BASE_LR | 0.001 |
| MAX_ITER | 500 |
| WARMUP_ITERS | 100 |
| LR_STEPS | (250, 400) |
| Prototype | prototypes_init/DIOR_10shot.vitl14.bbox.p10.sk.pkl |
| BG Prototype | weights/initial/background/background_prototypes.vitl14.pth |
| TOPK | 5 |
| ATTN_FUSE_RATIO | 0.7 |
| BG_CLS_LOSS_WEIGHT | 0.2 |

## 执行步骤

| 步骤 | 说明 | 状态 |
|------|------|------|
| 01 | 环境安装 (farline conda env) | 完成 |
| 02 | DIOR 数据集扁平化 (train: 18463, test: 5000) | 完成 |
| 03 | 权重下载 (3/3 就位) | 完成 |
| 04 | 启用 DIOR 注册 (builtin.py) | 完成 |
| 05 | Prototype 提取 + Sinkhorn 聚类 (20类, cls_acc=1.0) | 完成 |
| 06 | 训练 + 评测 (500 iter, 单卡 H20) | 完成 |
