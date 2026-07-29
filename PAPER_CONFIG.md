# Shot-Adaptive Background Suppression for Few-Shot Object Detection

## 论文实验配置详情

---

### 1. 数据集

- **PASCAL VOC 2007 + 2012** (trainval 07+12, test 07)
- **Split 1** (Novel: bird, bus, cow, motorbike, sofa; Base: 其他 15 类)
- **Shot 设置**: 1-shot, 3-shot, 5-shot（每类 K 张标注图片）
- 训练图片 size: 640×640

---

### 2. 模型架构

- **Backbone**: YOLO11s (width=0.5, 9.4M 参数)
- **Neck**: YOLO11 FPN (P3/8: 128ch, P4/16: 256ch, P5/32: 512ch)
- **检测头**: FSODDetect — 将标准 Detect head 的 cv3 分类分支最后一层替换为 **CosineConv2d**

#### CosineConv2d

```
输入: 特征图 x ∈ R^(B, C, H, W)
权重: W ∈ R^(Nc, C)  (L2-normalized)
输出: score = exp(scale_c) * cos(x, W_c) + bias_c

推理时加入背景抑制:
score = exp(scale_c) * [cos(x, W_c) - γ * cos(x, b)] + bias_c
```

其中 b 是背景原型向量，γ 是推理时抑制强度。

---

### 3. 训练流程

#### Step 1: Base Pretrain (100 epochs)
- 在 15 个 base 类上标准 YOLO 训练
- Batch size: 64, lr=0.01, SGD
- 产出: `base_pretrain_cosine/weights/best.pt`

#### Step 2: 背景原型提取 (offline)
- 从 base_train 中采样 200 张图片
- 提取非目标区域（不与任何 GT bbox 重叠的空间位置）的 cv3 特征
- 均值池化 → 背景原型向量 b ∈ R^128

#### Step 3: 原型提取与正交化
- 从 support set (novel_finetune) 中提取每类视觉原型 p_cls
- Shot-Adaptive 正交化:

```
β = β_max * exp(-k * (shot - 1))    其中 β_max=0.5, k=0.5
p'_cls = normalize(p_cls - β * <p_cls, b̂> * b̂)
```

#### Step 4: Novel Finetune (200 epochs)
- 初始化 CosineConv2d 权重为 p'_cls
- 注入推理时背景抑制 (γ)
- Batch size: 4, lr=0.0005, AdamW
- Freeze backbone: 前 10 epochs
- cos_lr: True
- patience: 30

---

### 4. Shot-Adaptive 公式

| Shot | β (原型正交化) | γ (推理抑制) | 效果 |
|------|---------------|-------------|------|
| 1 | 0.500 | 0.300 | 强抑制 |
| 2 | 0.303 | 0.182 | 过渡 |
| 3 | 0.184 | 0.110 | 中等 |
| 5 | 0.068 | 0.041 | 微弱 |
| 10 | 0.006 | 0.003 | 几乎关闭 |

**设计动机**: 1-shot 时原型噪声大，cos(p, b) 高，需要强抑制；shot 越多原型越干净，抑制应自然衰减。指数衰减公式使得不需要 per-shot 调参。

---

### 5. 实验结果

| Shot | 方法 | β | γ | mAP50 | mAP50-95 | ΔmAP50 |
|------|------|---|---|-------|----------|--------|
| 1-shot | Baseline | — | — | 0.2732 | 0.1545 | — |
| 1-shot | **+ bg_suppress** | 0.50 | 0.30 | **0.3316** | **0.1991** | **+21.4%** |
| 3-shot | Baseline | — | — | 0.6515 | 0.4611 | — |
| 3-shot | + bg_suppress (fixed) | 0.50 | 0.30 | 0.6372 | 0.4448 | −2.2% |
| 3-shot | + bg_suppress (adapt) | 0.18 | 0.11 | 0.6413 | 0.4461 | −1.6% |
| 5-shot | Baseline | — | — | 0.7160 | 0.5164 | — |
| 5-shot | **+ bg_suppress (adapt)** | 0.07 | 0.04 | **0.7308** | **0.5286** | **+2.1%** |

---

### 6. 消融分析要点

1. **固定 β=0.5 只在 1-shot 有效，3-shot 上严重负面** (−2.2%)
2. **Shot-Adaptive 让 bg_suppress 成为跨 shot 方法**：通过指数衰减，1-shot 强抑制 (+21.4%)，5-shot 微弱抑制 (+2.1%)
3. **3-shot 仍然微负** (−1.6%)，可通过进一步调优 decay_rate 改善
4. **cos_lr 在 1-shot 无额外增益，3-shot 微正** (+0.8%)
5. **P/R trade-off**: 背景抑制大幅提升 Recall 但降低 Precision，符合"减少 FN 但增加 FP"的预期

---

### 7. 方法对比（已尝试的全部方向）

| 方法 | 机制 | 1-shot | 3-shot | 5-shot | 结论 |
|------|------|--------|--------|--------|------|
| Baseline (cosine+proto) | 静态原型 | 0.273 | 0.652 | 0.716 | 基线 |
| bg_suppress (fixed) | 减法/正交化 | **+21.4%** | −2.2% | N/A | 仅1-shot |
| bg_suppress (shot-adaptive) | 自适应减法 | **+21.4%** | −1.6% | **+2.1%** | 跨shot |
| PrototypeAdapter (未训练) | 加法/动态原型 | −2.1% | N/A | N/A | 失败 |
| PrototypeAdapter (base训练) | 学习加法 | −50.8% | N/A | N/A | 失败 |
| Focus Module (base训练) | 特征调制 | 崩溃 | N/A | N/A | 失败 |
| cos_lr | LR调度 | +0.2% | +0.8% | N/A | 辅助 |

---

### 8. 关键代码文件

| 文件 | 作用 |
|------|------|
| `fsod/fsod/modules/cosine_head.py` | CosineConv2d + FSODDetect（含背景抑制推理） |
| `fsod/fsod/modules/background_suppression.py` | 背景原型提取 + 正交化 |
| `fsod/fsod/modules/prototype.py` | 原型提取 + 初始化 |
| `fsod/scripts/train_fsod.py` | 训练入口（--background-suppression --bg-beta --bg-gamma） |
| `conditional_prototype/modules/adaptive_bg_suppress.py` | Shot-Adaptive 公式 |
| `conditional_prototype/modules/support_encoder.py` | 支持集编码器（废弃） |
| `conditional_prototype/modules/focus_module.py` | 特征聚焦模块（废弃） |
| `conditional_prototype/modules/prototype_adapter.py` | 条件原型适配器（废弃） |