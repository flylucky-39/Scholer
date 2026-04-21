# Real-Time Cross-Domain Few-Shot Object Detection
## (Idea A) 实验计划与方案文档

> 本文档不修改任何已有代码，仅作为新方向的规划。落地实现时建议在新 worktree（如 `FSOD_LLM_cdfsod`）中创建新脚本。

---

## 1. 问题陈述

**研究问题**：
当前 FSOD 领域的主要矛盾：
- **精度**导向：DE-ViT/CD-ViTO 等 SOTA 用 ViT-L 重型 backbone，单图推理 100ms+，工业完全不可用
- **实时**导向：YOLO 系列在 FSOD 场景几乎无人研究
- **跨域**：source domain (COCO) → target domain（医疗/工业/水下/卫星）时，纯视觉 prototype 严重失真

**核心命题**：
> 在 in-domain FSOD 中视觉 prototype 已经足够好（VLM 提供的语义先验冗余）；
> 但在 cross-domain FSOD 中视觉 prototype 失效（域漂移），此时 VLM 的语义先验是真正稀缺的、跨域稳定的信号。
> 因此 **VLM 引导应该作为"域漂移自适应"的工具**，而非通用增强模块。

---

## 2. 解释你过去 6 个月负结果的合法性

| 实验 | 设定 | VLM 增益 | 解释 |
|---|---|---|---|
| COCO 30-shot in-domain | source=COCO base, target=COCO novel | ~0% | base prototype 已经足够好 |
| COCO 30-shot in-domain | 同上 | 不稳定 | 视觉信号已饱和，VLM 信号被淹没 |
| **CD-FSOD（待验证）** | source=COCO, target=DIOR/DeepFish/... | **预期 +5~15 mAP** | 视觉 prototype 失效，VLM 语义稳定 |

**这正是 motivation 的灵魂**：
> "We observe that VLM guidance brings marginal gain on in-domain FSOD (Table X), but yields substantial improvement when domain shift is large (Table Y), suggesting VLM should be applied **selectively** based on domain gap."

---

## 3. 方法设计

### 3.1 框架总览

```
                ┌─── Source domain (COCO) pretrain ───┐
                │       YOLO11s + Cosine head         │
                └──────────────┬──────────────────────┘
                               │ frozen backbone
                               ▼
       ┌───────────────────────────────────────────────────┐
       │         Few-shot target domain (K shots)          │
       │                                                   │
       │   Visual branch:  RoIAlign → mean → P_v(c)        │
       │   Textual branch: Florence-2 caption → MLP → P_t(c)│
       │                                                   │
       │   Domain-Gap Estimator (DGE):                     │
       │     g = 1 - cos(CLIP_img(target), CLIP_img(source))│
       │                                                   │
       │   Adaptive fusion:                                │
       │     P(c) = (1-α(g)) * P_v(c) + α(g) * P_t(c)      │
       │     α(g) = sigmoid(k * (g - g0))                  │
       │                                                   │
       │   Init CosineConv2d weights for novel classes     │
       └───────────────────────────────────────────────────┘
```

### 3.2 三个真正的创新点

1. **Domain-Gap Estimator (DGE)**
   - 用 frozen CLIP image encoder 计算 source 与 target support 图像的全局相似度
   - 输出标量 `g ∈ [0, 1]`，0 = 同分布，1 = 完全异质
   - **零额外训练成本**

2. **Domain-Adaptive Fusion (DAF)**
   - α 不再是 learnable scalar，而是 `g` 的可学习函数 `α(g) = σ(k(g - g0))`
   - 只学 2 个参数 `k, g0`，避免过拟合
   - 在 base pretrain 阶段就在合成 domain shift（augmentation）下学好

3. **Cross-Domain Prototype Calibration (CDPC)**
   - 当 g 大时，对视觉 prototype 做"拉向文本中心"的微调
   - 公式：`P_v ← P_v + λ(g) * (P_t - P_v) / ||P_t - P_v||`
   - 防止视觉离群点污染原型

### 3.3 模块复用映射

| 现有代码模块 | 在新方案中的角色 | 改动 |
|---|---|---|
| `fsod/modules/cosine_head.py` | 分类头 | 0 改动 |
| `fsod/modules/prototype.py` | 视觉 prototype 提取 | 0 改动 |
| `fsod/modules/adaptation.py` (FiLM) | 文本 prototype 生成 | 微调：去掉 FiLM 复杂结构，简化为 MLP |
| `scripts/train_fsod.py` | 训练循环 | 新建 `scripts/train_cdfsod.py` |
| Florence-2 caption pipeline | text 信号源 | 0 改动 |

---

## 4. 实验计划

### 4.1 数据集（必须有）

**主基准：CD-FSOD-Bench**（CVPR 2024 NTIRE Challenge）
- Source: COCO base 60 类
- Target (6 个领域，每个 1/5/10-shot)：
  - **ArTaxOr**：节肢动物分类
  - **Clipart1k**：剪贴画
  - **DIOR**：遥感（航拍）
  - **DeepFish**：水下鱼
  - **NEU-DET**：钢铁表面缺陷
  - **UODD**：水下垃圾

**辅助基准（in-domain 对照）**：
- COCO novel 20 类（保留你现有结果）
- VOC split-1/2/3（FSOD 标配，必须有）

### 4.2 对比方法（baseline）

| 类别 | 方法 | 来源 |
|---|---|---|
| 经典 FSOD | TFA, DeFRCN, FSCE | 都有公开代码 |
| Transformer FSOD | Meta-DETR, DE-ViT | 有公开代码 |
| **CD-FSOD SOTA** | **CD-ViTO** | ECCV'24, 必须比 |
| OV 检测 | GroundingDINO (zero-shot) | 公开 |
| 你的方法（消融） | YOLO+cosine, +proto, +text, +DGE, +DAF, +CDPC | 自己 |

### 4.3 主表设计（提前规划 Table 1）

```
Table 1: Cross-Domain FSOD on CD-FSOD-Bench (mAP50)

Method        Backbone    Params  FPS  | ArTaxOr Clipart DIOR DeepFish NEU UODD | Avg
─────────────────────────────────────────────────────────────────────────────────
DE-ViT        ViT-L       300M    3    | xx.x   xx.x   xx.x ...
CD-ViTO       DINOv2-L    300M    3    | xx.x   xx.x   xx.x ...
─────────────────────────────────────────────────────────────────────────────────
YOLO11s+TFA   YOLO11s     10M     80   | xx.x   xx.x   xx.x ...
YOLO11s+ours  YOLO11s     10M     78   | xx.x   xx.x   xx.x ...   ← target
```

**关键卖点**：参数 10–30 倍小、速度 25 倍快，精度差距控制在 5 mAP 以内。

### 4.4 消融实验（提前规划 Table 2）

| Variant | Visual Proto | Text Proto | DGE | DAF | CDPC | Avg mAP |
|---|---|---|---|---|---|---|
| (a) | ✓ | – | – | – | – | baseline |
| (b) | ✓ | ✓ (fixed α=0.5) | – | – | – |  |
| (c) | ✓ | ✓ | ✓ | – | – |  |
| (d) | ✓ | ✓ | ✓ | ✓ | – |  |
| (e) **full** | ✓ | ✓ | ✓ | ✓ | ✓ |  |

### 4.5 稳定性验证（regulator-friendly）

每个数字 **3 个 seed 平均 ± 标准差**。这是当前 FSOD 论文最缺的，也是审稿人最容易抓的。

### 4.6 时间预算估计（不给具体小时数，给阶段）

| 阶段 | 任务 |
|---|---|
| P1 | 跑通 CD-FSOD-Bench 数据加载 + YOLO11s 纯 baseline 在 6 个 target 上的结果 |
| P2 | 实现 DGE，验证 g 与 mAP-drop 的相关性（一张关键的 motivation 图） |
| P3 | 实现 DAF + CDPC，跑完整 ablation |
| P4 | 复现 CD-ViTO 等对比方法，跑 SOTA 对照 |
| P5 | 跨 3 seeds 重跑主表 |
| P6 | 写作 + 投稿 |

---

## 5. 投稿策略

### Tier 1（advisable target）
- **WACV 2026** (deadline ~ Aug 2025)
- **BMVC 2025** (deadline ~ Jul 2025)
- **ICCV 2025 Workshop on CD-FSOD / NTIRE**

### Tier 2（如果主表碾压）
- **ICCV 2025 / CVPR 2026 main track**
- **Pattern Recognition Journal**

### Tier 3（保底）
- **ACCV 2026, ICIP 2025**

---

## 6. 立即可做的第一步（无需写新代码）

1. 访问 https://github.com/lovelyqian/CDFSOD-benchmark 下载 6 个数据集
2. 用现有 `scripts/train_fsod.py` 改一下数据路径，跑纯 cosine baseline 在 ArTaxOr / DIOR 两个最具代表性的 domain
3. 同时跑 cosine + 现有 VLM init（你已有的实现）
4. 看 VLM 是否在跨域设置下涨点（这个结果决定整个 idea 是否成立）

如果第 4 步 VLM 跨域涨 5+ mAP → 立即推进
如果 VLM 跨域不涨甚至跌 → 立即转 Idea B（稳定性分析）或 Idea C（OV 蒸馏）

---

## 7. 风险评估

| 风险 | 概率 | 缓解 |
|---|---|---|
| YOLO11s 在 CD-FSOD 上太弱（base 表征不够） | 中 | 用 YOLO11l 或 backbone 替换为 DINOv2 small |
| CD-ViTO 已经基本饱和 | 中 | 主打"实时" + "10x 小" 卖点 |
| Florence-2 caption 在跨域图像上质量崩盘 | 中 | 备用 SigLIP / CLIP-text，用类名直接编码 |
| g 估计不可靠 | 低 | 用多种相似度（CLIP/DINOv2/MAE）融合 |
| reviewer 质疑"VLM 早就被用过" | 高 | 强调 **selective use based on domain gap**，并诚实展示 in-domain 负结果 |

---

## 8. 与现有项目的关系

- 现有 COCO 30-shot 实验数据 → 写论文时作为 **in-domain 负结果对照**（"VLM marginal on in-domain"）
- 现有 EMA bug 修复经历 → 写在 Implementation Details 强调 reproducibility
- 现有 cosine + prototype 模块 → 直接复用为新方法的 baseline 组件

整个新方向**对现有代码 0 破坏**，纯加法。
