# FSOD_VLM 代码架构说明

> 帮你理清这个项目到底在做什么。

---

## 一、最基础的问题

YOLO 做目标检测，标准流程是：

```
图片 → backbone → neck → 分类头(Conv2d) → 判断"这是什么类"
```

这个**分类头**本质上是一层卷积：`score = 权重 · 特征 + 偏置`

它同时看特征的**方向**和**长度**。

**Few-shot 的问题**：每类只有 10 张图，特征向量的"长度"很不稳定（同一个类，不同图片提取出来的特征模可能差很多），导致权重不知道怎么调。

---

## 二、你的第一个改动：余弦分类器

你把最后的 `nn.Conv2d` 换成了 `CosineConv2d`：

```python
# 标准分类器
score = W · x + b         # 看方向 + 长度

# 余弦分类器
score = s * cos(W, x) + b  # 只看方向
```

做法是：把权重和特征都做 L2 归一化（拧成单位长度），只保留方向，去掉长度。

**效果**：原来 noisy 的长度信号被剔除，分类器只根据"方向"做判断。

**代价**：有些情况下长度也是有信息的（模型确认程度），去掉后信息量减少。

---

## 三、你的第二个改动：Prototype 初始化

换成余弦分类器后，novel 类的分类权重还是随机初始化的。10 张图要从随机方向学起，很难学准。

**Prototype 的做法**：

```
base pretrain 的 backbone（已经会看图了）
        ↓
    过 support set 的图片
        ↓
    在分类头倒数第二层把特征抠出来（256维）
        ↓
    同一类的特征取平均 → "该类长得像什么方向"
        ↓
    把这个方向作为分类权重的初始值
```

这样 novel 类一开始的权重就不是瞎蒙的，而是知道个大概方向了。

**代码位置**：`modules/prototype.py`

---

## 四、你的第三个改动：Florence-2 文本原型

Visual prototype 依赖 backbone 的特征空间。如果 base 类和 novel 类长得完全不一样（比如 VOC 的自然图片 → DIOR 的卫星图），backbone 的特征空间就失灵了，提出来的视觉原型不准。

**VLM 文本原型解决了什么问题**？"车"这个单词的文本向量，在 VOC 空间里长这样，在 DIOR 空间里也长这样 — **不受 domain gap 影响**。

**你们的实现方式**（`modules/adaptation.py`）：

这套流程分三步：

### 第一步：提取文本向量

```
Florence-2 文本编码器
        ↓
    把类名（"cat"、"dog"）变成 1024 维向量
```

### 第二步：训练一个 FiLM 映射网络

文本向量（1024维）和视觉特征（256维）不在同一个空间，没法直接当权重用。

所以你在 **base 类**上训练了一个映射网络：

```
文本向量(1024)
    ↓
查找到 该类 的文本描述（captions）→ 编码成文本描述向量
    ↓
    和 该类 的 visual prototype 一起输入 FiLM 网络
    ↓
    输出 cosine head 权重的预测值
    ↓
    和 base 类上 pretrain 好的真实权重对比 → 计算 loss → 更新
```

### 第三步：用在 novel 类上

FiLM 网络在 base 类上训练完成后，直接输入 novel 类的文本描述向量和 visual prototype，输出 novel 类的初始化权重。

### 融合方式

你有两种原型（visual prototype + Florence-2 text prototype），代码里用可学习的 alpha 把它们融合：

```
最终权重 = alpha * 文本原型 + (1 - alpha) * 视觉原型
```

alpha 在 finetune 过程中自动调节。

---

## 五、三个实验的对应关系

| 实验名 | 做了什么 | 对应代码 |
|--------|----------|----------|
| `cosine` | 换余弦分类头，随机初始化 | `CosineConv2d` |
| `cosine_proto` | 余弦 + 视觉原型初始化 | `prototype.py` |
| `cosine_florence2` | 余弦 + Florence-2 文本原型初始化 | `adaptation.py` |
| `cosine_fused` | 余弦 + 视觉原型 + 文本原型 融合 | `prototype.py` + `adaptation.py` |
| `fused_mt1/2/3` | 多模板：一个类多条文本描述 | 同 fusion，multi-prompt |

---

## 六、还有一个分支：Dual-path 后融合

除了在训练时用 VLM 初始化权重（如上所说），你还有另一条线（`modules/dual_path_fusion.py`）：

```
推理时：
    YOLO 自己预测一个框（带置信度）
        +
    CLIP 验证器也给这个框打一个分
        ↓
    两种分数按策略融合（加权/相乘等）
```

这个是在推理阶段补救 — YOLO 检测出来的框，再拿 CLIP 验一遍，如果 CLIP 说这个框不像该类，就压低估；如果说很像，就抬高分。

对应 `runs/direction_c/dualpath_results.csv` 的数据，测的是不同融合策略（fixed/multiplicative、gamma、temperature）对结果的影响。

---

## 七、整个项目的图片流程

```
                   训练阶段                                   推理阶段
  ┌─────────────────────────────────────────────┐   ┌───────────────────┐
  │                                             │   │                   │
  │  base pretrain                              │   │  训练好的 YOLO-FSOD │
  │  (YOLO 在 base 类上训，数据充足)               │   │  (cosine head)    │
  │       ↓                                     │   │       ↓           │
  │  得到好用的 backbone + neck                   │   │  检测出框 + 分类    │
  │       ↓                                     │   │       ↓           │
  │  ┌── support set                            │   │  (可选的)          │
  │  │  ↓                                       │   │  CLIP 验证器二次打分│
  │  │  visual prototype ← backbone 提特征取均值  │   │       ↓           │
  │  │                                           │   │  融合/不融合       │
  │  │  Florence-2 文本原型 ← 类名 → 文本编码器    │   │       ↓           │
  │  │       ↓                                   │   │  最终检测结果      │
  │  │  FiLM 网络映射到视觉空间                     │   │                   │
  │  │       ↓                                   │   └───────────────────┘
  │  └──→ 融合：alpha * 文本 + (1-alpha) * 视觉    │
  │              ↓                                │
  │  初始化 CosineConv2d 的权重                     │
  │              ↓                                │
  │  novel 类 finetune                            │
  │  (在 support set 上训，cosine head 微调)        │
  └─────────────────────────────────────────────┘
```

---

## 八、从你的实验数据能看到什么

### 同域（VOC）

Visual prototype 本身就很强（base 和 novel 长得像），加 VLM 文本几乎没提升。

```
cosine         49.25
+ proto        51.30  ↑
+ florence2    50.10  ±
+ fused        51.98  ↑（但和 proto 差距很小）
```

结论：**在同域上，visual prototype 已经把信息吃透了，VLM 没有额外信息可加。**

### 跨域（DIOR）

Visual prototype 在低 shot 下基本不 work：
```
1-shot: mAP50-95 = 4.93
5-shot: mAP50-95 = 20.59
10-shot: mAP50-95 = 68.30
```

**这里缺失了 Florence-2 和 fused 的实验数据** — 按原理推测，VLM 文本原型在 1-shot 和 5-shot 上应该比 visual-only 好得多，因为文本向量不受"卫星图和日常图长不一样"的影响。

---

## 九、一句话总结

这个项目做了三件事：
1. **余弦分类器** — 把 noisy 的长度信号扔掉，只保留方向
2. **Prototype 初始化** — 用 support set 给 novel 类一个比随机好得多的起点
3. **VLM 文本原型** — 用 Florence-2 的文本编码器绕过 domain gap，让跨域 few-shot 有个靠谱的起点

**VLM 应该在跨域场景发力，但你的实验数据目前还缺这块的验证。**
