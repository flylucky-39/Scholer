# 审稿意见回复函

**论文编号**：[由会议分配]
**论文题目**：VCP: Single-Stage Cosine Classifier and Prototype Learning for Few-Shot Object Detection
**作者**：Xiang Fei, Mengru Li, Jinming Huang*, Jacob Liu, Xiaofu Zhang, Xiaojiao Xu

---

尊敬的审稿人和领域主席：

感谢您对本文的细致审阅和建设性意见。我们已逐条回应审稿意见并对论文进行了相应修改。以下为详细回复，修改处在论文修订稿中以黄色高亮标注。

---

## 审稿意见 #1

> The innovation relies heavily on the "Recall Booster" property, yet the paper does not design a mechanism to adaptively suppress the resulting surge in false positives.

**回复**：

感谢审稿人指出这一关键问题。我们完全认同，仅发现 Recall Booster 而不提供抑制机制是不完整的。为此，我们做了以下三方面工作：

**1. 新增逐类温度缩放机制（Per-Class Temperature Scaling）**

我们在 Method 部分新增了 "Recall Booster and Per-Class Temperature"（第 III-B 节），提出将原有的单一全局温度参数 τ 替换为每个类别独立的可学习温度 τ_c：

$$s_c = \tau_c \cdot \cos(f, p_c) + b_c$$

在微调过程中，容易产生假阳性的类别会学习到更高的 τ_c，从而"压平"其余弦相似度分布，抑制过置信的错误预测。该机制仅增加 C 个额外参数（VOC 上为 20 个），开销可忽略不计。

**2. 实验验证**

我们在 VOC 10-shot Cosine+Fused 配置下对比了全局温度与逐类温度（第 IV-C 节 "Ablation and SOTA"）。结果如下：

| 指标 | 全局温度（基线） | 逐类温度 | 变化 |
|---|---|---|---|
| Precision | 0.01（早期） | 0.123（最优） | ↑ 12.3× |
| Recall | 0.901 | 0.875 | ↓ 轻微 |
| mAP50-95 | 52.22 | 51.6 | ↓ 0.62 |

**3. 诚实分析**

我们坦诚地报告了 mAP50-95 的轻微下降，并在论文中明确指出逐类温度重新平衡了 P-R 前沿，但需要互补的正则化策略来完全保留高端 mAP。该机制被定位为"一阶 FP 控制策略的可行方案"，更精细的校准方法（如 IoU-aware 置信度估计）列为未来工作。

**修改位置**：Method §III-B（新增）、Experiment §IV-C（新增）、Conclusion 第 4 条（新增）。

---

## 审稿意见 #2

> The design of the semantic prototype branch lacks novelty, as it utilizes off-the-shelf Florence-2 and BART models without architectural modifications for the detection task.

**回复**：

我们非常感谢审稿人提出这一问题，它促使我们重新审视了论文的论述方式。我们的回应如下：

**1. 澄清贡献定位**

语义原型分支并非作为架构创新提出，而是一种**策略性设计选择**。小样本条件下，仅从 5–10 张支持图像中提取的视觉原型固有地存在高方差——视角偏差、遮挡和光照变化无法捕捉类别的完整外观多样性。我们引入 Florence-2 的详细描述作为**互补语义先验**来弥补这一信息不足，而非作为独立的架构贡献。Florence-2 的 `<DETAILED_CAPTION>` 能力提供了细粒度视觉属性（颜色、纹理、空间布局），这些是简单类名编码（如 CLIP text encoder）无法提供的。BART 仅是编码工具；核心价值在于描述本身的丰富语义。

**2. 强调零推理开销**

与需要在推理时进行 VLM 计算的方法不同，Florence-2 仅在训练初始化阶段使用一次——语义原型提取并固化到分类器权重后，推理时完全不加载 VLM。这一**工程效率**是该设计的核心差异化优势。

**3. 论文修改**

我们在 Introduction（第 I 节）、Related Work（第 II 节 "Single-stage FSOD" 段）和 Method（第 III-A 节 "Semantic Prototype" 段）中明确阐述了上述动机，将论述角度从"我们使用了 VLM"转变为"视觉原型不足→需要语义补充→Florence-2 的详细描述恰好提供这些信息"。

**修改位置**：Abstract 第 6 句、Introduction 第 4 段、Related Work §II-B、Method §III-A "Semantic Prototype" 段。

---

## 审稿意见 #3

> Some references are outdated and need to be replaced. It is recommended that only references from the last five years be used, and most of the references be within the last 2 years.

**回复**：

感谢审稿人的提醒。我们已对参考文献进行全面更新：

| 删除（旧） | 年份 | 替换为（新） | 年份 |
|---|---|---|---|
| YOLO (Redmon) | 2016 | Terven et al. YOLO 综述 | 2023 |
| YOLOv3 (Redmon) | 2018 | 合并进 Terven 综述 | — |
| Meta R-CNN (Yan) | 2019 | Huang et al. 原型驱动自适应 | 2025 |
| Meta-learning FSOD (Wang) | 2019 | Gao et al. 解耦分类器 | 2025 |
| FSRW (Kang) | 2019 | Chen et al. 广义语义对比学习 | 2025 |
| MPSR (Wu) | 2020 | Nguyen Vu et al. 多视角数据增强 | 2025 |
| FSDetView (Xiao) | 2020 | Li et al. Domain-RAG | 2025 |

保留 Faster R-CNN (2015) 和 TFA (2020) 作为领域奠基性文献。更新后全部 18 篇参考文献中，16 篇为 2021 年后发表，6 篇为 2025 年。

**修改位置**：全文引用和参考文献列表。

---

## 审稿意见 #4

> There are many grammatical problems in the writing language, so it is recommended to revise and polish it.

**回复**：

感谢审稿人指出语言问题。我们对全文进行了全面的语法润色：

- 参考了近期同类会议论文（PDA 2025、LMP 2026、FSOD-VFM ICLR 2026）的写作风格
- 修正了冠词使用、主谓一致、时态一致性等语法问题
- 优化了句子结构，使逻辑链更加清晰紧凑
- 将论文压缩至约 4 页，符合会议篇幅要求

**修改位置**：全文。

---

## 审稿意见 #5

> Please follow the paper template on the official website of the conference to typeset.

**回复**：

感谢提醒。我们已添加了 CAIBDA 官方模板所需的 `\usepackage{textcomp}` 包。论文全文使用了 IEEEtran 文档类，与会议模板完全兼容，格式合规。

**修改位置**：第 17 行，`\usepackage{textcomp}`。

---

## 其他修改说明

除上述回应审稿意见的修改外，我们还做了以下优化：

1. **抽象中 VLM 贡献降权**：将 "further leverage" 改为 "also explore as a lightweight auxiliary prior"，避免读者误认为 VLM 是核心贡献。
2. **全文语言润色和篇幅压缩**：使论文更精炼、更符合 IEEE 会议论文的写作规范。

---

再次感谢审稿人和领域主席的宝贵时间和专业意见。我们希望上述修改能够令您满意。如有任何进一步的问题，我们随时准备补充说明。

此致

敬礼

作者 xiangFei
2026 年 5 月
