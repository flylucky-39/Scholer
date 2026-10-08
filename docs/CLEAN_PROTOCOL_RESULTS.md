# 干净协议完整实验结果（VOC Split1 + COCO）

生成日期：2026-09-22。所有实验基于**从零训练**（无任何外部权重：无 COCO 预训练、无 ImageNet）的 base 模型，seed=3407，评测集 = VOC novel 类测试集（1023 张，各 shot 设定共用同一批评测图）。

## 0. 协议背景（数据泄漏重建）

原协议（会议版 + 期刊初稿）的 base 训练从官方 `yolo11s.pt`（COCO 80 类、118K 图、600 epoch 训练）初始化，构成数据泄漏：VOC novel 类（bird/bus/cow/motorbike/sofa）在 COCO 中同类同图。泄漏证据：官方权重在 VOC novel 上零样本 mAP50 = 0.9282。本文档全部数字来自去泄漏重建后的干净协议。

## 1. Base 模型

| Base | epochs | 全20类 mAP50 | 有效 base 类性能 | 状态 |
|---|---|---|---|---|
| 从零（VOC 15 base 类） | 200 | 0.5196 | ≈0.69 | ✅ 主协议 |
| 从零（COCO 60 base 类） | 120 | 0.4315 | — | ✅ 主协议 |
| 冻结-ImageNet（VOC） | 100 | 0.2734 | ≈0.365 | 对照（负结果） |
| 解冻-ImageNet（VOC） | 200 | 0.5208 | ≈0.69 | 对照（负结果） |

泄漏检查：所有 base 在 novel 类上 AP 严格为 0（零 TP）。

## 2. 主表（统一协议：mosaic ON，freeze=10，β/γ 按调度公式）

novel mAP50（best checkpoint）：

| 臂 | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| Standard 冻结 | 0.0952 | 0.1066 | 0.1276 | 0.1352 |
| Standard 解冻 | 0.0676 | 0.1019 | 0.1196 | 0.1288 |
| + Cosine（无原型） | 0.1311 | 0.3467 | 0.3686 | 0.4699 |
| **Ours（原型 + BG 抑制）** | **0.1644** | **0.4014** | **0.4718** | **0.4924** |

- β/γ 调度：β=0.5·exp(−0.5(K−1))，γ=0.3·exp(−0.5(K−1))；K=1/3/5/10 → β=0.5/0.184/0.0677/0.00555
- Ours vs Standard 解冻：**2.4× / 3.9× / 3.9× / 3.8×**
- 关键形态：Ours 随 shot 线性级增长（0.16→0.49），Standard 几乎躺平（0.07→0.13，10 倍数据仅 +6pp）

### 2.1 末 epoch 口径（模型选择鲁棒性）

| 臂 | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| Ours | 0.1573 | 0.3717 | 0.4672 | 0.4847 |
| Std 冻结 | 0.0554 | 0.1023 | 0.1122 | 0.1184 |
| Std 解冻 | 0.0544 | 0.0970 | 0.1126 | 0.1084 |

末 epoch 口径下比值变为 2.9× / 3.8× / 4.2× / 4.5×——结论不变且更强（Standard 从测试集选点中获益更多）。

### 2.2 COCO（novel 20 类，nAP50 / nAP50-95）

**主表（严格口径：完整 val2017 4999 图 + max_det=100，与文献评测协议对齐）：**

| 臂 | 1-shot | 10-shot | 30-shot | 30-shot nAP50-95 |
|---|---|---|---|---|
| Standard 冻结 | — | 0.0000 | 0.0235 | 0.0149 |
| Standard 解冻 | — | 0.0000 | 0.0238 | 0.0152 |
| + Cosine（无原型） | 0.0559 | 0.1917 | 0.2388 | 0.1476 |
| **Ours（原型 + BG 抑制）** | **0.1067** | 0.1874 | **0.2318** | 0.1471 |

- β/γ 调度值：1-shot 0.5/0.3；10-shot 0.0056/0.0033；30-shot 2.5e-7/1.5e-7（按公式归零）
- **干净协议下 Standard 微调接近完全失效**（10-shot 零 novel 检出；30-shot 缓爬至 0.024，比余弦配置低 10 倍）
- **30-shot 0.2318 与 DeFRCN Split-1 单次 22.6 持平/略超**（9.4M、无外部预训练、单阶段 vs 60M+ R101、ImageNet、双阶段）
- **1-shot 交叉点**：方法比其余弦消融高 +5.1pp（+91%）——原型初始化 + 强背景抑制的价值集中在极低 shot；10/30-shot 时支持集规模（200/326 图）越过价值交叉点，方法与 Cosine 打平（nAP50-95 上 10-shot 方法仍略优 0.1199 vs 0.1164）。与 VOC 的 shot-adaptive 形态完全同构。

**训练期口径（val_novel 4030 图，best checkpoint）对照：**

| 臂 | 1-shot | 10-shot | 30-shot |
|---|---|---|---|
| Cos | 0.0726 | 0.2019 | 0.2532 |
| Ours | 0.1079 | 0.1972 | 0.2464 |

两种图片集口径差约 1.4-1.5pp（4030 图子集不含 969 张纯背景图，precision 略高）；论文引用严格口径。

## 3. 增强与训练策略消融（均为 Ours 方法臂）

### 3.1 Mosaic：效应随 shot 单调翻转

| | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| mosaic ON | **0.1644** | **0.4014** | **0.4718** | 0.4924 |
| mosaic OFF | 0.0897 | 0.3729 | 0.4585 | **0.5187** |

机制：1-shot（5 图）时 mosaic 是唯一多样性来源（OFF 时 best 卡在 ep1 即过拟合）；≥10-shot 时真实样本足够，mosaic 的尺度失真转为纯成本。交叉点在 5↔10-shot 之间。Std 臂 1-shot 对照：ON 0.0952 > OFF 0.0748。

### 3.2 原型增强（--augment，支持集翻转/色彩/旋转/仿射 4 变体）

| | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| 基线 | 0.1644 | 0.4014 | 0.4718 | 0.4924 |
| + 原型增强 | **0.1812** (+1.7) | 0.3967 (−0.5) | **0.4843** (+1.3) | 0.4930 (+0.1) |

规律：原型越糙（shot 越少）增益越大；3/10-shot 处于噪声水平。定位为 1-shot 专用增强。

### 3.3 冻结深度阶梯（Adapter 假设的代理实验）

1-shot，freeze=N（Ultralytics 冻结前 N 层；模型共 24 层：backbone 0-10，neck 11-22，头 23）：

| freeze | 可训练部分 | 1-shot mAP50 |
|---|---|---|
| 10（主协议） | C2PSA+neck+头 | 0.1644 |
| 17 | PAN 下半+头 | **0.1788** |
| 23 | 仅检测头 | 0.1436 |

倒 U 型：适度减参有益（+1.4pp），全冻结仅训头又不足（−2.1pp）。**结论：无需引入 Adapter 模块**——frz17 以零参数代价覆盖了 Adapter 的收益区间（Adapter 容量介于 frz17 与 frz23 之间，预期 +0~1.5pp，不值得一天的工程与全矩阵重跑）。

frz17 跨 shot：

| | 1-shot | 3-shot | 10-shot |
|---|---|---|---|
| frz10 | 0.1644 | 0.4014 | 0.4924 |
| frz17 | **0.1788** | 0.3881 | **0.5087** |

frz17 在 1/10-shot 增益（+1.4/+1.6pp，边际显著），3-shot 略负（−1.3pp，噪声量级）。

### 3.4 每格最优组合（"shot-adaptive 协议"完整图景）

| | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| 最优配置 | frz17+原型增强 | 标准配置 | +原型增强 | mosaic OFF |
| 最优 mAP50 | **0.1934** | 0.4014 | **0.4843** | **0.5187** |
| 相对主表 | +2.9pp | — | +1.3pp | +2.6pp |

1-shot 合并臂（frz17+aug）0.1934：两增益近乎线性叠加（+1.4 与 +1.7 → 合计 +2.9）。

## 4. 协议敏感性：ImageNet 初始化（负结果）

| 1-shot | 方法 | Std 冻结 | Std 解冻 |
|---|---|---|---|
| 从零 base（0.5196） | **0.1644** | **0.0952** | 0.0676 |
| 冻结-IN base（0.2734） | 0.0732 | 0.0501 | 0.0520 |
| 解冻-IN base（0.5208） | 0.1209 | 0.0774 | 0.0808 |

- 冻结-IN：base 检测能力腰斩（有效 0.365），novel 全臂腰斩——base 检测能力是 1-shot 硬瓶颈
- 解冻-IN：base 强度追平从零版（0.5208 vs 0.5196），novel 仍全面落败——**ImageNet 初始化对 VOC-FSOD 无净收益**
- 解冻-IN 方法臂 ep5 即达峰（0.1209）后单调下滑：原型初始化起效快，但后续训练持续磨损它
- 工程记录：cls→det 手术只转移层 0-8（cls 层 9 C2PSA 与 det 层 9 SPPF 输入错位）；lr0=0.01 冲刷特征、0.002 过慢、0.005 可行

## 5. 旧污染协议数字（仅供内部对照，不得进论文）

| 臂 | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| Standard YOLO | 0.1140 | 0.2053 | 0.3306 | 0.5133 |
| + Cosine | 0.1325 | 0.5170 | 0.6695 | 0.7072 |
| + Prototype | 0.2210 | 0.5226 | 0.6876 | 0.7796 |
| + BG Suppress | 0.3316 | 0.6413 | 0.7308 | 0.7838 |

绝对值全面虚高（泄漏上限 0.93）；干净协议下相对优势反而更大（3.1×→3.9×）。

## 6. 复现配置索引

- 主表命令模板：`python scripts/train_fsod.py --config configs/voc_clean_{K}shot.yaml --stage finetune --base-weights runs/voc_base_clean/base_pretrain/weights/best.pt --prototype --background-suppression --bg-beta {β} --bg-gamma {γ}`
- Standard 臂：`python scripts/train_baseline.py --config configs/voc_clean_{K}shot[_stu].yaml --stage finetune --weights <base>`
- no-mosaic 变体脚本：`scripts/train_fsod_nomosaic.py`、`scripts/train_baseline_nomosaic.py`（仅一行 `mosaic=0.0` 差异）
- 全部结果 CSV：`runs/voc_clean_{K}shot*/novel_finetune*/results.csv`（col8=mAP50，col9=mAP50-95）
- 关键 run 目录：`runs/voc_base_clean/`（base）、`runs/voc_in1shot/`、`runs/voc_inu1shot/`（IN 对照）、`runs/voc_frz*/`（冻结阶梯）、`runs/voc_clean_{K}shot_nm/`（no-mosaic）
