# FSOD_VLM — Shot-Adaptive 背景抑制的小样本目标检测（v0.2.0）

基于 YOLO11s 的少样本目标检测（FSOD）：**原型初始化余弦分类头 + shot-adaptive 背景抑制**，base 模型从零训练（无 COCO 预训练、无 ImageNet 初始化）。

## ⚠️ 唯一可信实验数据源

**[`docs/CLEAN_PROTOCOL_RESULTS.md`](docs/CLEAN_PROTOCOL_RESULTS.md)（干净协议，2026-09-22）**

- 原协议（≤2026-05）的 base 训练从官方 `yolo11s.pt`（COCO 80 类）初始化，构成**数据泄漏**（VOC novel 类零样本 mAP50=0.93），全部旧数字作废
- 旧文档已归档至 `docs/archive/legacy_2026-05/`（带弃用横幅），旧 runs 已归档至 `runs/archive/`，**均不得引用**
- 新协议 run 的原始训练曲线 CSV：`docs/clean_protocol_csv/`（51 个 run + 严格评测记录）

## 核心结果（VOC Split1 novel mAP50，seed=3407，best checkpoint）

| 臂 | 1-shot | 3-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| Standard 解冻 | 0.0676 | 0.1019 | 0.1196 | 0.1288 |
| + Cosine（无原型） | 0.1311 | 0.3467 | 0.3686 | 0.4699 |
| **Ours（原型 + BG 抑制）** | **0.1644** | **0.4014** | **0.4718** | **0.4924** |

- Ours vs Standard 解冻：2.4× / 3.9× / 3.9× / 3.8×
- COCO 30-shot 严格口径（完整 val2017 + max_det=100）nAP50 = **0.2318**，与 DeFRCN Split-1 持平/略超（9.4M 参数、无预训练、单阶段）
- 每格最优 shot-adaptive 组合：1-shot 0.1934 / 3-shot 0.4014 / 5-shot 0.4843 / 10-shot 0.5187

## 复现入口

```bash
# 主表（Ours）
python scripts/train_fsod.py --config configs/voc_clean_{K}shot.yaml --stage finetune \
  --base-weights runs/voc_base_clean/base_pretrain/weights/best.pt \
  --prototype --background-suppression --bg-beta {β} --bg-gamma {γ}

# β/γ 调度公式：β = 0.5·exp(−0.5(K−1))，γ = 0.3·exp(−0.5(K−1))
# Standard 臂：scripts/train_baseline.py；no-mosaic 变体：train_*_nomosaic.py
```

完整命令索引与全部消融（mosaic 翻转、原型增强、frz 冻结阶梯、ImageNet 敏感性）见 `docs/CLEAN_PROTOCOL_RESULTS.md` §6。

注：干净协议的完整 runs 目录（含权重）在训练服务器上；本地仓库以 `docs/clean_protocol_csv/` 的 CSV 为准。

## 目录结构

```
fsod/                            # Python 包（voc/coco 数据集注册，modules: cosine_head / prototype / background_suppression 等）
scripts/                         # 训练 / 评测 / 数据准备入口
configs/                         # 配置（voc_clean_*、coco_*_clean* 为干净协议；旧方向配置仍在但对应 runs 已归档）
docs/CLEAN_PROTOCOL_RESULTS.md   # ⭐ 唯一可信结果文档
docs/clean_protocol_csv/         # 新协议 run CSV（bases / voc / coco）
docs/archive/                    # 旧文档归档（弃用，勿引用）
runs/archive/                    # 旧协议 runs 归档（数字作废）
third_party/ultralytics/         # vendored Ultralytics（含 FSODDetect 支持）
```
