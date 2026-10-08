> ⚠️ **已归档 · 旧协议数据（≤2026-05）**
> 本文件产生于**数据泄漏协议**（base 从 yolo11s.pt——COCO 80 类 600 epoch 官方权重——初始化，VOC novel 类零样本 mAP50=0.93），文中全部数字**不可用于论文或实验对比**。
> 唯一可信数据源：[`docs/CLEAN_PROTOCOL_RESULTS.md`](../../CLEAN_PROTOCOL_RESULTS.md)（干净协议，2026-09-22）。归档：2026-10-08。

# FSOD_VLM 项目交接文档

> 用途：替代当前 AI 助手时的全量上下文交接。
> 最后更新：2026-04-23
> 工作机：本地 Windows + 离线服务器 (`/root/epfs/07_FSOD_LLM/`)

---

## 0. 一句话现状

**Phase 1 (DAF) 已完成且强**；**Phase 2 (Dual-Path 方向 C v1) 已完成单数据集消融**，发现"CLIP 增益与 shot 数成反比"的清晰规律，**待做跨数据集验证 + SOTA 对比**才能撑成论文。

---

## 1. 项目目标演变

| 阶段 | 目标 | 状态 |
|---|---|---|
| L1 (废弃) | VLM-as-init pseudo-fusion | 已弃 |
| L3 (当前) | Direction C: Dual-Path Inference with VLM as Verifier | v1 完成 |

**核心立项假设**：YOLO 视觉检测器 + CLIP 文本验证器在推理时融合，在少样本/跨域场景下能比单路检测器更准。

**当前发现**：假设**部分成立**——增益强度依赖 shot 数与 domain 距离 CLIP 训练分布。

---

## 2. 技术栈

| 组件 | 版本/路径 | 用途 |
|---|---|---|
| YOLO11s | `third_party/ultralytics`，权重 `runs/.../best.pt` | 主检测器 |
| `FSODDetect` head | `fsod/fsod/modules/cosine_head.py` | 替换原 detect head, cosine classifier |
| Florence-2-base | 服务器 `/root/epfs/07_FSOD_LLM/models/Florence-2-base/` | **训练时** 文本原型源 (Phase 1 DAF) |
| CLIP ViT-B/32 | 本地 `models/clip-vit-base-patch32/` (580MB), 服务器同路径 | **推理时** 验证器 (Phase 2) |
| CD-FSOD-Bench | 服务器 `/root/epfs/07_FSOD_LLM/datasets/cdfsod_bench/` | 6 数据集×3 shot |
| Python | 本地 3.14.3 (`C:/Python314/python.exe`)，服务器 farline conda env | |

**离线约束**：服务器无外网，所有 HF 模型必须本地预下载，所有 `from_pretrained` 默认 `local_files_only=True`。

---

## 3. 已完成代码资产

### 3.1 Phase 1 (DAF) 已固化
- `fsod/scripts/train_cdfsod.py` — DAF 训练入口
- `fsod/fsod/modules/cosine_head.py`, `prototype.py`, `adaptation.py`
- `fsod/fsod/cdfsod/{datasets,domain_gap,calibration}.py`

### 3.2 Phase 2 (方向 C) 新增 (本次交付)
- `fsod/prompts/dior.json` — DIOR 20 类的 readable labels + 9 模板 prompt ensemble
- `fsod/scripts/extract_vlm_text_proto.py` — 离线生成 CLIP 文本原型 `[C, D]`
- `fsod/scripts/eval_fsod_dualpath.py` — 双路 evaluator
  - 关键 flags: `--fusion-mode {none,fixed,multiplicative,rerank}`, `--gamma`, `--vlm-temperature`, `--vlm-only-below`, `--split {val,test}`, `--clip-model`, `--first-conf`, `--final-conf`
- `fsod/scripts/download_clip_vitb32_minimal.py` — 离线机用，只拉 `pytorch_model.bin` + tokenizer (~580MB，省去 1.2GB 冗余)
- `fsod/fsod/modules/vlm_verifier.py` — `CLIPVerifier` 类，crop+pad+CLIP encode
- `fsod/fsod/modules/dual_path_fusion.py` — `fuse_box_scores()` 核心函数
- `fsod/fsod/modules/tests/test_dual_path_fusion.py` — 7 个单测
- `fsod/configs/cdfsod_DIOR_{1,5,10}shot.yaml` — 三档 shot 配置

### 3.3 关键设计决定
- **温度 T**：CLIP `logit_scale.exp()=100` 是训练用，**不是推理融合用**。实测 T=5 最佳。
- **Pad ratio**：1.2，给 CLIP crop 留 context。
- **离线模型加载**：所有 3 个 HF 加载点都加了 `local_files_only=True` 默认 + `cache_dir` 可选。

---

## 4. 实验结果（DIOR-10shot val split，daf_a0.12 权重）

### 4.1 Hyper-param sweep (10-shot only, 已收敛)

| Run | Fusion | T | gamma | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|---|
| **A baseline** | none | - | - | **0.8424** | 0.7447 | 0.728 | 0.620 |
| F_mult_T1 | mult | 1 | 0.5 | 0.7917 | 0.7331 | 0.662 | 0.723 |
| **F_mult_T5** ⭐ | mult | 5 | 0.5 | 0.8226 | **0.7500** | 0.637 | **0.812** |
| F_mult_T10 | mult | 10 | 0.5 | 0.8122 | 0.7363 | 0.589 | 0.826 |
| F_mult_T30 | mult | 30 | 0.5 | 0.7845 | 0.7022 | 0.455 | 0.839 |
| H_fixed_T10 | fixed | 10 | 0.3 | 0.8159 | 0.7266 | 0.604 | 0.743 |

**结论**：T=5 + multiplicative 是最佳配方。门控 (`vlm-only-below`) 在 val split 失效（YOLO conf 普遍高）。提高 final-conf 会砍掉低分检测，损失大。

### 4.2 跨 shot 对照 (DIOR, daf_a0.12 + F_mult_T5)

| Shot | Baseline mAP50 | + Dual-Path | Δ% | Baseline mAP50-95 | + Dual-Path | Δ% |
|---|---|---|---|---|---|---|
| 1 | 0.0625 | **0.0925** | **+48.0%** | 0.0483 | **0.0651** | **+34.8%** |
| 5 | 0.1983 | 0.1923 | −3.0% | 0.1529 | **0.1632** | **+6.7%** |
| 10 | 0.8424 | 0.8226 | −2.4% | 0.7447 | **0.7500** | **+0.7%** |

**核心发现**：
1. **mAP50-95 在所有 shot 设置下全部正向**，且增益单调衰减（+34.8%→+6.7%→+0.7%）。
2. 1-shot 下 +48% mAP50 是论文级数字。
3. 10-shot baseline 已接近视觉极限，CLIP 边际负收益（mAP50）但定位质量提升（mAP50-95）。

### 4.3 重要 caveat（前任 AI 踩过的坑）
- 1/5-shot 第一次跑出来异常（5-shot Δ=−29.9%），**根因是 `prepare_cdfsod.py` 在 val.json 缺失时 fallback 用 train 当 val**。修复：从 10-shot 拷贝 val.txt/test.txt manifest。
- DIOR test split 比 val split 难得多。Phase 1 报告 0.781 用 ultralytics 默认 NMS，本评估器 conf=0.001 更宽松，所以同模型 val 上能到 0.8424。**口径只在同 evaluator 内自洽**。

---

## 5. 论文定位

### 5.1 Title 候选
*Dual-Path Vision-Language Verification for Cross-Domain Few-Shot Detection: When and Why Does CLIP Help?*

### 5.2 核心贡献（按已有素材）
1. **DAF (主贡献，Phase 1)** — Cosine head + 域差异自适应的视觉-文本原型融合
2. **Dual-Path Inference (次贡献，Phase 2)** — 训练-推理解耦的 VLM 验证框架
3. **系统性消融** — 温度/门控/fusion mode/shot/dataset 的影响分析

### 5.3 核心 narrative
> "VLM verification 在少样本极端场景提供显著语义先验补偿 (1-shot +48% mAP50)，增益随 shot 数增加单调衰减；在 high-baseline 时主要体现于定位质量 (mAP50-95) 与召回率提升而非分类正确性。这表明 dual-path inference 最适合作为 **shot-adaptive 的辅助模块**，而非通用增强器。"

---

## 6. 待办（按优先级）

### 6.1 P0 必做（决定能不能投）
1. **跨数据集验证** (1-2 天，机器跑) — 6 数据集 × 3 shot = 18 组训练 + 18 组 dualpath eval
   - 数据集：ArTaxOr / Clipart1k / DIOR (已完成) / DeepFish / NEU-DET / UODD
   - 预期发现："CLIP 增益 = f(shot, domain-CLIP-distance)"，clipart 应该正向最大
   - **未启动**。configs 还没生成（只有 DIOR 三个）。需要按 `cdfsod_DIOR_*shot.yaml` 模板批量生成 5×3=15 份 configs。
2. **SOTA 对比** (1 天) — CD-ViTO / DE-ViT / Detection-PT 在同 split 上跑

### 6.2 P1 (论文质量)
3. **2×2 Ablation table** — 当前 baseline 是 "DAF only"，缺：
   - 裸 YOLO finetune（无 DAF 无 dual-path）
   - dual-path only（无 DAF）
   - DAF + dual-path（已有）
4. 替换 verifier 实验：RemoteCLIP / GeoRSCLIP，验证 "verifier 选对方向 C 才有效"

### 6.3 P2 (可选)
5. **修 fixed mode 的温度问题**：当前 fixed 实测全部下跌，是因 `softmax(s_vlm * T)` 在 T 过高时退化。新版尝试 T<10 + fixed γ<0.3 组合（H 实验已部分覆盖）。
6. 选择性融合优化：当前 `--vlm-only-below` 在 val 失效（YOLO 普遍 high-conf），可以改成基于 IoU 重叠或基于 class confidence margin 的门控。

---

## 7. 服务器执行指引

### 7.1 关键路径
```
/root/epfs/07_FSOD_LLM/fsod                          # 仓库
/root/epfs/07_FSOD_LLM/fsod/models/clip-vit-base-patch32   # CLIP (580MB)
/root/epfs/07_FSOD_LLM/models/Florence-2-base/             # Florence-2 (训练用)
/root/epfs/07_FSOD_LLM/datasets/cdfsod_bench/              # 6 数据集
runs/cdfsod_DIOR_10shot/cdfsod_DIOR_10shot_daf_a0.12/weights/best.pt  # 主权重
runs/vlm_assets/dior_clip_vitb32_text_proto.pt             # CLIP 文本原型
runs/direction_c/dualpath_results.csv                      # 实验结果汇总
```

### 7.2 快速复现命令
```bash
cd /root/epfs/07_FSOD_LLM/fsod
CLIP=models/clip-vit-base-patch32
YOLO=runs/cdfsod_DIOR_10shot/cdfsod_DIOR_10shot_daf_a0.12/weights/best.pt
PROTO=runs/vlm_assets/dior_clip_vitb32_text_proto.pt

# 抽文本原型 (一次性)
python scripts/extract_vlm_text_proto.py \
    --dataset DIOR --prompts prompts/dior.json \
    --model $CLIP --out $PROTO

# Baseline
python scripts/eval_fsod_dualpath.py \
    --config configs/cdfsod_DIOR_10shot.yaml \
    --weights $YOLO --text-proto $PROTO --clip-model $CLIP \
    --fusion-mode none --run-tag A_baseline_10shot

# 最佳配置
python scripts/eval_fsod_dualpath.py \
    --config configs/cdfsod_DIOR_10shot.yaml \
    --weights $YOLO --text-proto $PROTO --clip-model $CLIP \
    --fusion-mode multiplicative --gamma 0.5 \
    --vlm-temperature 5 --run-tag F_mult_T5_10shot

column -ts, runs/direction_c/dualpath_results.csv
```

### 7.3 跨数据集脚本模板（待执行）
```bash
for DS in ArTaxOr Clipart1k DeepFish NEU-DET UODD; do
  for SHOT in 1 5 10; do
    CFG=configs/cdfsod_${DS}_${SHOT}shot.yaml   # 需要先生成
    # 1. prepare
    python scripts/prepare_cdfsod.py --config $CFG
    # 2. fix val if missing (从某个已有 split 拷过来)
    # 3. train
    python scripts/train_cdfsod.py --config $CFG
    # 4. extract text proto (per dataset)
    python scripts/extract_vlm_text_proto.py \
        --dataset $DS --prompts prompts/${DS,,}.json \
        --model $CLIP --out runs/vlm_assets/${DS,,}_text_proto.pt
    # 5. eval baseline + best config
    YOLO=$(ls runs/cdfsod_${DS}_${SHOT}shot/cdfsod_${DS}_${SHOT}shot_daf_*/weights/best.pt | head -1)
    PROTO=runs/vlm_assets/${DS,,}_text_proto.pt
    for CFG_TAG in "none A_${DS}_${SHOT}shot" "multiplicative F_mult_T5_${DS}_${SHOT}shot"; do
      set -- $CFG_TAG
      python scripts/eval_fsod_dualpath.py \
          --config $CFG --weights $YOLO --text-proto $PROTO --clip-model $CLIP \
          --fusion-mode $1 --gamma 0.5 --vlm-temperature 5 --run-tag $2
    done
  done
done
```

**注意**：每个数据集要单独写 prompts json（参考 `prompts/dior.json`），且 `extract_vlm_text_proto.py` 内部 `get_dataset_spec` 只支持 `fsod/cdfsod/datasets.py` 已注册的数据集，可能需要扩展。

---

## 8. 给接手 AI 的关键提示

1. **不要再在 DIOR-10shot 上调参**——已经收敛，再调是噪声。
2. **mAP50 不是唯一指标**——本项目核心 finding 在 mAP50-95 上更明显。报告时两个都要给。
3. **温度 T 默认 100 是错的**——CLIP `logit_scale` 是对比损失常数，不能直接当 softmax 温度。本项目实测 T=5 最佳。
4. **prepare_cdfsod.py 的 val.json fallback 是地雷**——任何新 shot 配置务必检查 `data/.../manifests/val.txt` 是否真的指向 val split。
5. **离线服务器约束**——所有新 HF 模型加载必须支持 `local_files_only=True` + 本地目录路径。
6. **用户 claude.md 全局规则**：
   - 中文回答
   - 删文件前必须问"你确定要删除 [文件名] 吗？"

---

## 9. 文件位置快查

| 类别 | 路径 |
|---|---|
| 配置文件 | `fsod/configs/cdfsod_*.yaml` |
| 训练入口 | `fsod/scripts/train_cdfsod.py` |
| 评估入口 (新) | `fsod/scripts/eval_fsod_dualpath.py` |
| 评估入口 (旧) | `fsod/scripts/eval_fsod.py` |
| 数据准备 | `fsod/scripts/prepare_cdfsod.py` |
| CLIP 验证器 | `fsod/fsod/modules/vlm_verifier.py` |
| 融合函数 | `fsod/fsod/modules/dual_path_fusion.py` |
| 文本原型抽取 | `fsod/scripts/extract_vlm_text_proto.py` |
| Prompt 模板 | `fsod/prompts/{dataset}.json` |
| 数据集定义 | `fsod/fsod/cdfsod/datasets.py` |
| 实验汇总 | `runs/direction_c/dualpath_results.csv` |
| 文本原型 | `runs/vlm_assets/{dataset}_text_proto.pt` |

---

## 10. 历史决策记录（避免重复踩坑）

| 决策 | 原因 |
|---|---|
| 用 CLIP 而非 Florence-2 做验证器 | Florence-2 是 DaViT+BART, 非对比预训练，无 shared image-text space |
| 默认 `--split val` 而非 `test` | 与 Phase 1 baseline 对齐，test split 难度高得多 |
| pad_ratio=1.2 | 给 CLIP crop 留 context, 避免目标贴边 |
| T=5 而非 100 | 100 让 softmax 退化成 one-hot，错一次毁一个框 |
| `multiplicative` 而非 `fixed` | fixed 在 high-T 下完全崩；mult 更鲁棒 |
| 1/5/10 shot 用 10-shot 的 val.txt | DIOR 类相同，可复用；避免 val=train 的 fallback 陷阱 |
