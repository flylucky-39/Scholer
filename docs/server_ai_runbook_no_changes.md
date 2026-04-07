# 服务器 AI 执行手册（严格只读版）

这份文档是给服务器上的 AI 助手看的。

目标不是让它自由发挥，而是让它在 **不修改任何内容** 的前提下，严格按当前项目状态推进实验，并在出现错误时只做反馈，不做修复。

## 1. 总原则

服务器 AI 必须遵守以下硬约束：

1. 不允许修改任何代码文件。
2. 不允许修改任何配置文件。
3. 不允许修改数据集内容。
4. 不允许自动安装新依赖。
5. 不允许自动下载新模型。
6. 不允许自动调整训练参数。
7. 不允许自动修复错误。
8. 不允许执行任何破坏性 git 操作。
9. 不允许创建新的实验逻辑。
10. 只能基于当前仓库已有内容运行、检查、汇报。

如果出现任何错误，服务器 AI 的唯一正确行为是：

1. 停止继续执行后续步骤。
2. 记录失败命令。
3. 记录完整错误信息。
4. 汇报可能的原因。
5. 等待人工决定。

## 2. 当前项目状态

当前项目是一个 FSOD baseline 工程，核心目标是先完成 YOLO few-shot baseline，再考虑接入 Florence-2。

当前已知事实：

1. 项目目录：`/root/epfs/07_FSOD_LLM/fsod`
2. 原始 VOC 数据目录：`~/epfs/07_FSOD_LLM/datasets/VOCdevkit`
3. 生成后的 few-shot 数据目录：`./data/voc_fsod_split1_10shot`
4. 当前训练配置文件：`configs/baseline_voc_10shot.yaml`
5. 当前训练脚本：`scripts/train_baseline.py`
6. 当前评估脚本：`scripts/eval_baseline.py`
7. 当前模型默认是：`yolo11s.pt`
8. 当前流程是：`base train -> novel finetune -> eval`

## 3. 当前配置摘要

服务器 AI 不允许修改这些值，只允许读取确认：

1. `voc_root: ~/epfs/07_FSOD_LLM/datasets/VOCdevkit`
2. `output_root: ./data/voc_fsod_split1_10shot`
3. `model: yolo11s.pt`
4. `device: 0`
5. `workers: 16`
6. `epochs.base: 100`
7. `epochs.finetune: 50`
8. `batch_size.base: 64`
9. `batch_size.finetune: 16`
10. `freeze.backbone: 10`

## 4. 服务器 AI 的当前阶段任务

当前阶段不是继续改代码，而是完成 baseline 的后半程。

### 阶段 A：等待或确认 base train 结束

服务器 AI 需要检查：

1. `runs/fsod_baseline/base_pretrain/weights/best.pt` 是否存在
2. `runs/fsod_baseline/base_pretrain/weights/last.pt` 是否存在
3. base train 是否已经正常结束
4. base train 的输出目录是否完整

如果 base 训练尚未结束：

1. 继续观察训练状态
2. 汇报当前 epoch / loss / GPU 使用情况
3. 不要启动 finetune

如果 base 训练已经结束：

1. 进入阶段 B

### 阶段 B：运行 novel finetune

必须使用下面这条命令，不允许改脚本名，不允许改权重路径：

```bash
cd /root/epfs/07_FSOD_LLM/fsod
python scripts/train_baseline.py --config configs/baseline_voc_10shot.yaml --stage finetune --weights runs/fsod_baseline/base_pretrain/weights/best.pt
```

服务器 AI 在 finetune 启动后必须汇报：

1. 实际使用的数据 yaml 路径
2. 实际使用的权重路径
3. batch size
4. image size
5. device
6. 输出目录

### 阶段 C：运行评估

如果 finetune 成功结束，必须运行下面这条命令：

```bash
cd /root/epfs/07_FSOD_LLM/fsod
python scripts/eval_baseline.py --config configs/baseline_voc_10shot.yaml
```

服务器 AI 在评估后必须汇报：

1. `mAP@0.5`
2. `mAP@0.5:0.95` 对应脚本里的 `map`
3. `mAP@0.75`
4. 使用的 checkpoint 路径
5. 评估输出文件位置

## 5. 当前阶段允许做的事情

服务器 AI 只允许做这些：

1. 读取配置
2. 检查文件是否存在
3. 启动当前仓库已有脚本
4. 观察训练日志
5. 汇总评估结果
6. 报告错误

## 6. 当前阶段禁止做的事情

服务器 AI 一律禁止：

1. 修改 `configs/baseline_voc_10shot.yaml`
2. 修改 `scripts/train_baseline.py`
3. 修改 `scripts/eval_baseline.py`
4. 修改 `fsod/voc.py`
5. 重新划分数据集
6. 替换模型
7. 自动改 batch size
8. 自动改学习率
9. 自动安装缺失包
10. 自动下载 Florence-2、Qwen、GroundingDINO 等模型
11. 自动接入 ComfyUI
12. 自动生成新代码

## 7. 发现错误时的固定处理方式

如果执行过程中报错，服务器 AI 必须输出固定格式：

### 错误反馈模板

1. 失败步骤：
   - 例如：`novel finetune`
2. 失败命令：
   - 完整命令原文
3. 错误原文：
   - 原始 traceback 或终端报错
4. 可能原因：
   - 只允许分析，不允许修改
5. 当前建议：
   - 等待人工处理

服务器 AI 不允许：

1. 自行改代码再试
2. 自行改配置再试
3. 自行跳过错误继续执行

## 8. 未来阶段该做什么

未来阶段的工作顺序必须是下面这样，但 **未经人工明确批准，服务器 AI 不允许提前实施**。

### 未来阶段 1：baseline 结果确认

在 base 和 finetune 都完成后，先确认：

1. novel 类检测是否有提升
2. overall mAP 是否可接受
3. 当前 few-shot baseline 是否可以作为对照组

### 未来阶段 2：采样策略审查

如果要做论文级严格对比，再去检查当前 few-shot 采样是否要从“至少 K 个实例”改为“严格 K-shot”。

注意：

1. 服务器 AI 只允许提出建议
2. 不允许在未批准情况下修改采样逻辑

### 未来阶段 3：Florence-2 集成

如果 baseline 稳定，再进入 Florence-2 集成阶段。

未来集成目标是：

1. 用 Florence-2 输出 ROI / grounding 结果
2. 导出 bbox / mask / JSON
3. 再考虑如何接入 YOLO few-shot 流程

但当前阶段：

1. 不允许服务器 AI 主动实现 Florence-2 集成
2. 不允许服务器 AI 主动下载大模型并改代码

### 未来阶段 4：Prototype / Cosine Classifier

只有在 baseline 和 Florence-2 方案都明确后，才进入：

1. Prototype 模块
2. Cosine classifier
3. Feature adaption layer

当前阶段严禁跳到这里。

## 9. 建议服务器 AI 的最终汇报格式

服务器 AI 每次完成一个阶段后，统一按这个格式汇报：

### 阶段汇报

1. 当前阶段：
   - `base train` / `novel finetune` / `eval`
2. 是否成功：
   - `成功` / `失败`
3. 执行命令：
   - 完整命令
4. 核心输出：
   - checkpoint 路径 / mAP / 日志目录
5. 问题：
   - 若无则写 `无`
6. 下一步建议：
   - 只建议，不自动实施

## 10. 给服务器 AI 的一句话总要求

你不是来修改项目的，你是来 **严格按现有仓库内容执行、观察、汇报** 的。

如果一切正常，就推进到下一步。

如果出现错误，就停下并把错误完整反馈给人工。