# FSOD LLM Baseline

这版仓库先只做一个干净、可复现实验的 baseline，不把轻量视觉大模型定位、特征适配层、Prototype、Cosine Classifier 直接揉进来。

当前 baseline 定义如下：

1. 使用 Ultralytics YOLO11s 作为 baseline 检测器。
2. 用 VOC2007 trainval + VOC2012 trainval 做 base 训练。
3. 用自定义 novel 类别的 10-shot 样本做第二阶段微调。
4. 用 VOC2007 test 做统一评估，指标先看 mAP@0.5。

后续你可以在这个 baseline 上逐步加：

1. Cosine Classifier
2. Prototype 构建与更新
3. 轻量视觉模型定位模块
4. 特征适配层 / mask 融合

## 项目结构

```text
configs/
  baseline_voc_10shot.yaml    # 实验配置
scripts/
  prepare_voc_fewshot.py      # VOC 转 YOLO + few-shot 划分
  train_baseline.py           # 两阶段训练入口
  eval_baseline.py            # 评估入口
fsod/
  voc.py                      # VOC 数据处理工具
data/
  # prepare 后自动生成
```

## 环境

建议 Python 3.10 或 3.11。

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

默认假设你的 VOC 根目录结构如下：

```text
VOCdevkit/
  VOC2007/
  VOC2012/
```

在 [configs/baseline_voc_10shot.yaml](configs/baseline_voc_10shot.yaml) 里修改：

1. `voc_root`
2. `output_root`
3. `novel_classes`
4. `shot`

当前默认配置已经按你的服务器目录做了适配：

1. 项目目录：`~/epfs/07_FSOD_LLM/fsod`
2. VOC 数据目录：`~/epfs/07_FSOD_LLM/datasets/VOCdevkit`
3. 脚本现在支持 `~` 路径展开，并且相对路径统一按仓库根目录解析。

然后运行：

```bash
python scripts/prepare_voc_fewshot.py --config configs/baseline_voc_10shot.yaml
```

该脚本会做三件事：

1. 把 VOC 标注转换为 YOLO 格式。
2. 导出 base 训练集、few-shot 微调集、test 集。
3. 生成训练所需的数据集 yaml 和统计信息。

## 训练

完整两阶段训练：

```bash
python scripts/train_baseline.py --config configs/baseline_voc_10shot.yaml --stage all
```

只跑第一阶段 base 预训练：

```bash
python scripts/train_baseline.py --config configs/baseline_voc_10shot.yaml --stage base
```

只跑第二阶段 few-shot 微调：

```bash
python scripts/train_baseline.py --config configs/baseline_voc_10shot.yaml --stage finetune --weights runs/fsod_baseline/base_pretrain/weights/best.pt
```

## 评估

```bash
python scripts/eval_baseline.py --config configs/baseline_voc_10shot.yaml
```

默认会读取第二阶段微调后的权重进行评估。

## Baseline 边界

这版 baseline 故意保持简单，目的是先拿到一个可信的对照组：

1. 第一阶段只保留 base 类标注，novel 类在 base 训练中不参与监督。
2. 第二阶段用 novel few-shot 样本微调，同时可选混入少量 base replay 图像。
3. 分类头仍然使用 YOLO 原生分类方式，不引入 prototype / cosine。

## 建议的实验顺序

1. 先固定 5 个 novel 类，跑通 10-shot baseline。
2. 确认 base / novel / all 三组 AP 都能正常输出。
3. 再替换分类头为 cosine classifier。
4. 最后再加入定位模块和适配层。

## 下一步怎么接你的研究方向

你现在要做的是“重新开始做一版 baseline”，那最稳妥的顺序是：

1. 先用这套工程拿到可复现的 baseline 指标。
2. 在微调阶段插入 prototype 分支，先不动定位模块。
3. 验证 cosine classifier + prototype 的纯增益。
4. 最后再接入轻量视觉大模型和特征掩码，做完整消融。

这样实验逻辑是干净的，论文写作也更顺。

## GitLab 到服务器工作流

如果你准备把代码上传到 GitLab，再由服务器拉取运行，建议直接按这个顺序做：

1. 本地初始化 git 仓库并提交代码。
2. 在 GitLab 创建空仓库。
3. 本地添加 `origin` 并 push 到 GitLab。
4. 服务器配置 SSH key 后从 GitLab clone。
5. 后续开发统一走 `git add -> git commit -> git push`，服务器用 `git pull` 同步。

如果服务器分支历史被覆盖过，不要用 `git pull`，改用：

```bash
git fetch origin
git checkout FSOD_LLM
git reset --hard origin/FSOD_LLM
```

更具体的命令说明见 [docs/gitlab_server_workflow.md](docs/gitlab_server_workflow.md)。