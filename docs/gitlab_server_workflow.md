# GitLab 作为本地与服务器桥梁

这份说明对应你的使用场景：

1. 本地电脑写代码。
2. GitLab 做中转仓库。
3. 服务器从 GitLab 拉代码并运行训练。

## 推荐流程

统一采用这一条链路：

```text
本地开发机 -> git push -> GitLab -> 服务器 git fetch/reset
```

不要直接用手工拷贝代码到服务器，这样版本容易乱。

## 第一步：本地提交代码

当前项目已经初始化为 git 仓库。

你在本地执行：

```bash
git add .
git commit -m "init: fsod yolo baseline scaffold"
```

如果还没配置 git 用户信息：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 第二步：在 GitLab 创建空仓库

在 GitLab 上新建一个空项目，例如：

```text
fsod-llm-baseline
```

创建时不要勾选 README、.gitignore、License 初始化，这样最省事。

创建完成后你会拿到仓库地址，通常是下面两种之一：

SSH:

```text
git@gitlab.com:yourname/fsod-llm-baseline.git
```

HTTPS:

```text
https://gitlab.com/yourname/fsod-llm-baseline.git
```

优先推荐 SSH。

## 第三步：本地绑定 GitLab 并推送

有了 GitLab 仓库地址以后，本地执行：

```bash
git remote add origin <你的 GitLab 仓库地址>
git push -u origin main
```

如果已经加过 `origin`，改地址用：

```bash
git remote set-url origin <你的 GitLab 仓库地址>
```

查看远程是否配置成功：

```bash
git remote -v
```

## 第四步：服务器拉取代码

登录服务器后，建议先准备项目目录：

```bash
mkdir -p ~/projects
cd ~/projects
```

然后 clone：

```bash
git clone -b FSOD_LLM <你的 GitLab 仓库地址>
cd fsod
```

后续同步更新：

```bash
git fetch origin
git checkout FSOD_LLM
git reset --hard origin/FSOD_LLM
```

如果你的服务器目录固定是 `~/epfs/07_FSOD_LLM/fsod`，也可以直接在那个目录里做同步。

## 第五步：服务器环境准备

训练环境建议分开做，不要直接污染系统 Python。

如果服务器用 conda：

```bash
conda create -n fsod_llm python=3.11 -y
conda activate fsod_llm
pip install -r requirements.txt
```

如果服务器有 CUDA，需要按服务器实际 CUDA 版本安装 PyTorch。
`requirements.txt` 当前没锁死 torch，就是为了让你在服务器上按显卡环境安装。

通常建议先安装 PyTorch，再安装其余依赖，例如：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

上面的 `cu121` 只是例子，必须和服务器 CUDA 版本匹配。

## 第六步：服务器运行项目

准备数据后可按下面流程运行：

```bash
cd ~/epfs/07_FSOD_LLM/fsod
python scripts/prepare_voc_fewshot.py --config configs/baseline_voc_10shot.yaml
python scripts/train_baseline.py --config configs/baseline_voc_10shot.yaml --stage all
python scripts/eval_baseline.py --config configs/baseline_voc_10shot.yaml
```

## SSH 推荐配置

为了让本地和服务器都能稳定访问 GitLab，建议两边都使用 SSH key。

本地或服务器生成 key：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

查看公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

然后把公钥添加到 GitLab 的 SSH Keys 页面。

测试连接：

```bash
ssh -T git@gitlab.com
```

## 建议的协作规则

为了避免服务器上的实验代码和本地代码互相覆盖，建议固定规则：

1. 本地负责改代码、提交代码。
2. 服务器尽量不直接改业务代码。
3. 服务器只拉取、运行、记录日志和模型。
4. 训练产物不要提交到仓库，权重和日志继续放在 `.gitignore` 覆盖的目录里。

## 当前项目的推荐 push 内容

应该提交：

1. `configs/`
2. `scripts/`
3. `fsod/`
4. `README.md`
5. `requirements.txt`
6. `.gitignore`

不应该提交：

1. 数据集原图和标注
2. 训练权重
3. 运行日志
4. 本地虚拟环境目录

## 最后一步

你把 GitLab 仓库地址给我之后，我可以直接继续帮你做下面两件事之一：

1. 在本地把 `origin` 配好。
2. 如果你愿意，也可以继续帮你整理首次提交命令和服务器 clone 命令。