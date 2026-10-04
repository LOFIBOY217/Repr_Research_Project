# 图像重建 FD hacking 实验

独立于旧生成 pilot 的重建后训练项目，遵循 [已确定的架构设计](../fd_hacking_stage1/RECONSTRUCTION_CODE_ARCHITECTURE.md)。目标是比较 FD-only、AdvFD 和候选方法的 hacking 行为，而非追求重建排名。

当前实现 A：FD-only，以及 B：AdvFD-Reconstruction。两者都是原图 → 可训练 encoder → latent → 可训练 decoder → 重建图；B 保留静态 FD，并添加原方法的动态对抗 FD。没有像素、感知或 GAN 训练损失。候选方法 C 尚未接通；选择它会明确报错，绝不退回基线冒充新方法。

## 已实现的范围

- 单训练入口和严格配置；同时训练 E+D。
- Grounded 风格的 AutoencoderKL 后验均值重建，支持该格式的 SD-VAE、VA-VAE、REPA-E 权重。其他 tokenizer 的源码已归档，但还没有全部迁移为运行适配器。
- 官方 FD-Loss 的 Inception 特征路径，以及 timm 表征适配；提供 Inception 与 SIM 多表征配置。
- A 直接调用未修改的官方 FD-Loss `queue.py` 与 `losses.py`：固定真实参考，重建侧 EMA、完整队列或官方增量队列；保留 `FD / (FD.detach() + 0.01)`，并单独记录 raw FD。
- B 的动态分支保留 real whitening、真实与重建侧 EMA、原方法的启动及权重预热；Inception 全参数、SigLIP/MAE rank-16 LoRA，不采用候选方法的三点改动。
- 前向不修改统计，优化步骤完成后统一提交；固定表征仍可将输入梯度传给重建模型。
- 保存模型、静态表征、参考统计、重建统计、优化器、RNG、数据位置和代码指纹；B 额外保存动态表征、两侧 EMA、D 优化器和更新计数，支持完整断点恢复。
- 独立重建导图、多表征 FD、PSNR、SSIM、LPIPS 与逐图残差诊断。正式评价不接受少于 50k 图片，也不使用训练 EMA 代替整批评价统计。
- 单 GPU 作业模板、测试和离线小模型 smoke 配置。

## 安装

在本项目目录执行；本地已有独立 `.venv`，未修改全局 Python 环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,vision]'
.venv/bin/python -m pytest -q
```

nibi 的 CUDA PyTorch 需根据集群环境安装；本地 macOS 环境记录不是 CUDA 环境锁文件。不要将本地 `.venv` 复制到集群。

## 本地工程验证

以下使用随机小模型、合成图片和 32 张评价图，仅检查代码，不产生科学结论。输出目录必须为空；重复执行时通过 `--set train.output=...` 换一个新目录。

```bash
.venv/bin/python train.py --config configs/smoke.yaml
.venv/bin/python train.py --config configs/smoke.yaml \
  --resume runs/smoke/checkpoints/step_0000006.pt --set train.steps=9
.venv/bin/python evaluate.py --config configs/smoke.yaml \
  --checkpoint runs/smoke/checkpoints/step_0000006.pt \
  --output runs/smoke/evaluation_step6
```

真实架构测试会创建小型随机 AutoencoderKL 检查点，并运行 Inception 和 timm ViT 的梯度检查，不会为测试下载完整预训练模型。工程测试通过不代表已经验证真实 ImageNet 上的数值稳定性或 hacking 假设。

## ImageNet FD-only

先指定实际数据和预训练 tokenizer 的路径。路径不写死到作者或我们的账户；以下值须替换：

```bash
export IMAGENET_ROOT=/absolute/path/to/imagenet
export TOKENIZER_CHECKPOINT=/absolute/path/to/autoencoder_kl_checkpoint

.venv/bin/python prepare_reference.py --config configs/fd_only_inception.yaml --split train
.venv/bin/python train.py --config configs/fd_only_inception.yaml
.venv/bin/python evaluate.py --config configs/fd_only_inception.yaml \
  --checkpoint runs/fd_only_inception/checkpoints/step_0001000.pt \
  --output runs/fd_only_inception/evaluation_step1000
```

不传 `--checkpoint` 时评价原始 tokenizer。`configs/fd_only_queue.yaml` 切换重建特征队列；`configs/fd_only_sim.yaml` 同时使用 SigLIP、Inception、MAE。多表征会增加显存需求，第一轮用 Inception。

A 的 FD 计算以官方代码为准，不自行改造估计器或特征值梯度。固定版本 `5c03b8112fec8b9432631e4ce053c0d918cc24bc` 的 `frechet_distance/queue.py` 与 `losses.py` 原样复制到 `src/recon_fd/vendor/fd_loss/`，测试逐字节校验；`objectives/official_fd.py` 仅负责重建接口和状态生命周期。默认沿用官方 JiT recipe 的 eigvalsh 路径、EMA beta 0.999 与归一化 epsilon 0.01。完整队列使用官方 FP32 特征快照；`--set static.statistics=queue_online` 调用官方增量 sum/outer-product 算法。初始化直接调用官方累积与收尾函数，并强制 `initialization_samples == queue_size`。

不再让 A 调用此前自写的稳定化 FD 和 EMA/队列实现，包括自定义平方根导数和最终非负截断。数值异常由训练器检查并停止，不悄悄换另一套损失。统计在优化成功后提交一次，使用本轮前向的 detached 特征；正常步骤与官方在优化前入队的最终状态一致。这是 FD 核心的原样复用，不是把生成模型、数据管线和多 GPU 训练预算也搬进重建实验。B 的实现及公共独立评价此次保持不变；正式归因比较前，仍需单独核对 B 静态分支与这一官方后端的数值边界。

训练集从 `train/` 读取，评价集从 `val/` 读取；若实际名称为 `validation/`，显式覆盖 `data.val_path`。当前统一使用 Grounded/ADM 的中心裁剪，暂不引入随机增强，确保参考统计、输入和各基线匹配。训练真实参考从完整训练集固定随机抽取 50k，避免按类别目录取前 50k 导致偏样本；重建统计也先用 50k 初始化。初始化可能较慢，不发生参数更新。

FD-only 的表征与真实参考均固定，保持原基线。第三项研究设计仅针对未来候选方法 C：真实参考统计必须跟随当前特征提取器更新；EMA 是待验证的估计方式，而不是必须采用的机制。拟比较当前 batch 统计、用当前参数重编码固定真实图片池，以及 EMA。固定图片池不等于固定特征统计，不能把这项估计器消融提前加进 A 或 B 以改变基线。

## B 组 AdvFD 重建基线

配置与迁移边界见 [B 组实现说明](ADVFD_BASELINE.md)。默认 `configs/advfd_reconstruction.yaml` 使用 SIM 静态分支与 Inception 动态分支；`advfd_reconstruction_inception.yaml` 用于与单 Inception 的 A 组匹配，不冒充论文主配置。MAE/SigLIP 动态分支分别有配置文件。

```bash
.venv/bin/python train.py --config configs/advfd_smoke.yaml
.venv/bin/python train.py --config configs/advfd_reconstruction.yaml
```

第一行是 CPU 小模型工程测试，第二行需要实际 ImageNet、预训练权重和 CUDA。B 继续用同一个 `evaluate.py`，不修改 A/B 的固定评价器、FD 定义、导图协议或 50k 要求。不要直接比较静态表征数量不同的 A/B 来归因于动态分支。

## 评价与判据

评价先导出 50k 个稳定 ID 的 PNG，再顺序加载不同表征，控制单卡显存。参考缓存核对权重、表征版本、pooling、数据清单、样本数及裁剪协议。旧工程没有这些元信息的 `.npz` 不直接接受；首次由本项目重算。

`complete.json` 只有在全部样本完成后生成；缺图、重复、错误检查点或不一致数据会报错。中断的未完成导图目录不会被当成完整结果复用，请保留诊断并改用新的输出目录。逐图指标统一比较原图与导出的量化 PNG；PSNR 对完全相同输入使用 120 dB 上限，策略写入结果。

比较不同检查点的固定评价结果：

```bash
.venv/bin/python analyze.py \
  --metrics runs/fd_only_inception/evaluation_step0/metrics.json \
            runs/fd_only_inception/evaluation_step1000/metrics.json \
  --target inception --heldout dinov2 clip \
  --output runs/fd_only_inception/diagnostic.json
```

分歧报警只用于筛选；默认 1% 是描述性阈值，不是统计显著性检验。残差频率变化、PSNR 下降、FD 低于 real–real，都不单独证明视觉 hacking。配对打乱测试验证了边缘分布 FD 无法保证逐图重建正确；正式结论仍需要重复实验和小规模图像核查。

## 目录与方法边界

`src/recon_fd/` 分为 `tokenizers`、`representations`、`objectives`、`engine`、`evaluation`、`diagnostics`，与架构设计一致；`vendor/fd_loss` 隔离未修改的官方计算核心。A 的官方统计前向与 B 的 `statistics.preview()` 均不修改缓存，`commit()` 带版本检查；重复提交会失败。

三项设计的配置位置为 `static.enabled`、`adaptive.trainable_scope`、`adaptive.real_stats.mode`。A 和 B 已可执行；B 强制保留 static、论文参数范围和真实侧 EMA，避免混入 C 的消融。候选方法 C 尚未实现。

A 配置中未启用的 `adaptive.real_stats.mode: ema` 是预留值；B 中它表示明确保留 AdvFD 原机制，都不决定 C 的最终估计器。当前参数重算与 EMA 的切换仍未实现。

输出在 `runs/`，参考统计在 `cache/`，均不提交 Git。断点恢复可增加 `train.steps` 或改变日志间隔，不能更换训练数据、目标统计、学习率或实现代码。若回到较早检查点而原目录已有更晚日志，应改用新输出目录，防止混合轨迹。普通梯度累积不等于大 batch FD，所以首版明确只允许 `grad_accumulation=1`。

`jobs/` 提供单卡模板。A 的 nibi smoke 已在此前提交，记录见 `validation/nibi-smoke-23218660.json`，其中状态仅代表提交时的检查，不是实时状态。本次 B 代码没有同步到该作业目录，也没有提交 B 作业。集群执行前设置 `FD_PROJECT`、`FD_PYTHON`、数据与权重路径；通用训练模板通过 `FD_CONFIG` 选择 A 或 B。

## 来源与待验证项

三份原始源码放在 `third_party/`，不参与运行时导入。迁移后的模块统一在 `recon_fd` 命名空间，避免三仓库同名包冲突。版本与版权见 [来源记录](third_party/SOURCES.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。

旧 A checkpoint 的统计键及实现指纹与当前版本不同，不能直接续训；较早 B checkpoint 也会因全包指纹变化被严格恢复检查拒绝，不自动放宽检查。

当前已完成本地 CPU 单元测试和端到端小模型验证。真实预训练权重的完整后训练、ImageNet 50k 和 B 的 CUDA 显存尚未验证；不将这些未验证项写成“无 bug”或方法有效的结论。

具体检查项与结果见 [本地验证记录](VALIDATION.md)。

## GitHub 同步规则

代码、配置与作业脚本统一通过本地 commit/push → GitHub → nibi pull 同步；不再用 rsync/scp 覆盖服务器代码。规则保存在 [AGENTS.md](AGENTS.md)。仓库为 `LOFIBOY217/Repr_Research_Project`，重建开发分支为 `codex/fd-reconstruction-baselines`；后续服务器更新使用该分支的 `git pull --ff-only`，先确认工作区干净且无作业依赖正在修改的目录，并核对 commit。

nibi 的旧工作目录仍包含未提交文件和此前复制的同名目录，且旧 smoke 作业 23218660 在此次检查时仍处于 PENDING。因此本次不直接切换、覆盖或 pull 旧目录，也不取消作业；需先安排干净的 Git checkout 或在保留现有工作的前提下完成迁移。不要把 GitHub 已 push 写成 nibi 已 pull。

运行输出、数据、权重、环境和主机专用作业记录不推送。公开仓库不包含用户提供的 Grounded 原始源码；该本地参考不影响本项目运行或测试。FD-Loss 和 AdvFD 的公开 MIT 快照及必要许可可以随代码分发。
