# B 组 AdvFD 图像重建基线

B 组将 AdvFD 的生成输出替换为重建输出 `x_hat = decoder(encoder(x))`，G-step 联合训练 E+D。保留静态 FD、原方法的动态参数范围、真实与重建侧 EMA，以及 real whitening；不加入候选方法 C 的三点改动，也不添加逐图配对训练损失。

这是损失机制的重建迁移，不是原生成论文的完整硬件与训练预算复现。论文依据为 [AdvFD 第 5.3 节与附录 E](https://arxiv.org/html/2608.11205v1)；代码固定在 [4e4cfed](https://github.com/GasaiYU/AdvFD/tree/4e4cfed944e4fc38a75fae3ea7701ae9e5587060)。

## 保留的损失与状态

G-step 最小化 `static normalized FD + lambda(t) * dynamic normalized FD`。D-step 最大化未归一化的动态 whitened FD；不能把 G-step 的 stop-gradient 归一化或 lambda 权重误加到 D-step。归一化分母为 `FD.detach() + 0.01`，没有新增 metric。

| 项目 | B 的实现 |
| --- | --- |
| 静态分支 | 特征网络与真实参考固定；与 A 共用逐字节匹配两仓库的官方 queue/losses 核心 |
| 动态真实统计 | 初值直接复制参考均值与协方差，不进行额外 ddof 修正；随后按原机制 EMA 更新 |
| 动态重建统计 | 到动态分支启动时，复制匹配静态分支当时的重建均值与二阶原始矩，不复制过时初值 |
| Whitening | 保留原代码对真实和重建协方差的对角加载、真实变换 detach、FP64 分解和 FP32 返回 |
| 可训练参数 | Inception 全参数，包括 BatchNorm affine；SigLIP/MAE 仅 rank-16 LoRA 的 QKV 参数 |
| 网络运行模式 | 动态表征始终 eval，固定 BatchNorm 运行缓冲；这不等于冻结 affine 参数 |
| 统计提交 | 前向只计算候选值；按 G 侧的新 ψ 特征提交一次，不把 D 前向重复计入 EMA |

`vendor/advfd/adversarial.py` 原样保存官方完整文件；动态统计继承其中的 `FeatureStatsEMA`，whitening 直接调用其函数。`vendor/advfd/lora.py` 保留官方 `LoRAQKVLinear` 类原文，仅调整导入与打包。`objectives/adaptive_fd.py`、`objectives/whitening.py`、`representations/lora.py` 和 `engine/adversarial.py` 负责重建接口、冻结范围及状态生命周期；运行时不依赖 `third_party` 同名包。

## 配置与原始来源

| 配置 | 静态分支 | 动态分支 | 用途 |
| --- | --- | --- | --- |
| `advfd_reconstruction.yaml` | SIM | Inception 全参数 | 主 B 配置，动态设置对齐官方 JiT SIM recipe |
| `advfd_reconstruction_inception.yaml` | Inception | Inception 全参数 | 与 `fd_only_inception.yaml` 匹配；不是论文主 SIM 配置 |
| `advfd_reconstruction_mae.yaml` | SIM | MAE rank-16 LoRA | 论文所述参数范围的 backbone 对照 |
| `advfd_reconstruction_siglip.yaml` | SIM | SigLIP rank-16 LoRA | 同上 |
| `advfd_smoke.yaml` | 随机小表征 | 独立小表征副本 | CPU 工程测试，不是论文模型或研究结果 |

主 B 的动态设置来自官方 `scripts/table_3_JiT_adv_fd_sim.sh`：权重 0.1，D 学习率 1e-6，每两轮更新一次、每次一步，启动位置 1000、线性预热 4000，动态 EMA beta 0.99，whitening epsilon 0.001，梯度裁剪 1。D AdamW 的 betas 为 0.9/0.999，weight decay 为 0，沿用 `main_fd.py` 默认值。静态 EMA beta 0.999，SIM 三项单位权重；输入分辨率和 pooling 沿用静态适配器。

论文明确写 rank-16 LoRA，但固定代码快照中的 MAE/SigLIP 消融脚本默认 rank-8。按本项目要求，rank 采用论文的 16；alpha 16、QKV targets、dropout 0、D 学习率 2e-5、真实统计每两轮更新等其余细节来自对应官方脚本，不声称论文已逐项给出。此差异保存在配置注释中，不能称为与发布脚本逐字相同。

## 每轮执行与恢复

当前代码沿用官方可执行实现的 D-then-G 顺序；[论文 Algorithm 1 第 6–16 行](https://arxiv.org/html/2608.11205v1) 则是 G-then-D，而且用更新后的 G 重新生成。两者不能声称同时一致。已向用户提出选择，在得到回复前保持已有顺序，不擅自改为另一种训练轨迹。

代码依据为固定快照 `main_fd.py`：第 1148 行附近先生成带梯度样本，第 1763 行执行 D optimizer，第 1822 行起冻结 ψ 并计算 G 侧动态特征，第 2878 行才执行 G optimizer；重建适配仍只前向一次 E+D。顺序回归测试同时检查有 D 更新和跳过 D 更新的轮次。

1. 用 E+D 计算一次重建并保留 G 所需计算图。D 只接收 detached 重建和真实图片，E+D 参数在 D 阶段冻结。
2. 达到启动位置且满足更新频率时，更新 ψ；不提交真实或重建 EMA。
3. 冻结 ψ，重新提取动态特征。E+D 通过固定静态网络和当前 ψ 接收梯度；没有为 D 更新重新生成另一批图。
4. G 优化成功后，提交静态与动态统计。动态真实侧可以按原脚本频率跳过更新；重建侧每个 active G 步提交一次。

调度位置采用官方的零起点 `current_step`，即已完成的 G 步数。`start_step=1000` 表示完成 1000 步静态训练后，下一轮启动 D；此时动态 G 权重为零，之后线性增长。日志的 `step` 是本轮完成后的步数，另存 `adv_schedule_step`，避免 off-by-one。即使权重尚未增长，静态训练仍继续。

完整 checkpoint 包含 E+D、静态与动态网络、全部统计、G/D 两个优化器、更新计数、RNG、数据位置及代码指纹。`advfd_parameters.json` 保存动态可训练参数名称和数量。缺失 D 优化器时拒绝恢复，不悄悄重新初始化。

## 公平比较与未复制部分

A/B 继续共享本项目的 tokenizer、数据、E+D AdamW、batch 和训练预算。当前公共默认是单 GPU、batch 16、E+D 学习率 1e-6、常数学习率和 10k 步；这些是重建实验设置，不冒充原论文的多 GPU、global batch 1024、125k 步和生成器 warmup/cosine 日程。是否恢复原训练预算与日程也待用户确认，不会把小 batch 的 EMA 宣称为等效大 batch。

G AdamW betas 0.9/0.95、weight decay 0、不裁剪梯度，与官方默认对齐。D 的裁剪阈值为 1，二者不能混用。未移植生成器的 EDM 模型权重 EMA；FD-Loss 论文表 B.2 写模型 EMA 为 none，但官方 JiT 脚本启用 EDM EMA，属于另一处论文与脚本差异。当前一直用在线 E+D 权重评价；统计 EMA 与模型权重 EMA 是不同机制。

其他仍需区分的设置：本项目以同一批原图作为重建输入和动态真实样本，属于任务迁移；固定训练真实参考选 50k，官方参考来自完整 ImageNet train；重建统计初始化总数同为 50k，但分块 batch 目前为 16 而非官方 256。当前训练 FP32、中心裁剪且无随机翻转，也不是原论文 bf16 和训练增强的完整复现。损失公式、可训练范围正确不等于这些设置已经对齐。

B 默认 SIM 时应与 A 的 `fd_only_sim.yaml` 对照；若先用 A 的单 Inception，则选 B 的匹配配置。不可把静态网络数量不同造成的差异全部归因于动态学习。

独立评价器、普通样本 FD、配对指标、PNG 量化、真实评价参考与 50k 要求保持现有 A/B 共同协议。没有将训练动态 ψ 用作跨检查点评分尺，也没有新增论文 FD-r3/FD-r6 聚合或借用不匹配的 valFD 常数。保留 FD 定义不等于声称重建评价与论文生成评价的数值可直接比较。

本迁移另外保留现有工程保护：非有限值停止、原子 checkpoint、显式统计提交、参考指纹检查。正常路径的数值与梯度做官方实现对照；异常路径不照搬上游的静默跳步行为。

A/B 静态计算核心现已统一为官方原码，并检查同配置下 loss、输入梯度及状态严格一致。动态 EMA 也不再通过自写协方差更新重构二阶矩，而是直接执行官方原位 mean/m2 更新；连续 12 步的缓冲和梯度与原文件严格一致。以上修正消除已发现的核心实现偏差，不消除前述预算、预处理、评价协议及论文/代码顺序分歧。

## 运行入口与验证范围

```bash
.venv/bin/python train.py --config configs/advfd_smoke.yaml
.venv/bin/python train.py --config configs/advfd_reconstruction.yaml
```

第一行无需预训练权重；第二行使用 README 中的 ImageNet 与 tokenizer 环境变量。通用单 GPU 作业模板可通过 `FD_CONFIG=configs/advfd_reconstruction.yaml` 选择 B，但本次不提交新作业，也不覆盖此前 nibi A smoke 的代码快照。

本地验证覆盖 whitening 值与梯度、EMA 初始化/更新、官方损失更新规则、D/G 梯度隔离、LoRA 数值与训练范围、BatchNorm 缓冲、启动/预热/频率、动态启动前后恢复，以及独立评价。真实大模型 CUDA 和 50k 流程仍待验证，不能据此宣称方法有效或已无 bug。详见 [验证记录](VALIDATION.md)。
