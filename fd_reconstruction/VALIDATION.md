# FD 重建三组方法验证记录

验证日期：2026 年 10 月 4 日。环境为 macOS ARM64、Python 3.13.5、CPU；没有 CUDA。所有小模型结果只用于工程验证，不是 ImageNet 实验结果。

## 双侧当前统计扩展

新增 `ours_current_both` 后，本地完整测试为 **87 passed**，包括原 73 项与新增 14 项。新增覆盖：无任何 EMA 构造/状态；真实与重建缓存按 ψ/E+D 双版本刷新；3/7 张 microbatch 的不整除图片池，D/G 梯度分别与一次性整池计算图核对；完整 D→G 优化器更新对照；每次 1/2 个 D 更新的顺序、梯度隔离与统计版本；重放失配时禁止 optimizer step；训练 2 步→恢复到 4 步与连续 4 步逐项完全一致；独立评价；配置与 50k 防线；实际小型随机 AutoencoderKL 和 timm ViT 的 checkpointed 反向；eval 模式下 BatchNorm 缓冲不变、affine 仍有梯度。

新版每次 G 更新覆盖整池，不再是当前小 batch 加历史 fake EMA。局部公式数值、梯度和恢复检查均只在本地 CPU 完成；预训练 CUDA、50k 成本与 hacking 效果仍未验证。旧配置/作业仍是原单侧当前统计方案，不能把旧短测成功当作新版验证。新版 nibi 入口为 `jobs/ours_current_both_nibi.sbatch`；它复用 smoke controller 的显式 `current_both` 模式，检查双侧版本、无 EMA、恢复一致性及当前 checkpoint 的真实/重建统计。必须以该模式自己的 `result.json` 判断是否通过。

独立 CLI 产物在 `runs/verified_C_current_both_20261004/`：32 图训练池，训练 2 步后恢复到 4 步，并另跑连续 4 步。模型、目标状态和两个优化器逐位相同；另完成 32 图固定评价器评价。4 步共 2 次 D、4 次 G，真实/重建缓存分别编码 3/6 遍，192 张样本参与梯度重放；每个 loss 的双方 ψ 版本及 fake 的 E+D 版本匹配，EMA 更新计数始终为零。实现指纹为 `c6f36350fcb5ccb76ae7fb2a8e65d6c21db77ee8935fa6a2276bfe30ccd7a3a4`。Python 编译、作业 Bash 语法、依赖检查、wheel 构建和仓库外导入新模块均通过；A/B 官方核心与原 B 步骤文件未改动。

## 自动测试

在 `fd_reconstruction` 下执行 `.venv/bin/python -m pytest -q`，新增 C 后的结果为 **73 passed**，包含此前 A/B 的 59 项回归测试。A/B 两组训练、恢复和端到端测试均已接入官方静态后端；B 动态统计与 whitening 直接调用官方原码，C 复用这些动态计算核心但有独立的目标与状态管理。

A 官方后端新增检查：

- 两个运行时核心文件与固定版本官方源码逐字节一致。
- EMA、完整队列和增量队列的初始化、raw FD、FP32 返回、特征梯度、连续 7 步提交与队列绕回，分别在 FP32/FP64 输入下与官方代码严格相等。
- 三种统计模式各完成 6 步重建训练，与直接调用官方函数的对照循环逐项核对损失、E+D 参数和全部队列状态；包括原实现优化前入队与本项目优化后提交的等价性。
- 退化输入的值与梯度遵循当前 PyTorch 下官方代码的行为，不使用自定义有限平方根导数替换。
- 非官方初始化数量、归一化 epsilon、EMA beta=0 以及重复初始化/提交均被拒绝。

已覆盖：

- 与固定版本官方 FD-Loss 的普通情况下 FD 数值、梯度、EMA 统计和队列替换对照。
- G-step 确实更新 encoder 和 decoder，固定特征提取器保持不变。
- 重复 forward 不修改统计，重复 commit 被拒绝。
- 旧稳定化 FD 的退化协方差与非有限输入检查继续保留，服务于独立评价；A/B 训练核心使用官方行为，外层训练器检查非有限损失和梯度并停止。
- 连续训练与经命令行保存、恢复后的最终模型和统计状态逐项一致。
- 实际 AutoencoderKL 的小型随机检查点加载与反向传播。
- 实际 Inception 架构的输入梯度；timm ViT 的本地权重加载、CLS/平均 pooling 和输入梯度。
- 缺图、样本数不足、参考元信息不匹配、错误配置和未实现方法拒绝。
- 配对打乱不改变分布 FD，但改变逐图重建指标。
- 训练、队列模式、恢复、独立评价、分歧筛选的端到端检查。

B 组新增覆盖：

- A/B 共用静态文件与两份官方快照逐字节一致；同配置下静态 loss、输入梯度及提交状态严格相等。
- 动态 `adversarial.py` 逐字节一致，LoRA 类源码原文一致，梯度范数辅助文件除末尾换行外一致；EMA 连续 12 步缓冲与梯度严格匹配原文件。
- G 不裁剪时梯度不被缩放，非有限梯度被拒绝；D 仍裁剪为 1。默认 start 1000、warmup 4000、D frequency 2 的端点有显式检查。
- 执行事件验证每轮只重建一次、D→G→统计提交；跳过 D 的轮次只有 G→统计提交。这验证官方代码顺序，不冒充论文 Algorithm 1 的 G→D。
- Whitening 的数值、fake 梯度和 real detach 与官方函数对齐，包括低秩协方差。
- 动态真实初始化直接复制参考协方差，EMA 前向、梯度与更新和官方 `FeatureStatsEMA` 对照。
- 使用上游 whitening/EMA 函数手工构造一轮 D/G 更新，与新训练器的损失和参数更新逐项对照。
- D 不向 E+D 传播梯度，G 不更新 ψ；D 前向不提交 EMA，重复提交被拒绝。
- 静态项始终保留；真实/重建 EMA 的不同频率、动态启动、线性预热和零起点调度。
- Inception 参数策略的 BatchNorm fixture：affine 可训练、运行缓冲不变。
- QKV LoRA 与官方类的输出和输入梯度对照；真实 timm tiny ViT 的初始输出不变、仅适配器更新、冻结后输入梯度保留，以及接入 B 完整训练器。
- 在动态启动前或启动后中断恢复，最终 E+D、静态/动态网络、G/D 优化器与统计状态均与连续运行逐项完全一致；缺失 D 优化器的 checkpoint 被拒绝。
- B 拒绝关闭静态项、修改论文参数范围或换用候选真实参考估计器；所有 B 配置通过校验。

## 可检查的运行输出

C 新增 14 项检查：无静态构造/初始化依赖、真实参考与当前参数直接重编码一致、同版本缓存复用和过期版本拒绝、1/2 次 D 的执行顺序与梯度隔离、重复提交拒绝、直接调用官方动态 FD 的更新对照、主方法/真实 EMA/固定参考三种配置的精确 CPU 恢复和独立评价、配置与 50k 限制、真实 timm ViT 全参数/LoRA（含 checkpointing）、加回静态项的冻结保护，以及随机小型 AutoencoderKL 的 E+D 联合训练。未改动 A/B 官方数值文件或原训练步骤。

`runs/verified_C_release_20261004/` 是 C 当前版本的独立 CLI 运行：训练 2 步、恢复到 4 步，再评价 32 张合成重建。共 2 次 D 更新、4 次重建 EMA 提交、3 次真实池编码（初始化及两个新 ψ 版本）；每步真实特征版本都与 loss 的 ψ 版本一致，静态项为空。实现指纹为 `4ee68512645367288478708a591ea8e3b0304b571950512aac6eb3116362b7d3`。编译、作业 Bash 语法、依赖检查和 wheel 打包均通过，仓库外可从 wheel 导入 C，不依赖第三方源码目录。这些产物不提交 Git，早期 C 开发检查点不用于当前版本精确恢复。

当前版本的 `runs/verified_alignment_A_20261004/` 和 `runs/verified_alignment_B_20261004/` 均已通过独立 CLI：训练 3 步，恢复到 6 步，导出并评价 32 张合成重建。两组静态统计提交都是 6 次；B 有 2 次 D 更新及真实/重建各 4 次动态 EMA 提交。实现指纹为 `885faf0cf8858f8bf376940a15d1c8b95505cd1b13c67ad554cddc606e58e287`。这两个 smoke 的 EMA/lr 配置不同，仅分别验证链路，不用于 A/B 效果比较；静态算法等价性由相同配置的独立测试确认。

`runs/verified_fd_only_official_20261004/` 是 A 官方后端替换后的新 CLI 运行：训练 3 步、从 checkpoint 恢复到 6 步，然后独立评价第 6 步的 32 张合成图片。6 步均有 encoder/decoder 梯度、静态统计提交计数依次为 1 至 6；独立导图清单与评价都完成。实现指纹为 `4c7f67b982fc0ee2ce46570929ae15f65ab0b2674962b8606daabbb4c703a8b8`。这些小模型结果仅证明接口可执行，不证明 ImageNet 50k 稳定性或 FD hacking。

`runs/verified_fd_only/` 保留了 A 阶段代码版本的工程运行：先训练 3 步，恢复到 6 步，再分别评价第 0 步和第 6 步。B 开发改变了实现指纹，这批旧 checkpoint 不声称能在新代码下精确续训。

- `train.jsonl`：6 个训练步，EMA 提交计数为 1 至 6，记录两组参数的梯度范数。
- `checkpoints/`：第 0、3、6 步的完整状态。
- `evaluation_step0/` 与 `evaluation_step6/`：各 32 张合成输入的重建、完整性清单、两个小型固定评价器 FD、PSNR、SSIM 和逐图残差指标。
- `diagnostic.json`：两个固定评价器的变化比较，明确标记 `engineering_only: true`。

`runs/verified_advfd_B/` 保留了此前 B 版本的独立 CLI 运行：训练 3 步、恢复到 6 步、评价第 0/6 步、运行原有分歧筛选。使用合成图片和小型随机网络，各检查点评价 32 张图。此次 A/B 后端对齐改变了统计键及全包实现指纹；旧 checkpoint 不能在当前版本下精确续训。

- `train.jsonl` 有 6 个 G 步，前两步只有静态训练，累计 2 次 D 更新、4 次动态真实/重建 EMA 提交、6 次静态 EMA 提交。
- `advfd_parameters.json` 保存本次小模型动态参数集合；checkpoint 包含两个优化器。
- 两个评价目录均完成 32 张导图和固定独立评价，`diagnostic.json` 标记 `engineering_only: true`。指标变化只用于检查链路，不是 hacking 研究结果。

这些文件属于忽略的运行产物，不随源代码提交。较早的 `runs/smoke/` 是开发中间版本，不作为当前代码的可恢复检查点；恢复操作会检查实现指纹。

本次另通过 Python 编译、`pip check` 和 wheel 构建；在仓库外直接从 wheel 导入 A/B 官方核心及 B 适配器成功，不依赖 `third_party` 运行时目录。三个单卡作业脚本的 Bash 语法检查此前通过，本次未改这些脚本。完整本地依赖版本记录在 `validation/environment-macos.txt`，不应直接当作 nibi CUDA 安装方案。

## 尚未验证

尚未运行真实预训练 tokenizer 的完整 ImageNet 后训练、50k 正式评价、预训练大表征或 LPIPS 权重的完整集成流程，也未验证 CUDA 峰值显存、吞吐与集群环境。单元测试不证明大模型长训练稳定，更不证明已经发现或避免 FD hacking。

此前 A 的 nibi smoke 作业编号为 23218660；2026 年 10 月 4 日检查仍为 PENDING，工作目录是旧 checkout。没有覆盖该目录、取消或重提交作业。新 GitHub checkout `Repr_Research_Project_fd_reconstruction` 在 C 同步前检查为干净且无依赖它的作业。C 已完成本地工程实现，GPU 短测入口为 `jobs/ours_smoke_nibi.sbatch`；提交后必须以实际日志和 `result.json` 为验证依据。MAE/SigLIP 大型预训练 LoRA/full 尚未跑通 GPU 集成。

本次集群检查显示 H100、A100、MIG 和其他 GPU 节点处于 down、drained 或 inval，属于外部调度限制。短测使用真实预训练 SD-VAE/Inception、128 张训练参考和各 32 张评价图，只验证实现、参考重算与恢复，不作为 50k FD 稳定性或视觉 hacking 证据。排队或成功 pull 不能写成 CUDA 测试通过。

A/B 静态核心差异已修正；执行顺序已由用户确认跟随官方代码 D→G，现有时序回归测试与该决定一致。原训练预算、精度、预处理及参考/评价数据范围仍未全部对齐，见 `ADVFD_BASELINE.md`。测试通过不能替代其余研究协议决策，也不能作为完整论文复现的声明。
