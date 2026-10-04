# 图像重建 FD hacking 实验代码结构

本设计服务于已确定的三项因素：去掉静态 FD、全参数更新动态特征提取器、真实参考统计必须跟随当前特征提取器更新。EMA 是待验证的估计方式，而不是必须采用的机制。各组共同训练 encoder 和 decoder，以独立评价检验抗 hacking 行为，而不是以重建分数排名作为目标。

独立的重建实验包使用一套训练循环，通过配置切换 FD-only、AdvFD-Reconstruction 和候选方法。Grounded 提供重建适配参考，官方 FD-Loss 提供静态 FD 数值参考，AdvFD 提供动态 FD 参考。A 与 B 已完成本地工程实现，见 [项目说明](../fd_reconstruction/README.md) 与 [B 组迁移边界](../fd_reconstruction/ADVFD_BASELINE.md)；下述候选方法 C 及真实参考估计器切换仍是待实现设计。

## 三份代码如何分工

| 来源与实际入口 | 可复用内容 | 需要改写或隔离的部分 |
| --- | --- | --- |
| Grounded 的 `main.py`、`models/adapter.py`、各 tokenizer 目录 | 原图到重建图的链路、预训练权重加载、不同 tokenizer 的适配 | 原循环混合普通 GAN 判别器与多种损失；不能把这个判别器直接当成 AdvFD 的动态表征 |
| 官方 FD-Loss 的 `frechet_distance/losses.py`、`queue.py`、`repr_models.py` | 可微 FD、固定特征空间的重建侧队列或 EMA、表征预处理 | `main_fd.py` 是从噪声生成图片的入口，不作为我们的训练入口 |
| AdvFD 的 `frechet_distance/adversarial.py` 及 `main_fd.py` 中动态分支 | `FeatureStatsEMA`、真实特征 whitening、动态 FD、交替更新和动态状态保存 | 主文件混有生成、LoRA、PatchGAN、DMD 和其他实验选项；只迁移本项目需要的机制 |
| Grounded 的重建与表征评价脚本 | 配对样本导出、多个表征、逐图指标 | `eval_reconstruction.py` 中部分指标计算被注释，不能把成功导图视为完成评价 |

Grounded 的 `FrechetLoss.forward()` 会更新队列，特征提取器默认冻结；`DecoderAdapter.eval_mode()` 还会将参数梯度全部关闭。迁移时必须显式处理这些行为，而不只是添加一个新的 loss 类。

## 建议目录

项目根目录的 `fd_reconstruction` 按以下目标结构逐步实现；现有 `fd_hacking_stage1` 的生成实验和结果保留不动。

```text
fd_reconstruction/
├── third_party/                  # 三份固定版本源码，只读参考及来源记录
├── configs/                     # 公共配置、三组方法、三个消融、机器路径
├── src/recon_fd/
│   ├── data.py                  # ImageNet、样本 ID、配对变换
│   ├── tokenizers/              # E→latent→D，适配 Grounded 的模型
│   ├── representations/         # Inception、CLIP、DINOv2 等，独立于 E
│   ├── vendor/                  # FD-Loss 与 AdvFD 官方原码，固定版本并校验
│   ├── objectives/
│   │   ├── frechet.py           # 独立评价 FD，不用于 A/B 训练核心
│   │   ├── official_fd.py       # A/B 静态分支，调用官方 queue/losses
│   │   ├── statistics.py        # 固定统计、当前参数重算、可选 EMA、静态特征队列
│   │   ├── whitening.py         # whitening 与数值正则化
│   │   ├── static_fd.py         # 仅供 FD-only、AdvFD 和消融
│   │   └── adaptive_fd.py       # 动态 FD，共享于 AdvFD 和候选方法
│   ├── engine/                 # 训练阶段、梯度开关、优化器、完整断点
│   ├── evaluation/             # 独立冻结评价器，正式 50k 统计与配对指标
│   └── diagnostics/            # 指标分歧、表征漂移、EMA 滞后、异常样本
├── train.py                     # 唯一训练入口
├── evaluate.py                  # 独立评价入口，不修改训练状态
├── tests/                       # 梯度、统计、数值、断点和评价协议测试
└── jobs/                        # 单 GPU 训练与评价作业
```

不把三个仓库同时加入 `sys.path`：它们都有 `models`、`utils` 或 `frechet_distance` 等同名包，容易导入错误实现。确需迁移的代码进入 `recon_fd` 命名空间，保留来源与许可信息，并用原实现做一致性测试。先完成必要模块，不建设通用训练平台。

## 模块之间的约定

**重建模型与评价表征彻底分开。** `Reconstructor` 接收原图，返回重建图；`Representation` 接收图片，返回指定层和 pooling 的特征。即使 tokenizer encoder 和评价器都叫 DINOv2，也不能共享可变权重。第一阶段先接通一个连续 latent tokenizer，再扩展其他家族；离散 tokenizer 必须单独确认 straight-through 等训练路径，不能用不可微的 token ID 接口冒充联合训练。

**统一图片与样本约定。** 公共接口采用 RGB 浮点 `[0,1]`；Grounded 所需的 `[-1,1]` 转换由 tokenizer 适配层完成。resize、crop、antialias、pooling、clamp 和导出量化规则都记录在配置中。原图与重建图来自同一次输入变换，携带相同 `sample_id`，不能分别随机裁剪。保持各基线相同的输出处理，并记录裁剪饱和比例。

**真实参考估计器可替换，统计状态由训练器管理。** 动态真实侧支持的目标设计包括当前 batch 统计、当前 ψ 重编码固定真实图片池，以及可选 EMA；不能把“动态参考”在接口中等同于 EMA。真实图片池缓存的是图片或样本 ID，不是旧 ψ 下的特征。重算统计必须标记 ψ 版本，不能将旧版本冒充当前版本；样本数和重算频率单独配置。

`statistics.preview(features)` 计算候选统计，不修改缓存；涉及梯度的路径按 loss 与 whitening 的 detach 协议处理。`statistics.commit(detached_moments)` 只提交有状态估计器的更新；当前参数直接重算无需进行 EMA 提交。真实侧与重建侧状态分开。采用 EMA 时分别维护均值和二阶原始矩，历史缓存不保留计算图。动态分支不直接复用来自旧表征的长期特征队列；EMA 本身也存在跨表征历史混合，必须测量其滞后。

**参数训练范围与 train 模式分开。** 候选方法 C 拟将动态特征提取器全部参数加入 D 优化器；B 则保持论文范围：Inception 全参数，SigLIP/MAE rank-16 LoRA，不提前改成全参数。BatchNorm 的 running statistics 是 buffer，不是可训练参数；可以固定其运行统计，同时训练 affine 参数。沿用 AdvFD 表征分支的 eval-mode 行为，单独控制参数梯度，并输出 trainable parameter 清单。G-step 冻结这些参数，但不能对重建图经过 ψ 的前向使用 `no_grad()`。

**损失保持原有数值语义。** 原代码包含 `FD / (FD.detach() + eps)` 归一化，不能迁移时悄悄删除；日志必须同时保存 raw FD 和归一化 loss，后者接近 1 不是训练饱和证据。whitening 的正则化和 detach 位置需逐项对齐 AdvFD，不能用看似等价的公式直接替代。相关矩阵运算退出低精度 autocast，保留明确的高精度路径。

## 三组方法与三个开关

| 配置 | 静态 FD | 动态表征 | 动态真实参考统计 |
| --- | --- | --- | --- |
| `fd_only` | 有，固定表征和真实参考 | 无 | 不适用 |
| `advfd_reconstruction` | 有 | 按原 backbone 配置，Inception 为全参数 | EMA，保留原机制 |
| `ours` | 无 | 全参数 | 跟随当前 ψ 更新，估计方式显式选择 |
| `ours_add_static` | 有 | 与 ours 相同 | 与 ours 相同 |
| `ours_lora` | 无 | 同一个支持 LoRA 的 backbone | 与 ours 相同 |
| `ours_fixed_reference` | 无 | 与 ours 相同 | 固定共同初始化值 |

对应配置项为 `static.enabled`、`adaptive.trainable_scope` 和 `adaptive.real_stats.mode`。`ours` 不实例化静态训练网络，也不依赖静态分支为动态分支隐式提供特征。评价网络仅在独立评价中加载。

候选方法 C 的拟定真实侧模式为 `current_batch`、`reencode_pool`、`ema`；`frozen_initial` 仅供机制消融。前三种模式用于比较采样噪声、统计滞后和计算成本，尚未选定 C 的最终估计器。已实现的 B 强制保留原有真实侧 EMA，不提供 C 的估计器切换。A 配置中未启用的 `adaptive.real_stats.mode: ema` 只是预留值，不表示 C 必须采用 EMA。

三组都使用同一重建模型初始化、数据与 E+D 更新范围。单因素消融锁定其余设置，尤其是重建侧 EMA、动态 FD 权重、G/D 学习率、更新频率、whitening 和初始化。如果 `ours_add_static` 与匹配版 AdvFD 配置完全相同，就复用这一组，不重复跑。LoRA 对照必须在同一 ViT backbone 上进行，不能拿 Inception 全参数和 MAE LoRA 直接归因。

区分“尽量迁移原论文设置的基线”和“只改变一个因素的匹配消融”。原 AdvFD 脚本有静态训练预热及动态权重调度；去掉 static 后若照搬动态权重为零的初始区间，会变成没有训练信号。候选方法先做不更新模型的统计初始化，再启用动态目标；匹配消融采用相同日程。任何对原基线日程的调整都在配置中公开记录。

## 一个训练迭代怎么走

用户已确认 B 跟随官方可执行代码：先 D-step 后 G-step，复用同一批重建图；下述流程即为已选定顺序。论文 Algorithm 1 的先 G-step 后 D-step、用更新后的 G 重新生成仍作为原始差异记录，不再是待确认项。候选方法确定后再锁定共同日程，不能把两种顺序混用后归因于方法；此次确认不改变训练预算。

1. 读取原图 batch，以当前 E+D 计算重建并保留 G 计算图；D-step 只接收它的 detached 副本。根据配置准备当前 ψ 下的真实参考候选统计。B 的 EMA 模式读取历史状态参与估计；C 拟定的直接重算模式重新编码指定真实样本。
2. 若本轮需要 D-step，固定重建模型的参数与可变运行状态，更新 ψ，最大化动态 FD。真实参考用于 whitening 的 detach 规则保持与原实现一致；不在每次 loss 前向中提交有状态统计。
3. 固定更新后的 ψ，重新计算真实参考候选统计及重建的动态特征，更新 E+D；B 与官方实现一样复用本轮同一重建图的 G 计算图，不额外重建另一批图片。直接重算模式不得复用 D-step 更新前的参考特征；EMA 模式用新 ψ 的特征和未重复提交的历史状态形成候选估计。候选方法仅使用动态 FD；AdvFD 加上静态项；FD-only 跳过 D-step，只使用静态项。不能用 detached 重建代替 G 的计算图。
4. 完成反向与有效优化步骤后，按约定频率提交有状态估计器的更新。用于本次更新的真实和重建特征来自同一 ψ 版本，默认采用本轮 G-step 前向的 detached 统计；记录该版本与更新计数，不额外将 D-step 的同一 batch 再计入一次。直接重算的真实统计不混入历史 EMA；重建侧 EMA 若保留，仍需独立记录其历史混合。
5. 保存 raw FD、有效 loss、各模块梯度、真实与重建特征尺度、统计模式、ψ 版本、重算或 EMA 更新次数和样本计数。非有限值先保存诊断并停止，不把部分失败迭代当作正常 checkpoint 继续。

初始化从同一训练数据协议与初始 ψ 得到真实统计；预计算文件仅在权重、层、pooling 和预处理匹配时作为初值。按选定模式完成必要的 EMA 初始化或真实图片池准备后才训练；直接重算模式在 ψ 改变后重新计算，不能将初值长期复用。动态分支设置 `frozen_initial` 时才永久锁定真实参考；这是机制消融，不是原 AdvFD 的实现。FD-only 的表征固定，继续保留固定真实参考。

微 batch、一个 FD 统计覆盖的样本数、优化器累积步数和 EMA 更新次数分别记录。普通梯度累积不等于对拼接后的大 batch 算一次 FD；首版不把二者混称为有效 FD batch。先用显存检查与梯度 checkpointing 解决单卡问题，不用未验证的统计近似掩盖 batch 限制。

## 评价与证据输出

训练和正式评价使用不同的统计对象。训练动态参考会变；正式评价固定网络、真实参考与预处理，对完整 50,000 张 ImageNet 验证图做普通样本统计，不使用训练 EMA 代替。参考缓存带数据 split、样本数、权重、pooling 与预处理指纹，不兼容就拒绝加载。

单卡顺序执行：先用一个检查点重建并保存固定 ID 的 50k 图，再按表征逐个加载评价器，避免同时驻留多个大模型。统计累积采用流式方式。每个 checkpoint 的导图目录独立，完成标记必须核验 50k 个唯一 ID，不能把旧图、缺图或重复图混入。浮点配对指标与量化导图指标分开标注；检查点之间始终使用相同协议。

每个 run 保存三类输出：

- 训练状态：解析后的完整配置、来源版本、E+D、ψ、优化器与调度器、真实与重建估计器类型及状态、静态队列、随机状态与数据位置。真实图片池保存样本清单和预处理指纹；启用 EMA 时保存相应统计与更新计数。若启用模型权重 EMA，单独命名保存。断点必须支持恢复动态表征和统计，而不只是恢复 tokenizer。
- 固定评价：各表征 FD、LPIPS、PSNR、SSIM，按样本 ID 保存配对指标；DISTS 作为后续扩展。与训练共享表征家族的评价和真正 held-out 表征分别标注。
- 诊断：目标改善与独立评价恶化的候选区间、EMA 对当前真实统计的偏差、特征范数、异常残差和固定随机样本。小 batch 的统计偏差本身含采样噪声，不能直接叫漂移或 hacking。少量图像核查确认视觉含义，不输出未经验证的单一 hacking 分数。

动态 FD 不同时间点不直接视为同一把尺；需要时加载固定 ψ 快照与固定 whitening 统计交叉评价重建检查点。旧 pilot 中 near-floor 与震荡报警只能作为诊断参考，不能直接作为新实验成功判据。

## 必须先通过的测试

| 测试 | 要阻止的问题 |
| --- | --- |
| G-step 梯度与参数差分 | E、D 确实收到梯度并更新，ψ 不更新；避免 encoder 被适配器静默冻结 |
| D-step 梯度与参数差分 | 只有 ψ 更新；full 与 LoRA 的参数集合符合配置，不遗留非预期冻结层 |
| 静态分支禁用 | ours 的训练前向没有静态网络调用、隐藏静态 FD 或对应梯度 |
| 统计 preview 与 commit | 重复 forward 不改变 EMA；每轮只更新约定次数；真实与重建缓存不混用 |
| 当前参数参考重算 | ψ 改变后不复用旧特征；固定图片池按当前 ψ 重编码；直接重算模式不隐式混入 EMA |
| 数值与上游一致性 | 小型确定性特征上比较 raw FD、whitening 和梯度；相同统计应接近零，退化协方差不静默产生假分数 |
| 恢复一致性 | 连续运行与保存再恢复后的下一步，在确定性测试条件下结果一致 |
| 50k 评价完整性 | 无漏图、重复、错配，参考指纹正确；评价不会修改训练模型或训练 EMA |
| 配对打乱检查 | 打乱重建图与原图配对不改变边缘分布 FD，却应影响配对指标，验证系统能暴露“分布对了但重建错了” |

最后一项也是方法边界：即使动态 FD 优化理想，也不自动保证每张重建图对应自己的原图。我们暂不通过新增配对损失改变主方法，但必须具备检测该失败的能力。

## 最小落地顺序

先做 tokenizer 适配、固定 50k 评价和 FD-only，确认原始重建能复现；再加入与官方数值对齐的动态分支，使 AdvFD 和 ours 只通过配置切换；随后完成统计更新、梯度与断点测试。第一轮只接一个 tokenizer 和 Inception 动态分支，跑通后再补同 ViT 的 LoRA 对照及其他表征。作业始终单 GPU 顺序执行。

首次落地不引入生成模型、DMD、额外 GAN 判别器或新的配对训练损失，也不提前支持所有 tokenizer。正式实验前先锁定配置与评价协议，再提交 smoke test；smoke test 只验证实现，不作为研究结论。

## 已核对的代码来源

- Grounded-Frechet-Loss：用户提供的压缩包，SHA256 为 `247d5486ff1d497431331af4cb0c8e907f662dec70efca6a981efcefde2583b5`。以上重建适配与入口判断来自实际源码，不是只依赖 README。
- [官方 FD-Loss](https://github.com/Jiawei-Yang/FD-Loss/tree/5c03b8112fec8b9432631e4ce053c0d918cc24bc)：本次从官方仓库重新取得快照。`losses.py`、`queue.py`、`repr_models.py` 与现有本地副本逐文件比较一致。旧 `third_party/SOURCES.md` 中记录的完整 commit 有误，不应沿用。
- [官方 AdvFD](https://github.com/GasaiYU/AdvFD/tree/4e4cfed944e4fc38a75fae3ea7701ae9e5587060)：核对动态训练、统计与 whitening 实现及 `table_3_JiT_adv_fd_sim.sh` 等启动脚本。代码通用默认值不一定等于论文实验值，例如 whitening epsilon 在函数默认值和该脚本中不同；实现时必须显式配置。

新项目已将三份源码固定在 `fd_reconstruction/third_party/`，版本与许可边界见 [来源记录](../fd_reconstruction/third_party/SOURCES.md)。这些目录仅供参考，不参与运行时导入；旧 pilot 的占位目录及临时审核 checkout 不作为新项目的运行时依赖。
