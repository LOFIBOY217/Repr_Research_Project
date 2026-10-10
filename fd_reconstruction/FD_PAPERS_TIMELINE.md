# FD / FID 论文时间线：图像生成与重建

更新时间：2026-10-10。本文是与**图像生成、图像重建和 FD hacking**相关的精选阅读清单，并非穷尽性文献综述。时间按论文**首次公开**日期排列；FID 是使用 Inception 特征的 Fréchet Distance (FD)。

| 首次公开 | 论文 | 与本项目的联系 |
| --- | --- | --- |
| 2017-06 | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) | 新提出 FID，并将其作为替代 Inception Score 的生成图像评价指标；与本项目的 FD hacking 方法无直接联系。 |
| 2019-11 | [Effectively Unbiased FID and Inception Score and where to find them](https://arxiv.org/abs/1911.07023) | 有限样本 FID 的偏差随模型变化；即使样本数相同，也可能误排模型。属于**评价估计问题**，不是 FD hacking。 |
| 2020-03 | [Image Generation Via Minimizing Fréchet Distance in Discriminator Feature Space](https://arxiv.org/abs/2003.11774) | 判别器学特征，生成器最小化该特征空间的 FD；动态特征 FD 训练的早期先例，但不是图像重建。 |
| 2020-09 | [Backpropagating through Fréchet Inception Distance](https://arxiv.org/abs/2009.14075) | FastFID：让 FID 可用于反向传播和训练。 |
| 2021-04 | [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://arxiv.org/abs/2104.11222) | 缩放、压缩等预处理会改变 FID；提醒我们固定统一的 50K 评价协议。 |
| 2022-03 | [The Role of ImageNet Classes in Fréchet Inception Distance](https://arxiv.org/abs/2203.06026) | 改变类别直方图即可降低 FID，而图像质量未必改善。 |
| 2023-06 | [Exposing Flaws of Generative Model Evaluation Metrics and their Unfair Treatment of Diffusion Models](https://arxiv.org/abs/2306.04675) | 比较多个特征提取器、指标和人类评价，强调 DINOv2 的价值。 |
| 2023-11（CVPR 2024） | [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603) | 分析特征表达、正态假设、样本效率与感知失配。 |
| 2024-06 | [F?D: On Understanding the Role of Deep Feature Spaces on Face Generation Evaluation](https://openaccess.thecvf.com/content/CVPR2024W/ReGenAI/html/Kabra_FD_On_Understanding_the_Role_of_Deep_Feature_Spaces_on_CVPRW_2024_paper.html) | 不同特征空间对不同视觉属性的敏感性不同。 |
| 2025-05 | [TokBench: Evaluating Your Visual Tokenizer before Visual Generation](https://arxiv.org/abs/2505.18142) | **重建**研究：不同 tokenizer 的文字、人脸缺陷可能被常规指标漏掉；并非 FD hacking 专论。 |
| 2026-04 | [Representation Fréchet Loss for Visual Generation](https://arxiv.org/abs/2604.28190) | FD-Loss，项目 A 基线；也展示过度优化单一特征空间的风险。 |
| 2026-06 | [The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation](https://arxiv.org/abs/2606.20536) | FID 的训练种子与采样种子波动；小差异不宜过度解释。 |
| 2026-08 | [AdvFD: Boosting Visual Generation via Adversarial Fréchet Distance Loss](https://arxiv.org/abs/2608.11205) | 项目 B 基线：静态 FD 加动态对抗特征空间。 |
| 2026-10 | [Sample-Optimal Estimation of the Fréchet Inception Distance](https://arxiv.org/abs/2610.07114) | 新预印本：FID 有限样本估计误差与 50K 样本预算。 |

## 1. Heusel et al. (2017)：FID 的起点

**原文：** [论文页面及摘要](https://arxiv.org/abs/1706.08500) · [PDF](https://arxiv.org/pdf/1706.08500)；NeurIPS 2017。以下为覆盖摘要全部要点的中文意译，而非逐词对照。

### 摘要中文意译

生成对抗网络（GAN）可以用复杂模型生成逼真的图像，即使这类模型的最大似然难以计算；但 GAN 训练是否会收敛，当时仍缺少证明。作者提出“双时间尺度更新规则”（TTUR）：对于采用任意 GAN 损失的随机梯度训练，判别器和生成器分别使用各自的学习率。借助随机逼近理论，作者证明，在一定条件下，这样的训练会收敛到一个驻定的局部纳什均衡。论文还把这一结论扩展到常用的 Adam 优化器，并把 Adam 的动态解释为带摩擦的重球运动，因而倾向于目标函数中较平坦的极小值。为了评价 GAN 的图像生成效果，作者另外提出 Fréchet Inception Distance（FID），认为它比 Inception Score 更能反映生成图像与真实图像的相似程度。实验在 CelebA、CIFAR-10、SVHN、LSUN Bedrooms 以及 One Billion Word Benchmark 上，报告 TTUR 改善了 DCGAN 和改进型 Wasserstein GAN（WGAN-GP）的学习表现，优于通常的 GAN 训练方式。

### 贡献与 FD 的联系

1. **训练贡献是 TTUR。** 判别器和生成器采用不同的学习速率，并在论文假设下讨论局部收敛；不能把这篇论文误读成“用 FID 训练 GAN”。
2. **评价贡献是 FID。** 将真实图和生成图送入预训练 Inception-v3，用特征的均值、协方差拟合两个高斯分布，再计算它们的 Fréchet 距离。论文用它来**评估**生成模型，而不是作为优化器的目标。
3. **与我们的课题：** 后来的 FD-Loss 把这类分布距离转为训练目标；AdvFD 和我们的 B/C 则研究特征空间可被优化“钻空子”的问题。原始 FID 依赖固定 Inception 特征，也只比较两个图像集合的分布，不检查重建图是否对应自己的输入原图。因此在重建任务中，还必须单独报告配对指标和视觉错误；这属于我们的研究推论，不是 2017 论文自己的结论。

后续阅读重点：2017 年论文的 FID 测量定义与当时的扰动实验，应和 2022 年类别偏差、2023–2024 年感知失配研究放在一起看；早期“优于 Inception Score”不等于“FID 对所有视觉缺陷可靠”。

## 2. Chong & Forsyth (2019)：有限样本 FID 的偏差

**原文：** [论文页面及摘要](https://arxiv.org/abs/1911.07023) · [PDF](https://arxiv.org/pdf/1911.07023)；首次公开于 2019 年 11 月，发表于 CVPR 2020。以下为覆盖摘要全部要点的中文意译。

### 摘要中文意译

论文指出，用于评价生成模型的两个常见指标——Fréchet Inception Distance（FID）和 Inception Score（IS）——都有偏差：用有限个样本算出的分数，其期望并不等于指标的真实值。更糟的是，偏差取决于被评价的具体模型，因此模型 A 可能仅因偏差更小，就得到比模型 B 更好的分数。让所有模型使用相同数量的样本也无法解决这个问题；作者据此认为，按当时通常方式计算的 FID 或 IS 来比较模型并不可靠。接着，作者提出通过外推，估计样本数趋于无穷时的分数，分别称为 FID∞ 和 IS∞，从而得到实际中几乎无偏的估计。准确外推又需要可靠的有限样本分数；作者发现，准蒙特卡罗积分能够显著改进有限样本 FID 和 IS 的估计。外推后的分数可以直接替换通常计算的有限样本分数。此外，在 GAN 训练中使用低差异序列，也能让得到的生成器表现略有改善。

### 贡献与本项目的联系

1. **核心发现：** 有限样本 FID 的偏差不仅与样本数有关，也与模型有关。统一使用 50K 张图是必要的比较控制，但不等于完全消除偏差或保证细微排名可信。
2. **处理办法：** 对不同样本量的得分按 `1/N` 外推，估计 `FID∞`；准蒙特卡罗采样用于降低随机生成器的有限样本估计误差。后者不应不加区分地照搬到固定输入、确定性输出的重建实验。
3. **与 FD hacking 的界限：** 本文研究的是**分数估计不准**，不是模型优化固定特征空间后产生视觉缺陷。即便估计出无偏的 FID，仍可能存在特征表示与人类视觉不一致的问题。对我们的 A/B/C 对比，它主要提醒我们谨慎解释接近的 50K FD 数值，必要时做重复抽样或样本量外推。

## 3. Doan et al. (2020)：在动态判别器特征中训练 FD

**原文：** [论文页面及摘要](https://arxiv.org/abs/2003.11774) · [PDF](https://arxiv.org/pdf/2003.11774)；首次公开于 2020 年 3 月。以下为覆盖摘要全部要点的中文意译。

### 摘要中文意译

对于给定的图像生成问题，图像所处的内在流形通常维度较低。作者据此认为，与其在原始像素空间训练 GAN 生成器，不如在能够表达这一流形的低维特征空间里，最小化真实图像与生成图像之间的分布距离；他们选用 GAN 判别器的特征空间作为这种表示。分布距离有两种选择：Fréchet Distance（FD）或直接计算最优传输（OT），相应得到 Fréchet-GAN 和 OT-GAN 两种新方法。使用 FD 的想法受到 FID 作为图像生成评价指标取得成功的启发。Fréchet-GAN 有几项优势：作者提出了高效且数值稳定的 FD 及其梯度计算方法；FD 的计算开销显著低于 OT，因此训练时可以使用更大的 mini-batch。作者还在多个基准数据集上实验，报告 Fréchet-GAN（尤其如此）和 OT-GAN 的图像生成能力，优于所比较的基于 Wasserstein 距离的代表性原始与对偶方法。

### 方法、贡献与本项目的联系

1. **训练方法：** 判别器照常学习区分真实与生成图像；生成器不直接用常规 GAN loss，而是最小化两组图像在判别器中间特征中的 FD。两者交替更新，所以特征空间会随训练变化。该 FD **不是**固定 Inception 特征上的 FID；论文仍用 FID 等指标评价结果。
2. **技术贡献：** 为使 FD 可训练，论文处理协方差矩阵平方根的数值稳定性及梯度问题；同时比较了判别器特征空间中的 FD 与 OT。
3. **对我们很关键的先例：** “让特征提取器动态学习，再用 FD 训练图像模型”在 2020 年就已有实现，因此不能把这个宽泛想法当作我们 C 方法的全新贡献。但本文研究的是**无条件图像生成**，判别器通过真假分类训练，并非像 AdvFD 那样直接最大化 FD；它也没有研究配对图像重建、FD hacking 诊断或我们的三项改动组合。
