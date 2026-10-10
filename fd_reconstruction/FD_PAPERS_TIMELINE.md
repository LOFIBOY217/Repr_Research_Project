# FD / FID 论文时间线：图像生成与重建

更新时间：2026-10-10。本文是与**图像生成、图像重建和 FD hacking**相关的精选阅读清单，并非穷尽性文献综述。时间按论文**首次公开**日期排列；FID 是使用 Inception 特征的 Fréchet Distance (FD)。

| 首次公开 | 论文 | 与本项目的联系 |
| --- | --- | --- |
| 2017-06 | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) | 提出 FID，最初用于**评价**生成图像。 |
| 2019-11 | [Effectively Unbiased FID and Inception Score and where to find them](https://arxiv.org/abs/1911.07023) | 有限样本 FID 的偏差和模型排序问题。 |
| 2020-03 | [Image Generation Via Minimizing Fréchet Distance in Discriminator Feature Space](https://arxiv.org/abs/2003.11774) | 在可学习的判别器特征空间中用 FD 训练生成器；动态特征 FD 的早期先例。 |
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
