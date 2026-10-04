# C 组动态参考重建方法

C 的工程实现用于检验三个改动能否缓解重建中的 FD hacking，不预设有效。A/B 原数值核心与训练路径保持不变；C 使用独立目标与训练步骤，共享重建适配器、官方 whitening/EMA 核心、checkpoint 和独立评价。

## 已实现的三项改动

1. 主方法 `ours` 不加载静态训练网络，不计算 static FD，也不借助静态分支初始化统计。动态 ψ 直接从其预训练权重构造。
2. D-step 将 ψ 的全部参数放入优化器；G-step 冻结 ψ 并联合训练 E+D。网络保持 eval 模式，BatchNorm 的运行缓冲不更新，affine 参数仍可训练。参数清单保存在 `advfd_parameters.json`。
3. 真实图片池固定，真实特征统计按当前 ψ 重算。每次 D 更新使旧版本缓存失效；G-step 使用更新后的 ψ 重算参考。ψ 没有改变的轮次才可复用缓存，不跨版本使用旧特征。

真实参考用 FP64 分块样本均值和协方差（ddof=1），不构建反向图；whitening 的 detach 与正则化保留官方机制。默认训练参考池 50k，重建侧初始化 50k，正式独立评价至少 50k。分块只是节省内存，不减少总样本数。

## 每步执行与保留项

先做不更新参数的真实/重建统计初始化，再进入训练。每轮只执行一次 E+D 重建，依次执行 D-step、G-step，复用同一重建图；D 接收 detached 副本，G 保留输入梯度。D 最大化未归一化 whitened FD，G 最小化 `weight * FD / (FD.detach() + 0.01)`，默认 weight 0.1。

默认 D 每两轮更新一次，每次一步，梯度裁剪 1。C 没有静态目标，因此从初始化后立即启用非零动态权重，不能照搬 B 的 1000 步静态阶段和零起点 warmup。B 本身不改日程；C 的加回 static、LoRA 和统计估计器消融采用 C 的同一日程，不能把 B/C 的全部差异归因于三项改动。

重建侧仍直接使用官方 `FeatureStatsEMA`，beta 0.99，每个成功 G 步提交一次。它包含旧 ψ 与旧 E+D 的历史信息，日志明确标记 `fake_statistics_historical`。因此当前实现只保证重编码真实池使用当前 ψ，并不保证两侧统计都完全实时，也不证明已消除 hacking。未加入像素、感知、GAN、额外语义锚点或新 FD 公式。

## 配置与代码

| 配置 | 与 C 主方法的区别 |
| --- | --- |
| `ours_reconstruction.yaml` | Inception 全参数，真实池按当前 ψ 重算，无 static |
| `ours_reconstruction_mae.yaml` | 换为 MAE 全参数，支持 gradient checkpointing |
| `ours_lora_mae.yaml` | 同一 MAE 改为 rank-16 QKV LoRA，用于参数范围消融 |
| `ours_add_static.yaml` | 加回 Inception static FD，使用 C 日程，不冒充 B |
| `ours_real_ema.yaml` | 真实侧改用官方历史 EMA，每个成功 G 步提交一次 |
| `ours_fixed_reference.yaml` | 故意固定初始真实统计，仅用于机制消融 |
| `ours_smoke.yaml` | 随机小模型和合成图片，仅用于工程检查 |
| `ours_nibi_smoke.yaml` | 真实 SD-VAE/Inception，128 张训练参考、32 张评价图，仅工程检查 |

Inception 在 B 中已经全参数训练，不能用 Inception B/C 对比声称验证了全参数改动。该因素必须使用同一个 ViT backbone 的 LoRA/full 对照。首版支持匹配的 MAE/SigLIP 架构；提供 MAE 配置，不自动下载额外权重。

`objectives/candidate_fd.py` 管理独立动态网络与重建 EMA，`objectives/real_reference.py` 管理参考池和版本，`engine/candidate.py` 管理 D→G。`config.py` 对主方法及各消融分别校验，不允许无声明地开启 static、切换 LoRA 或改变参考估计器。当前 batch 直接估计尚未实现，主方法不会悄悄降级为 batch 统计或 EMA。

checkpoint 包含 ψ 版本、参考缓存、重建 EMA、可选真实 EMA 和两个优化器。配置签名记录估计方式、池大小及 seed；数据指纹和 `real_reference_manifest.json` 记录池身份。`real_feature_version` 与 `loss_psi_version` 应在主方法中相等；故意固定参考的消融明确例外。EMA 模式记录为历史混合，而不是当前参数精确重编码。

## 验证与运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python train.py --config configs/ours_smoke.yaml
```

本地验证包括 A/B 回归、无静态构造依赖、与直接官方动态 FD 计算对照、参考按当前参数重编码、相同版本缓存复用、单步和多步 D 顺序、梯度隔离、重复/过期提交拒绝、真实 timm ViT full/LoRA、随机 AutoencoderKL 联合训练，以及主方法/EMA/固定参考的断点恢复和独立评价。小模型结果不能支持视觉质量结论。

`jobs/ours_smoke_nibi.sbatch` 调用 `scripts/ours_nibi_smoke.py`，只允许在单张 CUDA GPU 的 SLURM allocation 内执行。脚本检查指定 Git commit，依次做 tiny C、tiny EMA、真实 C 的训练 2 步→恢复到 4 步→独立连续 4 步对照，再评价真实 C 的初始和最终各 32 张重建。最后用 checkpoint 的当前 ψ 独立重编码真实池，核对缓存，不只核对版本标签。GPU 恢复比较报告容差以及是否逐位相同，不把容差一致表述为逐位一致。

每阶段输出日志、耗时和峰值显存；只有所有断言完成才写 `result.json` 的 `status: passed`。集群排队不等于验证通过。先确认这个短测的结果及重编码成本，再安排正式 50k 实验；训练预算暂未改成原论文配置。

所有代码经本地 commit/push → GitHub → nibi pull；不复制覆盖旧工作目录，也不取消旧 A 作业。CUDA 预训练集成与正式 50k 流程在收到实际成功结果前均属于待验证项。
