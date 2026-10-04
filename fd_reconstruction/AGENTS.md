# 代码版本管理

用户要求本地与 nibi 之间的代码管理必须通过 GitHub push 和 pull。

- 代码、配置和作业脚本先在本地修改并验证，再限定提交范围，commit 后 push 到 GitHub 分支；nibi 使用对应分支的 pull 获取代码。
- 不通过 rsync、scp、SFTP 或临时复制覆盖服务器代码，不在服务器临时补丁后忘记回传 Git。数据、模型权重和实验产物不属于代码同步，不提交 Git。
- 远端首次部署可 clone；后续使用 `git pull --ff-only`。执行前检查远端改动和依赖该目录的作业，遇到冲突不强制覆盖、不丢弃现有工作。
- 提交只包含本任务文件，保留无关改动；禁止默认 force push。新开发分支使用 `codex/` 前缀。
- 运行实验时记录 Git commit；核对本地、GitHub 与 nibi 的版本，不能用本地已通过测试代替远端已更新的证据。
- `third_party/Grounded-Frechet-Loss-main/` 是用户提供且无明确再分发许可的本地参考，不推送公开仓库。FD-Loss 与 AdvFD 的公开 MIT 源码保留许可和固定版本记录。
