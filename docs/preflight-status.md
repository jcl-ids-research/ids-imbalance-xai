# 发布前状态

**日期**：2026-09-08
**目标仓库**：`https://github.com/jcl-ids-research/ids-imbalance-xai`（尚未提交）

## 当前门禁结果

| 项目 | 结果 |
|---|---|
| `ruff check src tests scripts` | 通过 |
| `ruff format --check src tests scripts` | 通过，90 个文件 |
| `pytest` | 通过，44 项（默认跳过慢测试） |
| `pytest -m slow` | 通过，7 项端到端流水线 |
| `ids-reproduce claims` | 21 项论文数字全部一致，零漂移 |
| `ids-reproduce plan` | 正确列出 18 次运行 |
| 构建产物 | wheel 含 `py.typed`，元数据为 `License: MIT` |
| 模块大小 | 41 个包内模块，最大 210 行纯代码 |
| 凭据与个人信息扫描 | 612 个文件，零命中 |
| 单文件体积 | 无文件接近 GitHub 100 MB 限制 |

严格 `basedpyright` 配置保留为类型债务审计，当前**不是**通过门禁；问题被记录而非屏蔽。

## 本轮完成的工作

- 四个数据集统一接入：新增 NSL-KDD、CIC-IDS-2017、CIC-DDoS2019 加载器与数据集注册表，UNSW 增加多类支持；
- 新增 `reproduction/` 包与 `ids-reproduce` 入口，支持 `plan` / `check` / `run` / `verify` / `claims`；
- 新增无 GPU 的论文数字回归套件，从仓库内归档结果重算全部头条数字；
- 归档服务器实验源码快照（117 文件 + 逐文件 SHA-256）与权威结果（283 个文件）；
- 脚本按职责拆分：`scripts/` 保留 33 个实验与部署脚本并纳入门禁，136 个稿件排版工具移入 `legacy/manuscript_tools/`；
- 修复三个在 Python 3.10 下无法解析的脚本，并修正其中被双重转义、实际永远匹配不到的正则；
- 部署脚本改为从环境变量读取主机、指纹与口令，删除两个内嵌明文口令的一次性脚本；
- 补齐 `LICENSE`、`py.typed`、CI 工作流与 `CITATION.cff`；
- 修正 `pyproject.toml` 中与 LICENSE 文件矛盾的 `Proprietary` 声明，并验证 wheel 元数据为 MIT；
- 新增端到端流水线测试，覆盖论文报告的全部六种「数据集 × 任务」组合与续跑行为。

## 已知限制

- 复现全部 18 次运行需要 CUDA 设备与自行下载的公开数据集，成本以 GPU 小时计；
- 论文结论基于 3 个随机种子，属方向性证据，不支持显著性主张；
- 仓库不含数据集、模型权重、预测数组与 Optuna 数据库，这些仍保存在训练服务器；
- 本机无可用视觉模型，图件清晰度依据像素尺寸与 PDF 渲染文本确认，未做人工视觉复核。

## 尚未执行

- GitHub 账号注册与仓库推送（按要求推迟到全部就绪后进行）。

## 历史状态

2026-08-31 的上线前检查记录（当时 9 个本机测试、服务器 CUDA smoke、晚间调优排程）已由本文件取代。相关调优结论保留在
[tuning-result-20260901.md](tuning-result-20260901.md) 与 [outer-holdout-result.md](outer-holdout-result.md)。
