# 发布前状态

**日期**：2026-09-09
**仓库**：`https://github.com/jcl-ids-research/ids-imbalance-xai`（已公开）

## 当前门禁结果

| 项目 | 结果 |
|---|---|
| `ruff check src tests scripts` | 通过 |
| `ruff format --check src tests scripts` | 通过，102 个文件 |
| `pytest` | 通过，113 项（默认跳过慢测试） |
| `pytest -m slow` | 通过，7 项端到端流水线 |
| `ids-reproduce claims` | 39 项论文数字全部一致，零漂移 |
| `ids-reproduce plan` | 正确列出 18 次运行 |
| `ids-reproduce audit` | 9 个实验族源码、证据、命令声明全部存在 |
| `scripts/check_type_debt.py` | 通过，类型债务未增长 |
| 覆盖率 | 71.39%，CI 设 70% 下限 |
| CFF 1.2 / 构建 | `cffconvert --validate` 通过，wheel + sdist 构建成功 |
| 模块大小 | 全部包内模块 ≤ 250 行纯代码 |
| 凭据与个人信息扫描 | 零命中 |

严格 `basedpyright` 仍非通过门禁：现有 509 errors / 26 warnings 主要来自
numpy、pandas 与 torch 的未标注返回值。改为**基线棘轮**——
`scripts/check_type_debt.py` 记录当前计数并在 CI 中执行，
新增诊断会导致失败，计数只能下降。这样债务被冻结而非被屏蔽。

## 本轮完成的工作

- 四个数据集统一接入：新增 NSL-KDD、CIC-IDS-2017、CIC-DDoS2019 加载器与数据集注册表，UNSW 增加多类支持；
- 新增 `reproduction/` 包与 `ids-reproduce` 入口，支持 `plan` / `check` / `run` / `verify` / `claims`；
- 新增无 GPU 的论文数字回归套件，从仓库内归档结果重算全部头条数字；
- 归档服务器实验源码快照（117 文件 + 逐文件 SHA-256）与权威结果（283 个文件）；
- 脚本按职责拆分：`scripts/` 保留 33 个实验与部署脚本并纳入门禁，136 个稿件排版工具移入 `legacy/manuscript_tools/`（该目录其后未随代码发布，见文末补记）；
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

以上条目均已完成，保留于此以对照当时的计划。

## 发布后补记（2026-09-08）

- 仓库已推送至 `github.com/jcl-ids-research/ids-imbalance-xai`，可见性为 public；
- `legacy/` 未随代码发布，其后在本机被误删且不可恢复。该目录只含稿件排版工具，
  不在复现路径上，已发布代码对其引用为零，论文数字核验不受影响；
- 补齐 §4.2.5 训练侧稀缺性实验的证据（`evidence/paper_results/scarcity/`，3 个种子），
  该实验此前是全文唯一无归档证据的结论。论文 12 个相关数字与证据逐项吻合。

## 第三轮审计后的修补（2026-09-09）

外部审计指出正文与制品仍有问题，逐条处理如下：

- **正文过度表述**：「approximately 100% on every dataset and seed」改为
  「on all nine measured dataset-seed pairs」。保真度族实际只有 9 个组合
  （UNSW 三种子，其余三个数据集各两种子），原措辞暗示 12 个；
- **`verify` 漏比多分类**：`ids-reproduce verify` 此前只匹配 `*_binary_seed*.json`，
  6 个多分类重跑结果被静默跳过。现按 `task` 字段分派到对应证据族，
  缺参照时显式报 `SKIP` 而非无声忽略；
- **声明校验扩展**：新增「9 个保真度组合」「MMD² 8/9 改善」「5 项三种子同号
  （3 正 2 负）」三类断言，`claims` 由 33 增至 39。这些正是此前只写在正文、
  无自动核验的表述；
- **类型抑制清零**：5 处 `# type: ignore` 全部移除，改用 `TypeGuard`
  与边界处的 JSON 收窄。无效的 `--dataset` / `--task` 现在被明确拒绝而非静默通过；
- **模块拆分**：`claims.py` 一度增至 355 行，按职责拆为 `archive_io`（JSON 收窄）、
  `measurements`（度量）、`fidelity`（校正代价）与 `claims`（论文数字对照）；
- **覆盖率**：此前几乎无测试的模块现已覆盖——扩散模型 21%→100%、均衡 33%→100%、
  分类器训练 31%→100%、调参 runner 0%→100%、调参目标 0%→76%、搜索空间 0%→89%，
  测试由 48 项增至 120 项（113 快速 + 7 端到端），整体覆盖率 60%→71%，CI 设 70% 下限。新增测试锁的是机制而非数字：
  噪声调度与论文公式一致、扩容上限确实留下残余不平衡、合成行不会越出真实取值范围、
  早停真的提前结束、返回的是最佳权重而非最后一轮、调参目标拿不到评测分区、
  相同种子的 study 可复现；
- **发布元数据**：`CITATION.cff` 补 `authors`（匿名审稿期占位）并经 `cffconvert`
  验证符合 CFF 1.2；`pyproject.toml` 的 license 迁移到 SPDX 字符串，消除弃用警告。

## 复现面显式化（2026-09-09）

审计指出「全部实验族都有源码和证据，但没有全部纳入统一、可移植的复现入口」。处理如下：

- 新增 `reproduction/families.py`，把论文 9 个实验族登记为两类——
  `maintained`（主消融、XGBoost、训练侧稀缺性、调参/保留集）与
  `archived-only`（经典/深度基线、保真度/坍缩/校正、对抗、注意力、图件）；
- 新增 `ids-reproduce families`（列出）与 `ids-reproduce audit`（验证每个族的
  源码、证据、命令声明真实存在，缺失即非零退出），并纳入 CI；
- 训练侧稀缺性从 `scripts/minority_shift.py` 一次性脚本迁移为正式入口
  `ids-scarcity`，原脚本保留为薄包装转发，不破坏服务器既有调用；
- 文档删除「一键复现全部实验」的过度承诺，如实标注 archived-only 族
  证据可查但不可一键重生成。

## 历史状态

2026-08-31 的上线前检查记录（当时 9 个本机测试、服务器 CUDA smoke、晚间调优排程）已由本文件取代。相关调优结论保留在
[tuning-result-20260901.md](tuning-result-20260901.md) 与 [outer-holdout-result.md](outer-holdout-result.md)。
