# RM-DMVT Engineering Project

RM-DMVT is the engineering implementation of the paper's unified model:

```text
class-conditional DDPM
        -> inverse quantile transform
        -> rank-matched discrete correction
        -> balanced training cache
        -> multi-view Transformer classifier
```

The project keeps the two components modular for tuning and ablation, while `RankMatchedDiffusionMultiViewPipeline` exposes one unified experiment pipeline.

## Open in PyCharm

1. Open **`D:\论文\code`** as the project directory.
2. In *Settings > Python Interpreter*, create or select a Python 3.10 interpreter.
3. Install `uv` once, then run:

   ```powershell
   uv sync --all-groups
   ```

4. PyCharm reads `pyproject.toml`, recognises the `src/` layout, pytest, ruff and basedpyright settings automatically.
5. Do not mark `legacy/` as a source root. It is retained only for provenance.

The server uses Python 3.10, so the project deliberately targets Python 3.10 even if a newer local interpreter is available.

## Directory layout

```text
code/
├── pyproject.toml            single dependency and quality configuration
├── configs/                  reproducible baseline, tuning and server presets
├── docs/                     architecture, protocol and decision records
├── src/ids_diffusion/
│   ├── cli/                  ids-prepare / ids-train / ids-tune / ids-eval
│   ├── data/                 official split, thinning, views and cache
│   ├── models/               DDPM, multi-view encoder and classifier
│   ├── training/             correction, balancing, fitting and pipeline
│   └── tuning/               constrained search spaces and Optuna objectives
├── tests/                    unit and CLI tests
├── scripts/                  experiment, server and deployment helpers (lint-gated)
├── evidence/paper_results/   archived server results behind the reported tables
├── server_archive/           immutable snapshot of the original experiment code
├── runs/                     local outputs, ignored by Git
└── legacy/
    ├── scripts/              earlier one-off scripts, not active code
    └── manuscript_tools/     Word and PDF tooling for the paper, not experiment code
```

`scripts/` holds only what a reproduction or a server run needs: deployment,
smoke tests, run launchers and result inspection. The manuscript and
response-letter tooling was moved to `legacy/manuscript_tools/` so that nothing
in the reproduction path can be mistaken for typesetting code. Both `legacy/`
subdirectories are excluded from the quality gates; `scripts/` is not.

## Reproducing the paper

`docs/项目手册.md` is the full handbook (in Chinese): five-minute start,
repository map, command reference, evidence policy and troubleshooting.
`docs/DATASETS.md` lists the publisher sources, the files each loader reads and
the split policy. The whole reported matrix runs through one command group:

```powershell
uv run ids-reproduce plan
uv run ids-reproduce check  --data-root nslkdd=D:\data\NSL-KDD
uv run ids-reproduce run    --data-root nslkdd=D:\data\NSL-KDD --output runs\reproduction --device cuda
uv run ids-reproduce verify --metrics runs\reproduction\metrics
```

`plan` prints the 18 runs behind the tables, `check` refuses to start when a
dataset file is missing, `run` is resumable and writes one cache, metrics file
and log per job, and `verify` differences each rerun against the archived
server result for the same dataset and seed.

### Checking the paper without a GPU

Every headline number in the manuscript is recomputed from the archived server
results committed in this repository:

```powershell
uv run pytest tests/test_paper_claims.py
uv run ids-reproduce claims
```

This takes seconds, needs no dataset and no GPU, and covers the four-benchmark
ablation, the multi-class result, the adversarial cost of augmentation, the
residual imbalance left by the expansion cap, the cost of the discrete
correction against its pre-declared tolerance, and the model-independent
balancing effect. A mismatch means the archive and the manuscript disagree.

`docs/审计对照表.md` maps every manuscript table to its evidence files and to
the archived script that produced them, for reviewers auditing coverage.

`server_archive/legacy_paper/` is a read-only provenance snapshot of the code
that produced the published numbers, with a per-file SHA-256 manifest. It is
kept unmodified: the maintained package here reruns the same protocol, and the
snapshot lets a reviewer confirm nothing was quietly rewritten.

## Why tuning is staged

Classifier and generator parameters are not searched blindly together.

1. **Classifier search** uses fixed corrected training caches. Only Transformer and optimiser parameters vary.
2. The top three classifier configurations are repeated with seeds 42, 123 and 456.
3. **Diffusion search** fixes the selected classifier and regenerates corrected data for each trial.
4. The final locked configuration is rerun on all four datasets and all four ablations.

The test set never participates in parameter selection. Optuna maximises validation macro F1 only.

## Prepare corrected caches

Run once per seed and dataset on the GPU server:

```bash
ids-prepare \
  --data-root /opt/UNSW-NB15 \
  --dataset unsw \
  --task binary \
  --output runs/cache/unsw_seed42.npz \
  --seed 42 \
  --device cuda
```

`--dataset` accepts `unsw`, `nslkdd`, `cicids2017` and `cicddos2019`.
`--task multiclass` is available for the two benchmarks that publish an attack
category column.

The cache contains raw training, corrected balanced training, validation, untouched real test data and the exact view partition. It contains no executable Python objects.

## Tune the classifier

```bash
ids-tune classifier \
  --cache runs/cache/unsw_seed42.npz \
  --output studies/classifier-best.json \
  --storage sqlite:///studies/rm-dmvt.db \
  --trials 30 \
  --device cuda
```

Fast screening uses one seed. The best three configurations must then be confirmed with all three seeds before any claim is made.

## Tune the diffusion module

```bash
ids-tune diffusion \
  --data-root /opt/UNSW-NB15 \
  --output studies/diffusion-best.json \
  --storage sqlite:///studies/rm-dmvt.db \
  --trials 15 \
  --seed 42 \
  --device cuda
```

This stage regenerates and corrects training data for each trial, so it is substantially slower than classifier tuning.

## Train and evaluate four variants

```bash
ids-train \
  --cache runs/cache/unsw_seed42.npz \
  --output runs/final/seed42.json \
  --seed 42 \
  --device cuda

ids-eval runs/final/seed42.json
```

The output always contains:

- full RM-DMVT;
- without diffusion;
- without multi-view;
- neither component.

## Quality gate

```powershell
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest
uv run pytest -m slow
```

`pytest` runs the fast suite. `pytest -m slow` drives the real reproduction
entry point over miniature synthetic benchmarks, covering every dataset and task
combination the paper reports, still without real data or a GPU.

The executable release gate is `ruff` plus the test suite. The strict
`basedpyright` configuration is retained as a type-debt audit rather than a
passing gate; findings there are recorded, not suppressed.

On Windows, a Python 3.10 editable install can drop its generated `.pth` entry
when the checkout path contains non-ASCII characters. Install the wheel form in
that case:

```powershell
uv sync --python 3.10 --all-groups --no-editable
uv run --no-sync ids-reproduce plan
```

## Server policy

- Long optimisation jobs run only outside the server's daytime peak.
- Smoke tests may run during peak hours but must finish within minutes.
- Every long job writes to a new run directory and never overwrites paper results.
- A stop time must be supplied by the job wrapper; unattended jobs are killed before 09:00.

## Current baseline

UNSW-NB15 macro F1:

```text
RM-DMVT   88.90
XGBoost   90.20
gap       -1.30
```

The tuning programme asks whether this gap comes from fixed parameters or from the known advantage of tree models on tabular data. Reaching 90.20 in one run is insufficient; the final configuration must improve the three-seed mean without destabilising NSL-KDD.
