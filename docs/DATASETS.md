# Datasets and reproduction inputs

The four public benchmarks are not redistributed with this repository. Each one
is downloaded from its publisher, placed in a directory of your choice, and
passed to the CLI. Nothing else is required to reproduce the reported runs.

## Sources and expected files

| Dataset | Publisher source | Files the loader reads |
|---|---|---|
| UNSW-NB15 | UNSW Canberra, *The UNSW-NB15 Dataset* release page | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| NSL-KDD | Canadian Institute for Cybersecurity, NSL-KDD | `KDDTrain+.txt`, `KDDTest+.txt` |
| CIC-IDS-2017 | Canadian Institute for Cybersecurity, IDS 2017 | every `*.csv` under `MachineLearningCVE/` |
| CIC-DDoS2019 | Canadian Institute for Cybersecurity, DDoS 2019 | every `*.csv` under `all/` |

For the CIC releases you may pass either the release root or the CSV directory
itself; the loader resolves the documented subdirectory when it is present.

## Split policy

- **UNSW-NB15 and NSL-KDD** keep the publisher's training and test files. They
  are never merged and never resampled together.
- **The CIC releases publish no official split.** One stratified split is drawn
  per seed, and every encoder, imputer and scaler is then fitted on the
  training partition alone.
- A validation partition is always carved out of training, never out of test.
- Balancing, diffusion training and rank-matched correction touch training rows
  only.

Because the CIC CSVs are time ordered, a bounded read per file is followed by a
stratified subsample rather than a head truncation, which would otherwise
return almost pure benign traffic.

## Verify inputs before running

```powershell
uv run ids-reproduce check `
  --data-root unsw=D:\data\UNSW-NB15 `
  --data-root nslkdd=D:\data\NSL-KDD `
  --data-root cicids2017=D:\data\CIC-IDS-2017 `
  --data-root cicddos2019=D:\data\CIC-DDoS2019
```

The command names every missing file and exits non-zero before any training
starts.

## Reproduce the reported matrix

```powershell
uv run ids-reproduce plan

uv run ids-reproduce run `
  --data-root unsw=D:\data\UNSW-NB15 `
  --data-root nslkdd=D:\data\NSL-KDD `
  --data-root cicids2017=D:\data\CIC-IDS-2017 `
  --data-root cicddos2019=D:\data\CIC-DDoS2019 `
  --output runs\reproduction `
  --device cuda
```

`plan` lists the 18 runs behind the reported tables: four benchmarks in binary
detection and the two category-labelled benchmarks in multi-class, each with
seeds 42, 123 and 456. `run` writes one prepared cache, one metrics file and
one log per job, and skips any job whose metrics already exist, so an
interrupted reproduction can be resumed.

To check the wiring on a laptop before committing GPU time, add `--smoke`. The
shortened schedule proves the pipeline executes end to end; it does not
reproduce reported numbers.

## Compare against the archived results

```powershell
uv run ids-reproduce verify `
  --metrics runs\reproduction\metrics `
  --evidence evidence\paper_results\phase1_corrected `
  --tolerance 0.02
```

Each reproduced binary run is differenced against the archived server result
for the same dataset and seed, across all four ablation variants and five
metrics. The command prints the largest movement per run and exits non-zero
when anything falls outside the tolerance, so a reviewer sees the size and
location of a difference rather than a bare pass or fail.

## Resource expectations

The reported runs were produced on a single CUDA GPU with Python 3.10. A full
four-benchmark, three-seed reproduction is measured in GPU-hours, dominated by
class-conditional DDPM training. CPU execution is supported and is intended for
wiring checks rather than for reproducing published numbers.

Reported figures come from three seeds. As stated in the manuscript, that is a
directional sample rather than a basis for significance claims, so small
differences between a rerun and the archive are expected.
