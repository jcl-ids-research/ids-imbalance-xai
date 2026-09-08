"""Cross-dataset ablation under the corrected protocol.

Tests the paper's generalization claim ("diffusion improves cross-dataset
stability") on NSL-KDD, CIC-IDS-2017 and CIC-DDoS2019 using the same corrected
machinery validated on UNSW-NB15:

  - NSL-KDD keeps its official KDDTrain+/KDDTest+ split (no merging).
  - CIC datasets have no official split, so a stratified split is created once
    per seed; every encoder / imputer / scaler is still fit on TRAIN ONLY.
  - Class-conditional DDPM (quantile space, T=500) trained on subtrain only.
  - Paired initialization; epoch-wise shuffling.
  - Multi-view uses equal positional splits (these datasets have no published
    semantic view mapping); the diffusion contrast is unaffected because both
    members of each pair share the same partition.

Run:
  /usr/bin/python3 correct_cross_dataset.py --dataset nslkdd --seed 42
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler


def predict(model, X: "np.ndarray", device, batch: int):
    """Forward pass returning (hard predictions, softmax probabilities).

    Mirrors run_instrumented.py:predict so cross-dataset runs emit the same
    artifacts as the UNSW run. Eval mode is deterministic, so calling this in
    addition to _evaluate does not change any reported metric.
    """
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.FloatTensor(X[i:i + batch]).to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correct_unsw_ablation import (  # noqa: E402
    AUGMENT_FRAC,
    CLASSIFIER_BATCH,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_PATIENCE,
    DDPM_EPOCHS,
    Classifier,
    LoadedData,
    MLPDDPM,
    MultiViewEncoder,
    SingleViewEncoder,
    _evaluate,
    _train_classifier,
    rebalance,
    train_ddpm,
)

DEFAULT_OUT_DIR = "/opt/ids_revision/results/ablation_cross_correct"
CIC_MAX_TOTAL = 200_000
# CIC CSVs are time-ordered and attacks occupy specific windows, so a small
# head-read returns almost pure BENIGN. Read enough rows per file to cover the
# attack windows, then stratified-subsample down to CIC_MAX_TOTAL.
CIC_ROWS_PER_FILE = 50_000
CIC_READ_CAP = 500_000
N_VIEWS = 3

NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CIC_DROP = {
    "unnamed: 0", "flow id", "source ip", "src ip", "source port", "src port",
    "destination ip", "dst ip", "destination port", "dst port", "protocol",
    "timestamp", "simillarhttp", "inbound",
}


def positional_view_splits(n_features: int, n_views: int = N_VIEWS) -> list[list[int]]:
    base, rem = divmod(n_features, n_views)
    dims = [base + (1 if i < rem else 0) for i in range(n_views)]
    splits: list[list[int]] = []
    start = 0
    for d in dims:
        splits.append(list(range(start, start + d)))
        start += d
    return splits


def _encode_and_scale(tr: pd.DataFrame, te: pd.DataFrame,
                      feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit categorical maps, medians and the scaler on TRAIN only.

    Zero-variance columns are dropped: they carry no signal and make the
    quantile transform degenerate, which emits NaN synthetic samples.
    """
    tr = tr.copy()
    te = te.copy()
    for c in feat_cols:
        if tr[c].dtype == object:
            cats = sorted(tr[c].astype(str).unique())
            mapping = {v: i for i, v in enumerate(cats)}
            tr[c] = tr[c].astype(str).map(mapping).astype(np.float32)
            te[c] = te[c].astype(str).map(mapping).fillna(-1.0).astype(np.float32)

    X_tr = tr[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_te = te[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)
    medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(medians).fillna(0.0).astype(np.float32)
    X_te = X_te.fillna(medians).fillna(0.0).astype(np.float32)

    keep = [c for c in feat_cols if float(X_tr[c].std()) > 1e-8]
    dropped = len(feat_cols) - len(keep)
    if dropped:
        print(f"  [PREP] dropped {dropped} zero-variance features", flush=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr[keep].to_numpy())
    X_te_s = scaler.transform(X_te[keep].to_numpy())
    return X_tr_s, X_te_s, keep


def load_nsl_kdd(data_dir: str = "/opt/NSL-KDD", smoke: bool = False) -> LoadedData:
    tr = pd.read_csv(os.path.join(data_dir, "KDDTrain+.txt"), names=NSL_COLS)
    te = pd.read_csv(os.path.join(data_dir, "KDDTest+.txt"), names=NSL_COLS)
    if smoke:
        tr = tr.sample(n=4000, random_state=0)
        te = te.sample(n=2000, random_state=0)

    y_tr = (tr["label"].astype(str) != "normal").astype(np.int64).to_numpy()
    y_te = (te["label"].astype(str) != "normal").astype(np.int64).to_numpy()
    feat_cols = [c for c in NSL_COLS if c not in ("label", "difficulty")]
    X_tr, X_te, kept = _encode_and_scale(tr, te, feat_cols)
    return LoadedData(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                      n_features=X_tr.shape[1], feature_names=tuple(kept))


def load_cic(path: str, seed: int, smoke: bool = False) -> LoadedData:
    files = sorted(f for f in glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")

    rows_cap = 4000 if smoke else CIC_ROWS_PER_FILE
    read_cap = 12_000 if smoke else CIC_READ_CAP
    keep_total = 6000 if smoke else CIC_MAX_TOTAL
    pieces: list[pd.DataFrame] = []
    total = 0
    for f in files:
        df = pd.read_csv(f, nrows=rows_cap, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            continue
        df = df.rename(columns={label_col: "Label"})
        pieces.append(df)
        total += len(df)
        if total >= read_cap:
            break
    df_all = pd.concat(pieces, ignore_index=True)

    y_full = np.array([0 if str(v).strip().upper() == "BENIGN" else 1
                       for v in df_all["Label"]], dtype=np.int64)
    if len(df_all) > keep_total and len(np.unique(y_full)) > 1:
        keep_idx, _ = train_test_split(
            np.arange(len(df_all)), train_size=keep_total,
            random_state=seed, stratify=y_full,
        )
        df_all = df_all.iloc[keep_idx].reset_index(drop=True)
        y_full = y_full[keep_idx]
    y = y_full
    feat_cols = [c for c in df_all.columns
                 if c != "Label" and c.lower() not in CIC_DROP]

    # No official split exists for CIC; create a stratified one, then fit all
    # preprocessing on the training partition only.
    idx_tr, idx_te = train_test_split(
        np.arange(len(df_all)), test_size=0.2, random_state=seed, stratify=y
    )
    tr = df_all.iloc[idx_tr].reset_index(drop=True)
    te = df_all.iloc[idx_te].reset_index(drop=True)
    X_tr, X_te, kept = _encode_and_scale(tr, te, feat_cols)
    return LoadedData(X_train=X_tr, y_train=y[idx_tr], X_test=X_te, y_test=y[idx_te],
                      n_features=X_tr.shape[1], feature_names=tuple(kept))


DATASETS = {
    "nslkdd": lambda seed, smoke: load_nsl_kdd(smoke=smoke),
    "cicids2017": lambda seed, smoke: load_cic("/opt/CIC-IDS-2017/MachineLearningCVE", seed, smoke),
    "cicddos2019": lambda seed, smoke: load_cic("/opt/CIC-DDoS2019/all", seed, smoke),
}


def run_seed(dataset: str, seed: int, out_dir: str, smoke: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DATASETS[dataset](seed, smoke)
    print(f"  [DATA] {dataset}: train={len(data.X_train)} test={len(data.X_test)} "
          f"features={data.n_features} attack_rate={data.y_train.mean():.3f}", flush=True)

    X_sub, X_val, y_sub, y_val = train_test_split(
        data.X_train, data.y_train, test_size=0.1, random_state=seed, stratify=data.y_train
    )

    ddpm_epochs = 2 if smoke else DDPM_EPOCHS
    cls_epochs = 2 if smoke else CLASSIFIER_EPOCHS
    cls_batch = 128 if smoke else CLASSIFIER_BATCH

    qt = QuantileTransformer(output_distribution="normal", random_state=seed,
                             n_quantiles=min(1000, len(X_sub)))
    X_sub_q = qt.fit_transform(X_sub)

    ddpm_models: dict[int, MLPDDPM] = {}
    for cls in np.unique(y_sub):
        m = MLPDDPM(data.n_features)
        train_ddpm(m, X_sub_q[y_sub == cls], device, ddpm_epochs, label=str(int(cls)))
        ddpm_models[int(cls)] = m

    # Rebalance training set with 15x cap
    X_bal, y_bal, balance_report, X_syn, y_syn = rebalance(
        ddpm_models, X_sub, y_sub, qt, device, seed
    )

    view_splits = positional_view_splits(data.n_features)
    print(f"  [VIEWS] sizes={[len(s) for s in view_splits]}", flush=True)
    print(f"  [BALANCE] {balance_report['imbalance_ratio_before']:.2f} -> "
          f"{balance_report['imbalance_ratio_after']:.2f}, "
          f"synthetic={balance_report['synthetic_total']}", flush=True)

    # Artifact directory mirrors run_instrumented.py layout.
    art_dir = os.path.join(out_dir, dataset, f"seed{seed}")
    os.makedirs(art_dir, exist_ok=True)

    # R1-3 / R2-3: real + synthetic side by side for fidelity testing
    # (KS / MMD / KL / correlation). This is the evidence the reviewers asked
    # for, and it matters most here because expansion ratios on the CIC sets
    # are far higher than on UNSW.
    n_keep = min(20000, len(X_sub))
    np.savez_compressed(
        os.path.join(art_dir, "samples.npz"),
        real=X_sub[:n_keep].astype(np.float32),
        real_labels=y_sub[:n_keep].astype(np.int64),
        synthetic=X_syn[:n_keep].astype(np.float32) if len(X_syn) > 0
                  else np.empty((0, data.n_features), dtype=np.float32),
        synthetic_labels=y_syn[:n_keep].astype(np.int64) if len(y_syn) > 0
                         else np.empty(0, dtype=np.int64),
        feature_names=np.array(data.feature_names, dtype=object),
    )
    print(f"  [SAVE] samples.npz real={n_keep} synthetic={len(X_syn)}", flush=True)

    full_model = Classifier(MultiViewEncoder(data.n_features, view_splits))
    wo_diff = copy.deepcopy(full_model)
    wo_mv = Classifier(SingleViewEncoder(data.n_features))
    wo_both = copy.deepcopy(wo_mv)

    # Training order preserved exactly as before so results stay comparable.
    specs = [
        ("full_model", full_model, X_bal, y_bal, True),
        ("wo_diffusion", wo_diff, X_sub, y_sub, True),
        ("wo_multiview", wo_mv, X_bal, y_bal, False),
        ("baseline", wo_both, X_sub, y_sub, False),
    ]

    variants = {}
    timings = {}
    for name, model, Xtr, ytr, is_mv in specs:
        t1 = time.time()
        _train_classifier(model, Xtr, ytr, X_val, y_val, device,
                          cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
        timings[name] = time.time() - t1

        # Metrics come from the untouched _evaluate path.
        variants[name] = _evaluate(model, data.X_test, data.y_test, device, cls_batch)

        # Extra deterministic pass purely to persist predictions/probabilities.
        y_pred, y_prob = predict(model, data.X_test, device, cls_batch)
        np.savez_compressed(
            os.path.join(art_dir, f"predictions_{name}.npz"),
            y_true=data.y_test.astype(np.int64),
            y_pred=y_pred.astype(np.int64),
            y_prob=y_prob.astype(np.float32),
        )
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
             "variant": name,
             "input_dim": int(data.n_features),
             "view_splits": view_splits,
             "multiview": is_mv},
            os.path.join(art_dir, f"model_{name}.pt"),
        )
        print(f"  [{name}] F1={variants[name].f1 * 100:.2f} "
              f"acc={variants[name].accuracy * 100:.2f} "
              f"({timings[name] / 60:.1f} min)", flush=True)

    result = {
        "dataset": dataset,
        "seed": seed,
        "protocol": "train-only-preprocessing + conditional-DDPM + mean-target-rebalance + paired-init + positional-views",
        "official_split": dataset == "nslkdd",
        "balance_report": balance_report,
        "sizes": {
            "train": int(len(X_sub)), "val": int(len(X_val)),
            "test": int(len(data.X_test)), "balanced_train": int(len(X_bal)),
            "features": int(data.n_features),
        },
        "variants": {k: v.as_dict() for k, v in variants.items()},
        "timings_sec": timings,
        "view_partition_type": "positional",
        "artifact_dir": art_dir,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] saved {out_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_seed(args.dataset, args.seed, args.out_dir, args.smoke)
    print(json.dumps(result["variants"], indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
