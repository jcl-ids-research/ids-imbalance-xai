"""Multi-class loaders for UNSW-NB15 (10 classes) and NSL-KDD (5 classes).

The binary loaders in correct_unsw_ablation.py and correct_cross_dataset.py
read the `label` column. The paper also reports attack-type classification
(Table 3 and Table 7), which needs the category column instead.

Everything else is deliberately identical to the binary path - the official
split, preprocessing fitted on training data only, the same feature set, the
same scaler - so a multi-class number can be placed next to a binary number
without a caveat about differing protocols.

NSL-KDD ships attack names, not categories. The test set contains 17 attack
names absent from the training set, which is the property that makes the
benchmark hard. Those names still belong to the four standard categories, so
the mapping below covers train and test names alike and refuses to guess: an
unmapped name raises rather than being silently bucketed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from correct_cross_dataset import NSL_COLS, _encode_and_scale
from correct_unsw_ablation import LoadedData

# NSL-KDD attack name -> category. Sources: the KDD'99 task description plus
# the additional names introduced in the NSL-KDD test set.
NSL_CATEGORY = {
    # Denial of service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probing
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to local
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L", "worm": "R2L",
    # User to root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    # Benign
    "normal": "Normal",
}

NSL_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def _nsl_categories(labels: pd.Series) -> np.ndarray:
    names = labels.astype(str).str.strip().str.rstrip(".")
    unknown = sorted(set(names) - set(NSL_CATEGORY))
    if unknown:
        raise KeyError(f"unmapped NSL-KDD attack names: {unknown}")
    index = {c: i for i, c in enumerate(NSL_CLASSES)}
    return names.map(NSL_CATEGORY).map(index).to_numpy().astype(np.int64)


def load_nsl_kdd_multiclass(data_dir: str = "/opt/NSL-KDD",
                            smoke: bool = False) -> tuple[LoadedData, list[str]]:
    tr = pd.read_csv(os.path.join(data_dir, "KDDTrain+.txt"), names=NSL_COLS)
    te = pd.read_csv(os.path.join(data_dir, "KDDTest+.txt"), names=NSL_COLS)
    if smoke:
        tr = tr.sample(n=4000, random_state=0)
        te = te.sample(n=2000, random_state=0)

    y_tr = _nsl_categories(tr["label"])
    y_te = _nsl_categories(te["label"])
    feat_cols = [c for c in NSL_COLS if c not in ("label", "difficulty")]
    X_tr, X_te, kept = _encode_and_scale(tr, te, feat_cols)
    data = LoadedData(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                      n_features=X_tr.shape[1], feature_names=tuple(kept))
    return data, list(NSL_CLASSES)


def load_unsw_multiclass(data_dir: str = "/opt/UNSW-NB15",
                         smoke: bool = False) -> tuple[LoadedData, list[str]]:
    """Same body as load_unsw, except the target is attack_cat, not label."""
    tr = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_training-set.csv"))
    te = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_testing-set.csv"))
    if smoke:
        tr = tr.sample(n=4000, random_state=0)
        te = te.sample(n=2000, random_state=0)

    for df in (tr, te):
        df.drop(columns=[c for c in ["id", "Unnamed: 0"] if c in df.columns],
                inplace=True, errors="ignore")

    feat_cols = [c for c in tr.columns if c not in ("label", "attack_cat")]
    cat_cols = [c for c in feat_cols if tr[c].dtype == object]

    for c in cat_cols:
        cats = sorted(tr[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(cats)}
        tr[c] = tr[c].astype(str).map(mapping).astype(np.float32)
        te[c] = te[c].astype(str).map(mapping).fillna(-1.0).astype(np.float32)

    medians = tr[feat_cols].median(numeric_only=True)
    tr = tr.fillna(medians)
    te = te.fillna(medians)

    # Category names come from the training set only, matching how every other
    # encoding in this pipeline is fitted. A test-only category would signal a
    # split problem, so it raises instead of being folded into an "other" bin.
    names = sorted(tr["attack_cat"].astype(str).str.strip().unique())
    index = {v: i for i, v in enumerate(names)}
    unseen = sorted(set(te["attack_cat"].astype(str).str.strip()) - set(names))
    if unseen:
        raise KeyError(f"attack_cat present in test but not train: {unseen}")

    y_tr = tr["attack_cat"].astype(str).str.strip().map(index).to_numpy().astype(np.int64)
    y_te = te["attack_cat"].astype(str).str.strip().map(index).to_numpy().astype(np.int64)

    X_tr = tr[feat_cols].astype(np.float32).to_numpy()
    X_te = te[feat_cols].astype(np.float32).to_numpy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    data = LoadedData(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                      n_features=X_tr.shape[1], feature_names=tuple(feat_cols))
    return data, names


MULTICLASS_DATASETS = {
    "unsw": load_unsw_multiclass,
    "nslkdd": load_nsl_kdd_multiclass,
}
