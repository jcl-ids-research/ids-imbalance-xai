"""Diagnose which features make QuantileTransformer.inverse_transform non-finite."""

from __future__ import annotations

import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, "/opt/ids_revision/deploy")
from correct_cross_dataset import DATASETS  # noqa: E402


def diagnose(dataset: str, seed: int = 42) -> None:
    print(f"===== {dataset} seed={seed} =====")
    data = DATASETS[dataset](seed, False)
    X_sub, _, y_sub, _ = train_test_split(
        data.X_train, data.y_train, test_size=0.1,
        random_state=seed, stratify=data.y_train,
    )
    print(f"X_sub finite: {np.isfinite(X_sub).all()}  shape={X_sub.shape}")

    qt = QuantileTransformer(output_distribution="normal", random_state=seed,
                             n_quantiles=min(1000, len(X_sub)))
    qt.fit(X_sub)
    q = qt.quantiles_
    print(f"quantiles_ shape={q.shape} finite={np.isfinite(q).all()}")
    if not np.isfinite(q).all():
        bad_cols = np.where(~np.isfinite(q).all(axis=0))[0]
        print(f"  non-finite quantile columns: {bad_cols.tolist()}")

    rng = np.random.default_rng(0)
    for cls in np.unique(y_sub):
        n = int((y_sub == cls).sum())
        probe = np.clip(rng.standard_normal((2000, X_sub.shape[1])), -4.0, 4.0)
        out = qt.inverse_transform(probe)
        finite = np.isfinite(out)
        if finite.all():
            print(f"  class {cls} (n={n}): probe inverse_transform all finite")
            continue
        per_col = (~finite).sum(axis=0)
        bad = np.where(per_col > 0)[0]
        print(f"  class {cls} (n={n}): {(~finite).sum()} non-finite in probe")
        for c in bad:
            name = data.feature_names[c] if c < len(data.feature_names) else f"col{c}"
            col = X_sub[:, c]
            print(f"    col {c} ({name}): bad={per_col[c]} "
                  f"real_min={col.min():.4g} real_max={col.max():.4g} "
                  f"std={col.std():.4g} n_unique={len(np.unique(col))} "
                  f"qmin={q[:, c].min():.4g} qmax={q[:, c].max():.4g}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["nslkdd"]:
        diagnose(ds)
