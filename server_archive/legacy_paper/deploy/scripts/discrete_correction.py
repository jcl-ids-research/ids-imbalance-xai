"""
Rank-matched marginal correction for discrete attributes in DDPM output.

Problem
-------
The generator runs in QuantileTransformer(normal) space. Sparse binary /
low-cardinality attributes -- exactly the ones that carry attack semantics
(root_shell, num_failed_logins, land, SYN Flag Count) -- have >99% of their
mass on one value. After the continuous forward/reverse diffusion and the
inverse transform, the generated values collapse onto that modal value, so the
attribute becomes a constant and loses all discriminative content.

The KS test does not detect this: with 99%+ mass on the mode, losing the rare
values moves the empirical CDF by <0.01, giving KS ~ 0.001 and p ~ 1.0.

Correction
----------
Copula-style marginal correction. For each column flagged as discrete in the
REAL TRAINING data:

    1. rank the generated values                 r_i in {0 .. n-1}
    2. convert to a quantile                     q_i = (r_i + 0.5) / n
    3. read the real empirical quantile function x_i = F_real^{-1}(q_i)

This reproduces the real marginal exactly (up to sampling granularity) while
keeping the rank ordering the diffusion model learned, so Spearman dependence
with the continuous columns is preserved.

Only the real TRAINING split is ever touched -- no test data is used anywhere.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MAX_UNIQUE = 25


def detect_discrete_columns(X_real: np.ndarray,
                            max_unique: int = DEFAULT_MAX_UNIQUE) -> np.ndarray:
    """Flag columns that behave as discrete in the real training data."""
    n_cols = X_real.shape[1]
    flags = np.zeros(n_cols, dtype=bool)
    for j in range(n_cols):
        col = X_real[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        if np.unique(col).size <= max_unique:
            flags[j] = True
    return flags


def rank_match_column(real_col: np.ndarray, syn_col: np.ndarray) -> np.ndarray:
    """Map generated values onto the real empirical marginal, preserving rank."""
    n = syn_col.size
    if n == 0:
        return syn_col

    real_sorted = np.sort(real_col[np.isfinite(real_col)])
    if real_sorted.size == 0:
        return syn_col

    # ranks of the generated values (ties broken deterministically)
    order = np.argsort(syn_col, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)

    q = (ranks + 0.5) / n
    idx = np.clip((q * real_sorted.size).astype(np.int64), 0, real_sorted.size - 1)
    return real_sorted[idx]


def correct_discrete(X_real: np.ndarray, X_syn: np.ndarray,
                     discrete_flags: np.ndarray | None = None,
                     max_unique: int = DEFAULT_MAX_UNIQUE
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the correction to every discrete column. Returns (corrected, flags)."""
    if X_syn.size == 0:
        return X_syn, np.zeros(X_real.shape[1], dtype=bool)

    if discrete_flags is None:
        discrete_flags = detect_discrete_columns(X_real, max_unique)

    out = X_syn.copy()
    for j in np.flatnonzero(discrete_flags):
        out[:, j] = rank_match_column(X_real[:, j], X_syn[:, j])
    return out, discrete_flags


def correct_discrete_per_class(X_real: np.ndarray, y_real: np.ndarray,
                               X_syn: np.ndarray, y_syn: np.ndarray,
                               max_unique: int = DEFAULT_MAX_UNIQUE
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Class-conditional correction.

    Attack semantics are class-specific: root_shell is near-always 0 for Normal
    traffic but carries signal for U2R. Matching the marginal of the matching
    class keeps that distinction instead of averaging it away.
    """
    flags = detect_discrete_columns(X_real, max_unique)
    out = X_syn.copy()
    if X_syn.size == 0:
        return out, flags

    for c in np.unique(y_syn):
        smask = y_syn == c
        rmask = y_real == c
        if smask.sum() == 0:
            continue
        # fall back to the pooled marginal when a class is too thin to estimate
        src = X_real[rmask] if rmask.sum() >= 20 else X_real
        for j in np.flatnonzero(flags):
            out[smask, j] = rank_match_column(src[:, j], X_syn[smask, j])
    return out, flags
