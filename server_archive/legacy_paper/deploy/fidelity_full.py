"""
Synthetic-sample fidelity evaluation.

Answers the two reviewer comments that named specific tests:

  R1-3  "no rigorous statistical validation (e.g., MMD, KL divergence, or
         distribution similarity tests) is provided to verify sample fidelity"
  R2-3  "may introduce unrealistic feature combinations ... no quantitative
         validation demonstrating that the generated samples preserve the
         original joint feature distribution or attack semantics"

Tests implemented
  1. KS two-sample test per feature        -> marginal distribution match
  2. Wasserstein distance per feature      -> marginal distance magnitude
  3. KL divergence per feature (binned)    -> named by R1-3
  4. MMD with RBF kernel (subsampled)      -> named by R1-3, joint distribution
  5. Correlation-matrix difference         -> joint structure (R2-3)
  6. Per-class MMD / KS                    -> attack semantics preserved (R2-3)

MMD uses a capped subsample because the RBF kernel matrix is O(n^2);
n=2000 keeps it at ~32 MB while remaining statistically adequate. The
subsample is repeated N_MMD_REPEATS times and averaged.

CPU only. No GPU. Does not touch running training jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

MMD_SUBSAMPLE = 2000
N_MMD_REPEATS = 5
KL_BINS = 50
RNG_SEED = 0


# ---------------------------------------------------------------- helpers


def ks_per_feature(real: np.ndarray, syn: np.ndarray) -> dict:
    stats_, pvals = [], []
    for i in range(real.shape[1]):
        s, p = stats.ks_2samp(real[:, i], syn[:, i])
        stats_.append(float(s))
        pvals.append(float(p))
    stats_ = np.array(stats_)
    pvals = np.array(pvals)
    return {
        "n_features": int(real.shape[1]),
        "pass_count": int((pvals > 0.05).sum()),
        "pass_rate": float((pvals > 0.05).mean()),
        "mean_ks": float(stats_.mean()),
        "median_ks": float(np.median(stats_)),
        "max_ks": float(stats_.max()),
        "per_feature_ks": stats_.tolist(),
        "per_feature_p": pvals.tolist(),
    }


def wasserstein_per_feature(real: np.ndarray, syn: np.ndarray) -> dict:
    d = [float(stats.wasserstein_distance(real[:, i], syn[:, i]))
         for i in range(real.shape[1])]
    d = np.array(d)
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "max": float(d.max()),
        "per_feature": d.tolist(),
    }


def kl_per_feature(real: np.ndarray, syn: np.ndarray, bins: int = KL_BINS) -> dict:
    """Symmetric-safe KL(real || synthetic) on a shared binning, with smoothing."""
    kls = []
    for i in range(real.shape[1]):
        lo = min(real[:, i].min(), syn[:, i].min())
        hi = max(real[:, i].max(), syn[:, i].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            kls.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(real[:, i], bins=edges)
        q, _ = np.histogram(syn[:, i], bins=edges)
        # Laplace smoothing so zero bins do not produce infinities
        p = (p + 1e-10) / (p.sum() + 1e-10 * bins)
        q = (q + 1e-10) / (q.sum() + 1e-10 * bins)
        kls.append(float(np.sum(p * np.log(p / q))))
    kls = np.array(kls)
    return {
        "mean": float(kls.mean()),
        "median": float(np.median(kls)),
        "max": float(kls.max()),
        "per_feature": kls.tolist(),
    }


def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD^2 estimate with an RBF kernel."""
    def k(A, B):
        d2 = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
              - 2.0 * A @ B.T)
        np.maximum(d2, 0, out=d2)
        return np.exp(-gamma * d2)

    n, m = len(X), len(Y)
    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1))
                 + Kyy.sum() / (m * (m - 1))
                 - 2.0 * Kxy.mean())


def mmd_test(real: np.ndarray, syn: np.ndarray,
             subsample: int = MMD_SUBSAMPLE,
             repeats: int = N_MMD_REPEATS) -> dict:
    """RBF-kernel MMD with median-heuristic bandwidth, averaged over subsamples.

    A permutation null is built from the pooled sample so the reported value
    can be judged against chance rather than read as a bare number.
    """
    rng = np.random.default_rng(RNG_SEED)
    vals, nulls = [], []

    for _ in range(repeats):
        a = real[rng.choice(len(real), min(subsample, len(real)), replace=False)]
        b = syn[rng.choice(len(syn), min(subsample, len(syn)), replace=False)]

        # median heuristic on a small pilot sample
        pilot = np.vstack([a[:500], b[:500]])
        d2 = ((pilot[:, None, :] - pilot[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-12)

        vals.append(_rbf_mmd2(a, b, gamma))

        # permutation null: split the pooled sample at random
        pooled = np.vstack([a, b])
        idx = rng.permutation(len(pooled))
        h = len(pooled) // 2
        nulls.append(_rbf_mmd2(pooled[idx[:h]], pooled[idx[h:2 * h]], gamma))

    vals = np.array(vals)
    nulls = np.array(nulls)
    return {
        "mmd2_mean": float(vals.mean()),
        "mmd2_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "null_mean": float(nulls.mean()),
        "null_std": float(nulls.std(ddof=1)) if len(nulls) > 1 else 0.0,
        "ratio_to_null": float(vals.mean() / nulls.mean()) if nulls.mean() > 0 else None,
        "subsample": int(min(subsample, len(real), len(syn))),
        "repeats": int(repeats),
    }


def correlation_diff(real: np.ndarray, syn: np.ndarray) -> dict:
    """Joint-structure check: how far do pairwise correlations drift? (R2-3)"""
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.corrcoef(real, rowvar=False)
        cs = np.corrcoef(syn, rowvar=False)
    cr = np.nan_to_num(cr)
    cs = np.nan_to_num(cs)
    iu = np.triu_indices_from(cr, k=1)
    diff = np.abs(cr[iu] - cs[iu])
    return {
        "n_pairs": int(len(diff)),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
        "max_abs_diff": float(diff.max()),
        "within_0.05": float((diff <= 0.05).mean()),
        "within_0.10": float((diff <= 0.10).mean()),
        "frobenius": float(np.linalg.norm(cr - cs)),
    }


# ---------------------------------------------------------------- driver


def evaluate(path: Path) -> dict | None:
    d = np.load(path, allow_pickle=True)
    real = d["real"].astype(np.float64)
    syn = d["synthetic"].astype(np.float64)
    real_y = d["real_labels"]
    syn_y = d["synthetic_labels"]
    names = [str(x) for x in d["feature_names"]]

    if len(syn) == 0:
        return {"error": "no synthetic samples", "n_real": int(len(real))}

    # Guard against non-finite values leaking into the tests.
    real = np.nan_to_num(real, posinf=0.0, neginf=0.0)
    syn = np.nan_to_num(syn, posinf=0.0, neginf=0.0)

    res: dict = {
        "n_real": int(len(real)),
        "n_synthetic": int(len(syn)),
        "n_features": int(real.shape[1]),
        "expansion_ratio": float(len(syn) / len(real)),
        "feature_names": names,
        "ks": ks_per_feature(real, syn),
        "wasserstein": wasserstein_per_feature(real, syn),
        "kl_divergence": kl_per_feature(real, syn),
        "mmd_rbf": mmd_test(real, syn),
        "correlation": correlation_diff(real, syn),
    }

    # R2-3: attack semantics -> repeat the joint test within each class
    per_class = {}
    for c in np.unique(syn_y):
        rmask = real_y == c
        smask = syn_y == c
        if rmask.sum() < 50 or smask.sum() < 50:
            continue
        per_class[str(int(c))] = {
            "n_real": int(rmask.sum()),
            "n_synthetic": int(smask.sum()),
            "ks_pass_rate": ks_per_feature(real[rmask], syn[smask])["pass_rate"],
            "mmd_rbf": mmd_test(real[rmask], syn[smask], repeats=3),
            "correlation": correlation_diff(real[rmask], syn[smask]),
        }
    res["per_class"] = per_class
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="results/fidelity")
    ap.add_argument("--datasets", default="unsw,nslkdd,cicids2017,cicddos2019")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        p = root / ds / f"seed{args.seed}" / "samples.npz"
        if not p.exists():
            print(f"[{ds}] samples.npz not found: {p}")
            continue
        print(f"[{ds}] evaluating {p} ...", flush=True)
        r = evaluate(p)
        if r is None:
            continue
        r["source"] = str(p)
        all_res[ds] = r

        if "error" in r:
            print(f"    {r['error']}")
            continue
        print(f"    real={r['n_real']} syn={r['n_synthetic']} "
              f"expansion={r['expansion_ratio']:.2f}x features={r['n_features']}")
        print(f"    KS   pass {r['ks']['pass_count']}/{r['ks']['n_features']} "
              f"({r['ks']['pass_rate'] * 100:.1f}%)  mean={r['ks']['mean_ks']:.4f}  "
              f"max={r['ks']['max_ks']:.4f}")
        print(f"    KL   mean={r['kl_divergence']['mean']:.4f}  "
              f"max={r['kl_divergence']['max']:.4f}")
        print(f"    W1   mean={r['wasserstein']['mean']:.4f}")
        m = r["mmd_rbf"]
        ratio = m["ratio_to_null"]
        ratio_txt = f"{ratio:.1f}x" if ratio is not None else "n/a"
        print(f"    MMD2 {m['mmd2_mean']:.6f}+/-{m['mmd2_std']:.6f}  "
              f"null={m['null_mean']:.6f}  ratio={ratio_txt}")
        c = r["correlation"]
        print(f"    CORR mean|d|={c['mean_abs_diff']:.4f}  "
              f"<=0.10 {c['within_0.10'] * 100:.1f}%  max={c['max_abs_diff']:.4f}")
        print()

    out = outdir / f"fidelity_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"[OK] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
