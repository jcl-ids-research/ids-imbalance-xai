"""
Offline validation of the discrete correction on existing samples.npz.

Three questions, all must pass before we agree to re-run 12 jobs:

  1. off-default recall  must go UP    (the failure we are fixing)
  2. MMD^2               must NOT go up materially (joint structure intact)
  3. correlation match   must NOT go down materially

Runs on saved artifacts only. No training, no GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from discrete_correction import correct_discrete_per_class, detect_discrete_columns

MMD_SUB = 2000
MMD_REP = 5
RNG = 0


def _rbf_mmd2(X, Y, gamma):
    def k(A, B):
        d2 = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2.0 * A @ B.T)
        np.maximum(d2, 0, out=d2)
        return np.exp(-gamma * d2)
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean())


def mmd2(real, syn, sub=MMD_SUB, rep=MMD_REP):
    rng = np.random.default_rng(RNG)
    vals = []
    for _ in range(rep):
        a = real[rng.choice(len(real), min(sub, len(real)), replace=False)]
        b = syn[rng.choice(len(syn), min(sub, len(syn)), replace=False)]
        pilot = np.vstack([a[:500], b[:500]])
        d2 = ((pilot[:, None, :] - pilot[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        vals.append(_rbf_mmd2(a, b, 1.0 / max(med, 1e-12)))
    return float(np.mean(vals))


def corr_within(real, syn, thr=0.10):
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.nan_to_num(np.corrcoef(real, rowvar=False))
        cs = np.nan_to_num(np.corrcoef(syn, rowvar=False))
    iu = np.triu_indices_from(cr, k=1)
    d = np.abs(cr[iu] - cs[iu])
    return float((d <= thr).mean()), float(d.mean())


def ks_pass_rate(real, syn):
    ps = [stats.ks_2samp(real[:, i], syn[:, i])[1] for i in range(real.shape[1])]
    return float(np.mean([p > 0.05 for p in ps]))


def off_default_stats(real, syn, flags):
    """Mean recall of off-modal mass across discrete columns."""
    recalls, uniq_ratio = [], []
    for j in np.flatnonzero(flags):
        r, s = real[:, j], syn[:, j]
        vals, cnts = np.unique(r, return_counts=True)
        mode = vals[np.argmax(cnts)]
        r_off = float((r != mode).mean())
        s_off = float((s != mode).mean())
        if r_off > 0:
            recalls.append(min(s_off / r_off, 2.0))
        ru, su = np.unique(r).size, np.unique(s).size
        uniq_ratio.append(su / ru if ru else 1.0)
    return (float(np.mean(recalls)) if recalls else None,
            float(np.mean(uniq_ratio)) if uniq_ratio else None,
            len(recalls))


def evaluate(path: Path) -> dict | None:
    d = np.load(path, allow_pickle=True)
    real = np.nan_to_num(d["real"].astype(np.float64), posinf=0, neginf=0)
    syn = np.nan_to_num(d["synthetic"].astype(np.float64), posinf=0, neginf=0)
    ry, sy = d["real_labels"], d["synthetic_labels"]
    if len(syn) == 0:
        return None

    fixed, flags = correct_discrete_per_class(real, ry, syn, sy)

    classes = np.unique(sy)
    m = np.isin(ry, classes)
    real_cm = real[m] if m.sum() > 50 else real

    out = {"n_features": int(real.shape[1]), "n_discrete": int(flags.sum())}
    for tag, S in (("before", syn), ("after", fixed)):
        rec, uratio, ndisc = off_default_stats(real_cm, S, flags)
        cw, cm_ = corr_within(real_cm, S)
        out[tag] = {
            "off_default_recall": rec,
            "unique_ratio": uratio,
            "n_discrete_scored": ndisc,
            "ks_pass_rate": ks_pass_rate(real_cm, S),
            "mmd2": mmd2(real_cm, S),
            "corr_within_0.10": cw,
            "corr_mean_abs_diff": cm_,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="results/fidelity")
    ap.add_argument("--datasets", default="unsw,nslkdd,cicids2017,cicddos2019")
    args = ap.parse_args()

    root, outdir = Path(args.root), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    allres, verdicts = {}, []

    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        p = root / ds / f"seed{args.seed}" / "samples.npz"
        if not p.exists():
            print(f"[{ds}] samples.npz missing")
            continue
        print(f"[{ds}] evaluating ...", flush=True)
        r = evaluate(p)
        if r is None:
            print(f"[{ds}] no synthetic samples")
            continue
        allres[ds] = r
        b, a = r["before"], r["after"]

        print(f"  discrete columns: {r['n_discrete']} / {r['n_features']}"
              f"   (scored: {b['n_discrete_scored']})")
        print(f"  {'metric':<24}{'before':>12}{'after':>12}{'change':>12}")
        print("  " + "-" * 60)

        def row(name, key, higher_better, fmt="{:.4f}"):
            bv, av = b[key], a[key]
            if bv is None or av is None:
                print(f"  {name:<24}{'n/a':>12}{'n/a':>12}")
                return None
            delta = av - bv
            good = (delta >= -1e-9) if higher_better else (delta <= 1e-9)
            mark = "OK" if good else "WORSE"
            print(f"  {name:<24}{fmt.format(bv):>12}{fmt.format(av):>12}"
                  f"{delta:>+11.4f} {mark}")
            return good

        r1 = row("off-default recall", "off_default_recall", True)
        r2 = row("unique ratio", "unique_ratio", True)
        r3 = row("KS pass rate", "ks_pass_rate", True)
        r4 = row("MMD^2 (lower better)", "mmd2", False)
        r5 = row("corr within 0.10", "corr_within_0.10", True)

        # gate: recall must improve; MMD and correlation must not degrade badly
        mmd_ok = a["mmd2"] <= b["mmd2"] * 1.10 + 1e-6
        corr_ok = a["corr_within_0.10"] >= b["corr_within_0.10"] - 0.02
        rec_ok = (a["off_default_recall"] or 0) > (b["off_default_recall"] or 0)
        verdict = rec_ok and mmd_ok and corr_ok
        verdicts.append((ds, verdict, rec_ok, mmd_ok, corr_ok))
        print(f"  -> recall_up={rec_ok}  mmd_ok={mmd_ok}  corr_ok={corr_ok}   "
              f"{'PASS' if verdict else 'FAIL'}")
        print()

    print("=" * 74)
    print("裁决")
    print("=" * 74)
    for ds, v, rec, mm, cc in verdicts:
        print(f"  {ds:<14}{'PASS' if v else 'FAIL':<7}"
              f"recall_up={rec}  mmd_ok={mm}  corr_ok={cc}")
    allpass = bool(verdicts) and all(v for _, v, _, _, _ in verdicts)
    print()
    print(f"总体: {'ALL PASS - 可以接进管线重跑' if allpass else 'NOT ALL PASS - 不要重跑'}")

    with open(outdir / f"correction_validation_seed{args.seed}.json", "w") as f:
        json.dump(allres, f, indent=2)
    print(f"[OK] saved {outdir / f'correction_validation_seed{args.seed}.json'}")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
