"""
Quantify the discrete-feature collapse that the KS test hides.

E10 asks whether generated numerical attributes combined with categorical ones
remain semantically valid. Sparse discrete features (flag counts, TTL codes,
protocol states) are the categorical-like attributes in these datasets. If the
DDPM smooths them to a single value, KS still passes -- because almost all the
mass sits on that value -- while the feature has actually lost its meaning.

Reported per feature:
  real_nonzero_rate / syn_nonzero_rate   coverage of non-default values
  real_n_unique / syn_n_unique           discreteness retained?
  coverage_ratio                         syn distinct values / real distinct values
  ks / ks_p                              what the KS test says (for contrast)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DISCRETE_MAX_UNIQUE = 25  # heuristic: <= this many distinct values -> discrete-like


def analyse(path: Path, top: int = 12) -> dict:
    d = np.load(path, allow_pickle=True)
    real = np.nan_to_num(d["real"].astype(np.float64), posinf=0, neginf=0)
    syn = np.nan_to_num(d["synthetic"].astype(np.float64), posinf=0, neginf=0)
    ry, sy = d["real_labels"], d["synthetic_labels"]
    names = [str(x) for x in d["feature_names"]]

    if len(syn) == 0:
        return {"error": "no synthetic"}

    # class-matched comparison
    classes = np.unique(sy)
    mask = np.isin(ry, classes)
    if mask.sum() > 50:
        real = real[mask]

    rows = []
    for i, nm in enumerate(names):
        r, s = real[:, i], syn[:, i]
        ru, su = np.unique(r), np.unique(s)
        # mode of the real feature = its "default" value
        vals, cnts = np.unique(r, return_counts=True)
        default = float(vals[np.argmax(cnts)])
        r_off = float((r != default).mean())
        s_off = float((s != default).mean())
        ks, p = stats.ks_2samp(r, s)
        rows.append({
            "feature": nm,
            "real_n_unique": int(len(ru)),
            "syn_n_unique": int(len(su)),
            "unique_ratio": float(len(su) / len(ru)) if len(ru) else None,
            "default_value": default,
            "real_off_default_rate": r_off,
            "syn_off_default_rate": s_off,
            "off_default_recall": float(s_off / r_off) if r_off > 0 else None,
            "ks": float(ks),
            "ks_p": float(p),
            "ks_passes": bool(p > 0.05),
            "discrete_like": bool(len(ru) <= DISCRETE_MAX_UNIQUE),
        })

    disc = [r for r in rows if r["discrete_like"]]
    # "Hidden failure": KS says the feature is fine, yet the synthetic data
    # recovers less than half of the real off-default mass.
    hidden = [
        r for r in disc
        if r["ks_passes"]
        and r["real_off_default_rate"] > 0
        and r["off_default_recall"] is not None
        and r["off_default_recall"] < 0.5
    ]

    return {
        "n_features": len(rows),
        "n_discrete_like": len(disc),
        "n_ks_pass": sum(1 for r in rows if r["ks_passes"]),
        "n_discrete_ks_pass": sum(1 for r in disc if r["ks_passes"]),
        "n_hidden_failures": len(hidden),
        "hidden_failures": sorted(hidden,
                                  key=lambda r: r["off_default_recall"])[:top],
        "all_features": rows,
    }


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
    allres = {}

    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        p = root / ds / f"seed{args.seed}" / "samples.npz"
        if not p.exists():
            print(f"[{ds}] samples.npz missing")
            continue
        r = analyse(p)
        allres[ds] = r
        if "error" in r:
            print(f"[{ds}] {r['error']}")
            continue

        print("=" * 78)
        print(f"[{ds}]  features={r['n_features']}  "
              f"discrete-like={r['n_discrete_like']}  "
              f"KS pass={r['n_ks_pass']}  "
              f"discrete KS pass={r['n_discrete_ks_pass']}")
        print(f"  KS 通过但离散取值实际丢失的特征: {r['n_hidden_failures']} 个")
        print("-" * 78)
        if r["hidden_failures"]:
            print(f"  {'feature':<28}{'uniq r/s':>12}{'off-default r/s':>20}"
                  f"{'recall':>9}{'KS':>8}{'p':>9}")
            for h in r["hidden_failures"]:
                print(f"  {h['feature']:<28}"
                      f"{h['real_n_unique']:>5}/{h['syn_n_unique']:<6}"
                      f"{h['real_off_default_rate'] * 100:>9.2f}%/"
                      f"{h['syn_off_default_rate'] * 100:>7.2f}%"
                      f"{h['off_default_recall'] * 100:>8.1f}%"
                      f"{h['ks']:>8.4f}{h['ks_p']:>9.3g}")
        print()

    out = outdir / f"discrete_collapse_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(allres, f, indent=2)
    print(f"[OK] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
