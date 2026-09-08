"""Recompute the manuscript's headline numbers from the archived results.

A reviewer should be able to check the paper's claims without a GPU and without
waiting for a full rerun. Every value below is derived from the archived server
results committed under `evidence/paper_results`, so a mismatch means either the
archive or the manuscript moved.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ids_diffusion.errors import DatasetFileError

DEFAULT_EVIDENCE_ROOT = Path("evidence/paper_results")
PAPER_SEEDS: tuple[int, ...] = (42, 123, 456)
FGSM_EPSILON = "0.05"
MULTICLASS_DATASETS: tuple[str, ...] = ("unsw", "nslkdd")
BASELINE_FAMILIES: tuple[str, ...] = (
    "baseline_fair_multiclass",
    "baseline_deep_multiclass",
)
CORRELATION_KEY = "corr_within_0.10"
CORRECTION_TOLERANCE = 2.0


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """One manuscript number next to the value the archive reproduces."""

    name: str
    claimed: float
    archived: float
    tolerance: float

    @property
    def difference(self) -> float:
        """Return the signed gap between archive and manuscript."""
        return self.archived - self.claimed

    @property
    def holds(self) -> bool:
        """Report whether the archive still supports the printed number."""
        return abs(self.difference) <= self.tolerance


def _read(path: Path) -> dict:
    if not path.is_file():
        raise DatasetFileError(path=str(path), detail="archived result not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_paths(root: Path, family: str, dataset: str) -> Iterable[Path]:
    for seed in PAPER_SEEDS:
        flat = root / family / f"{dataset}_seed{seed}.json"
        yield flat if flat.is_file() else root / family / dataset / f"seed{seed}" / "metrics.json"


def variant_means(root: Path, family: str, dataset: str) -> dict[str, float]:
    """Return the three-seed mean macro F1 of every variant, in percent."""
    collected: dict[str, list[float]] = {}
    for path in _seed_paths(root, family, dataset):
        for name, metrics in _read(path)["variants"].items():
            score = metrics.get("f1_macro", metrics.get("f1"))
            collected.setdefault(name, []).append(float(score) * 100)
    return {name: statistics.mean(values) for name, values in collected.items()}


def variant_spread(root: Path, family: str, dataset: str) -> dict[str, float]:
    """Return the three-seed standard deviation of every variant, in percent."""
    collected: dict[str, list[float]] = {}
    for path in _seed_paths(root, family, dataset):
        for name, metrics in _read(path)["variants"].items():
            score = metrics.get("f1_macro", metrics.get("f1"))
            collected.setdefault(name, []).append(float(score) * 100)
    return {name: statistics.stdev(values) for name, values in collected.items()}


def fgsm_accuracy_drop(root: Path, dataset: str, variant: str) -> float:
    """Return the mean accuracy lost at the reported FGSM budget, in points."""
    drops: list[float] = []
    for seed in PAPER_SEEDS:
        payload = _read(root / "adversarial" / f"adversarial_{dataset}_seed{seed}.json")
        entry = payload["variants"][variant]
        clean = float(entry["clean"]["acc"])
        attacked = float(entry["fgsm"][FGSM_EPSILON]["acc"])
        drops.append((clean - attacked) * 100)
    return statistics.mean(drops)


def residual_imbalance(root: Path, dataset: str) -> tuple[float, int]:
    """Return the post-balancing ratio and the expansion cap that produced it."""
    payload = _read(root / "phase1_corrected" / f"{dataset}_seed42.json")
    report = payload["balance_report"]
    return float(report["imbalance_ratio_after"]), int(report["expansion_cap"])


def balancing_effect(root: Path) -> tuple[int, int, float]:
    """Return improved pairs, total pairs and the largest gain from balancing.

    This is the model-independent result: for each classical or deep baseline on
    each labelled benchmark, balancing the training set is compared with leaving
    it untouched.
    """
    gains: list[float] = []
    for dataset in MULTICLASS_DATASETS:
        for family in BASELINE_FAMILIES:
            means = variant_means(root, family, dataset)
            models = {name.split("__")[0] for name in means}
            for model in sorted(models):
                raw = means.get(f"{model}__raw")
                balanced = means.get(f"{model}__balanced")
                if raw is None or balanced is None:
                    continue
                gains.append(balanced - raw)
    improved = sum(1 for gain in gains if gain > 0)
    return improved, len(gains), max(gains)


def correction_cost(root: Path, dataset: str) -> tuple[float, float]:
    """Return the min and max correlation-preservation cost of the correction.

    The correction restores discrete marginals but perturbs feature
    correlations. The manuscript reports that cost as a range in percentage
    points and discloses where it breaches the pre-declared tolerance, so the
    range is recomputed here rather than trusted.
    """
    costs: list[float] = []
    for path in sorted((root / "fidelity").glob("correction_validation_seed*.json")):
        payload = _read(path)
        block = payload.get(dataset)
        if not isinstance(block, dict):
            continue
        before = block.get("before", {}).get(CORRELATION_KEY)
        after = block.get("after", {}).get(CORRELATION_KEY)
        if before is None or after is None:
            continue
        costs.append((float(before) - float(after)) * 100)
    if not costs:
        raise DatasetFileError(
            path=str(root / "fidelity"),
            detail=f"no correction validation recorded for {dataset}",
        )
    return min(costs), max(costs)


def restored_off_default_recall(root: Path) -> tuple[float, float]:
    """Return the lowest and highest post-correction off-default recall."""
    values: list[float] = []
    for path in sorted((root / "fidelity").glob("correction_validation_seed*.json")):
        for block in _read(path).values():
            if not isinstance(block, dict):
                continue
            recall = block.get("after", {}).get("off_default_recall")
            if recall is not None:
                values.append(float(recall) * 100)
    if not values:
        raise DatasetFileError(path=str(root / "fidelity"), detail="no recall recorded")
    return min(values), max(values)


def scarcity_recall(root: Path, variant: str, label: int) -> tuple[float, float]:
    """Return the mean and spread of one class's recall under training-side scarcity.

    The scarcity run thins UNSW-NB15 attack rows to five per cent of their
    original count and leaves the test partition as published, so the attack
    class is rare while the model learns and ordinary when it is scored. It is
    the one setting in which the proposed model leads a gradient-boosted
    baseline on the metric the augmentation targets, so the numbers behind that
    claim are recomputed here rather than trusted.
    """
    values: list[float] = []
    for seed in PAPER_SEEDS:
        payload = _read(root / "scarcity" / f"keep005_seed{seed}.json")
        entry = payload["variants"][variant]
        match = next((c for c in entry["per_class"] if c["label"] == label), None)
        if match is None:
            raise DatasetFileError(
                path=str(root / "scarcity"), detail=f"class {label} missing for {variant}"
            )
        values.append(float(match["recall"]) * 100)
    return statistics.mean(values), statistics.stdev(values)


def scarcity_macro_f1(root: Path, variant: str) -> float:
    """Return the three-seed mean macro F1 of one variant in the scarcity run."""
    scores = [
        float(
            _read(root / "scarcity" / f"keep005_seed{seed}.json")["variants"][variant]["macro_f1"]
        )
        * 100
        for seed in PAPER_SEEDS
    ]
    return statistics.mean(scores)


def scarcity_thinning(root: Path) -> tuple[int, int]:
    """Return the attack-row count before and after thinning."""
    report = _read(root / "scarcity" / "keep005_seed42.json")["shift_report"]
    return int(report["attacks_before"]), int(report["attacks_after"])


def evaluate_claims(root: Path = DEFAULT_EVIDENCE_ROOT) -> tuple[ClaimResult, ...]:
    """Recompute every headline number the manuscript reports."""
    binary = {
        dataset: variant_means(root, "phase1_corrected", dataset)
        for dataset in ("unsw", "nslkdd", "cicids2017", "cicddos2019")
    }
    unsw_spread = variant_spread(root, "phase1_corrected", "unsw")
    multiclass = variant_means(root, "multiclass", "unsw")
    fair = variant_means(root, "baseline_fair_multiclass", "unsw")
    ratio, cap = residual_imbalance(root, "cicddos2019")
    improved, total, largest = balancing_effect(root)
    unsw_cost = correction_cost(root, "unsw")
    nslkdd_cost = correction_cost(root, "nslkdd")
    scarce_attack_full = scarcity_recall(root, "full_model", 1)
    scarce_attack_plain = scarcity_recall(root, "without_diffusion", 1)
    scarce_attack_xgb = scarcity_recall(root, "xgboost_raw", 1)
    scarce_normal_full = scarcity_recall(root, "full_model", 0)
    scarce_normal_plain = scarcity_recall(root, "without_diffusion", 0)
    attacks_before, attacks_after = scarcity_thinning(root)

    tolerance = 0.005
    results = [
        ClaimResult("unsw binary full model", 88.90, binary["unsw"]["full_model"], tolerance),
        ClaimResult(
            "unsw binary without diffusion", 87.22, binary["unsw"]["wo_diffusion"], tolerance
        ),
        ClaimResult("unsw binary full model spread", 0.57, unsw_spread["full_model"], tolerance),
        ClaimResult(
            "unsw binary without diffusion spread",
            0.34,
            unsw_spread["wo_diffusion"],
            tolerance,
        ),
        ClaimResult("nslkdd binary full model", 82.21, binary["nslkdd"]["full_model"], tolerance),
        ClaimResult(
            "cicids2017 binary full model", 98.20, binary["cicids2017"]["full_model"], tolerance
        ),
        ClaimResult(
            "cicddos2019 binary full model", 95.85, binary["cicddos2019"]["full_model"], tolerance
        ),
        ClaimResult("unsw multiclass full model", 45.93, multiclass["full_model"], tolerance),
        ClaimResult("unsw multiclass xgboost", 49.52, fair["XGBoost__balanced"], tolerance),
        ClaimResult(
            "unsw fgsm drop, augmented",
            13.10,
            fgsm_accuracy_drop(root, "unsw", "full_model"),
            tolerance,
        ),
        ClaimResult(
            "unsw fgsm drop, unaugmented",
            10.25,
            fgsm_accuracy_drop(root, "unsw", "baseline"),
            tolerance,
        ),
        ClaimResult("cicddos2019 residual imbalance", 1.07, ratio, tolerance),
        ClaimResult("expansion cap", 15.0, float(cap), 0.0),
        ClaimResult("balanced pairs improved", 15.0, float(improved), 0.0),
        ClaimResult("balanced pairs compared", 16.0, float(total), 0.0),
        ClaimResult("largest balancing gain", 17.70, largest, tolerance),
        ClaimResult("unsw correction cost, lowest", 0.12, unsw_cost[0], tolerance),
        ClaimResult("unsw correction cost, highest", 0.93, unsw_cost[1], tolerance),
        ClaimResult("nslkdd correction cost, lowest", 1.92, nslkdd_cost[0], tolerance),
        ClaimResult("nslkdd correction cost, highest", 2.69, nslkdd_cost[1], tolerance),
        ClaimResult("pre-declared correction tolerance", 2.00, CORRECTION_TOLERANCE, 0.0),
        ClaimResult("scarcity attack recall, augmented", 87.94, scarce_attack_full[0], tolerance),
        ClaimResult(
            "scarcity attack recall, unaugmented", 83.83, scarce_attack_plain[0], tolerance
        ),
        ClaimResult(
            "scarcity attack recall gain",
            4.11,
            scarce_attack_full[0] - scarce_attack_plain[0],
            tolerance,
        ),
        ClaimResult("scarcity attack recall, xgboost", 87.15, scarce_attack_xgb[0], tolerance),
        ClaimResult(
            "scarcity attack recall spread, augmented", 3.98, scarce_attack_full[1], tolerance
        ),
        ClaimResult(
            "scarcity attack recall spread, xgboost", 0.11, scarce_attack_xgb[1], tolerance
        ),
        ClaimResult("scarcity majority recall, augmented", 96.38, scarce_normal_full[0], tolerance),
        ClaimResult(
            "scarcity majority recall, unaugmented", 98.78, scarce_normal_plain[0], tolerance
        ),
        ClaimResult(
            "scarcity macro f1, augmented",
            91.71,
            scarcity_macro_f1(root, "full_model"),
            tolerance,
        ),
        ClaimResult(
            "scarcity macro f1, xgboost",
            92.11,
            scarcity_macro_f1(root, "xgboost_raw"),
            tolerance,
        ),
        ClaimResult("scarcity attack rows before", 119341.0, float(attacks_before), 0.0),
        ClaimResult("scarcity attack rows after", 5967.0, float(attacks_after), 0.0),
    ]
    return tuple(results)


__all__ = [
    "CORRECTION_TOLERANCE",
    "DEFAULT_EVIDENCE_ROOT",
    "ClaimResult",
    "balancing_effect",
    "correction_cost",
    "evaluate_claims",
    "fgsm_accuracy_drop",
    "residual_imbalance",
    "restored_off_default_recall",
    "scarcity_macro_f1",
    "scarcity_recall",
    "scarcity_thinning",
    "variant_means",
    "variant_spread",
]
