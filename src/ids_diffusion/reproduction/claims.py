"""Pair the manuscript's printed numbers with what the archive reproduces.

A reviewer should be able to check the paper's claims without a GPU and without
waiting for a full rerun. Every printed value below sits next to a reading taken
from the archived server results under `evidence/paper_results`, so a mismatch
means either the archive or the manuscript moved.

The readings themselves live in `measurements`, which knows nothing about what
was printed. Keeping the two apart means a disagreement always shows up as data
against prose rather than hiding inside one function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ids_diffusion.reproduction.fidelity import (
    correction_cost,
    fidelity_pairs,
    mmd_improvement,
    restored_off_default_recall,
)
from ids_diffusion.reproduction.measurements import (
    PAPER_SEEDS,
    balancing_effect,
    fgsm_accuracy_drop,
    residual_imbalance,
    scarcity_macro_f1,
    scarcity_recall,
    scarcity_thinning,
    seed_consistent_effects,
    variant_means,
    variant_spread,
)

DEFAULT_EVIDENCE_ROOT = Path("evidence/paper_results")
CORRECTION_TOLERANCE = 2.0
BINARY_BENCHMARKS: tuple[str, ...] = ("unsw", "nslkdd", "cicids2017", "cicddos2019")


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


def _headline_claims(root: Path, tol: float) -> list[ClaimResult]:
    """Return the cross-benchmark results reported in Tables 2 to 8."""
    binary = {name: variant_means(root, "phase1_corrected", name) for name in BINARY_BENCHMARKS}
    spread = variant_spread(root, "phase1_corrected", "unsw")
    multiclass = variant_means(root, "multiclass", "unsw")
    fair = variant_means(root, "baseline_fair_multiclass", "unsw")
    return [
        ClaimResult("unsw binary full model", 88.90, binary["unsw"]["full_model"], tol),
        ClaimResult("unsw binary without diffusion", 87.22, binary["unsw"]["wo_diffusion"], tol),
        ClaimResult("unsw binary full model spread", 0.57, spread["full_model"], tol),
        ClaimResult("unsw binary without diffusion spread", 0.34, spread["wo_diffusion"], tol),
        ClaimResult("nslkdd binary full model", 82.21, binary["nslkdd"]["full_model"], tol),
        ClaimResult("cicids2017 binary full model", 98.20, binary["cicids2017"]["full_model"], tol),
        ClaimResult(
            "cicddos2019 binary full model", 95.85, binary["cicddos2019"]["full_model"], tol
        ),
        ClaimResult("unsw multiclass full model", 45.93, multiclass["full_model"], tol),
        ClaimResult("unsw multiclass xgboost", 49.52, fair["XGBoost__balanced"], tol),
    ]


def _robustness_and_balance_claims(root: Path, tol: float) -> list[ClaimResult]:
    """Return the adversarial cost, the expansion cap and the balancing result."""
    ratio, cap = residual_imbalance(root, "cicddos2019")
    improved, total, largest = balancing_effect(root)
    return [
        ClaimResult(
            "unsw fgsm drop, augmented", 13.10, fgsm_accuracy_drop(root, "unsw", "full_model"), tol
        ),
        ClaimResult(
            "unsw fgsm drop, unaugmented", 10.25, fgsm_accuracy_drop(root, "unsw", "baseline"), tol
        ),
        ClaimResult("cicddos2019 residual imbalance", 1.07, ratio, tol),
        ClaimResult("expansion cap", 15.0, float(cap), 0.0),
        ClaimResult("balanced pairs improved", 15.0, float(improved), 0.0),
        ClaimResult("balanced pairs compared", 16.0, float(total), 0.0),
        ClaimResult("largest balancing gain", 17.70, largest, tol),
    ]


def _fidelity_claims(root: Path, tol: float) -> list[ClaimResult]:
    """Return what the rank-matched correction restores and what it costs."""
    unsw = correction_cost(root, "unsw")
    nslkdd = correction_cost(root, "nslkdd")
    mmd_improved, mmd_total, _ = mmd_improvement(root)
    return [
        ClaimResult("unsw correction cost, lowest", 0.12, unsw[0], tol),
        ClaimResult("unsw correction cost, highest", 0.93, unsw[1], tol),
        ClaimResult("nslkdd correction cost, lowest", 1.92, nslkdd[0], tol),
        ClaimResult("nslkdd correction cost, highest", 2.69, nslkdd[1], tol),
        ClaimResult("pre-declared correction tolerance", 2.00, CORRECTION_TOLERANCE, 0.0),
        ClaimResult("fidelity dataset-seed pairs", 9.0, float(fidelity_pairs(root)), 0.0),
        ClaimResult("mmd improved pairs", 8.0, float(mmd_improved), 0.0),
        ClaimResult("mmd measured pairs", 9.0, float(mmd_total), 0.0),
    ]


def _component_claims(root: Path) -> list[ClaimResult]:
    """Return how many component effects survive the seed-agreement bar."""
    agreeing, beneficial, harmful = seed_consistent_effects(root)
    return [
        ClaimResult("seed-consistent effects", 5.0, float(agreeing), 0.0),
        ClaimResult("seed-consistent effects, beneficial", 3.0, float(beneficial), 0.0),
        ClaimResult("seed-consistent effects, harmful", 2.0, float(harmful), 0.0),
    ]


def _scarcity_claims(root: Path, tol: float) -> list[ClaimResult]:
    """Return the minority-recall result under training-side scarcity."""
    attack_full = scarcity_recall(root, "full_model", 1)
    attack_plain = scarcity_recall(root, "without_diffusion", 1)
    attack_xgb = scarcity_recall(root, "xgboost_raw", 1)
    normal_full = scarcity_recall(root, "full_model", 0)
    normal_plain = scarcity_recall(root, "without_diffusion", 0)
    before, after = scarcity_thinning(root)
    return [
        ClaimResult("scarcity attack recall, augmented", 87.94, attack_full[0], tol),
        ClaimResult("scarcity attack recall, unaugmented", 83.83, attack_plain[0], tol),
        ClaimResult("scarcity attack recall gain", 4.11, attack_full[0] - attack_plain[0], tol),
        ClaimResult("scarcity attack recall, xgboost", 87.15, attack_xgb[0], tol),
        ClaimResult("scarcity attack recall spread, augmented", 3.98, attack_full[1], tol),
        ClaimResult("scarcity attack recall spread, xgboost", 0.11, attack_xgb[1], tol),
        ClaimResult("scarcity majority recall, augmented", 96.38, normal_full[0], tol),
        ClaimResult("scarcity majority recall, unaugmented", 98.78, normal_plain[0], tol),
        ClaimResult(
            "scarcity macro f1, augmented", 91.71, scarcity_macro_f1(root, "full_model"), tol
        ),
        ClaimResult(
            "scarcity macro f1, xgboost", 92.11, scarcity_macro_f1(root, "xgboost_raw"), tol
        ),
        ClaimResult("scarcity attack rows before", 119341.0, float(before), 0.0),
        ClaimResult("scarcity attack rows after", 5967.0, float(after), 0.0),
    ]


def evaluate_claims(root: Path = DEFAULT_EVIDENCE_ROOT) -> tuple[ClaimResult, ...]:
    """Recompute every headline number the manuscript reports."""
    tol = 0.005
    return (
        *_headline_claims(root, tol),
        *_robustness_and_balance_claims(root, tol),
        *_fidelity_claims(root, tol),
        *_component_claims(root),
        *_scarcity_claims(root, tol),
    )


__all__ = [
    "CORRECTION_TOLERANCE",
    "DEFAULT_EVIDENCE_ROOT",
    "PAPER_SEEDS",
    "ClaimResult",
    "balancing_effect",
    "correction_cost",
    "evaluate_claims",
    "fgsm_accuracy_drop",
    "fidelity_pairs",
    "mmd_improvement",
    "residual_imbalance",
    "restored_off_default_recall",
    "scarcity_macro_f1",
    "scarcity_recall",
    "scarcity_thinning",
    "seed_consistent_effects",
    "variant_means",
    "variant_spread",
]
