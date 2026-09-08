from __future__ import annotations

from pathlib import Path

import pytest

from ids_diffusion.reproduction.claims import (
    CORRECTION_TOLERANCE,
    balancing_effect,
    correction_cost,
    evaluate_claims,
    fgsm_accuracy_drop,
    residual_imbalance,
    restored_off_default_recall,
    variant_means,
)

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "paper_results"
pytestmark = pytest.mark.skipif(
    not EVIDENCE.is_dir(),
    reason="archived paper results are not present in this checkout",
)


def test_every_headline_number_is_reproduced_by_the_archive() -> None:
    # Given: the archived server results shipped with the repository
    # When: each printed number is recomputed from them
    results = evaluate_claims(EVIDENCE)

    # Then: no manuscript number drifts from its evidence
    drifted = [
        f"{result.name}: paper {result.claimed} vs archive {result.archived:.4f}"
        for result in results
        if not result.holds
    ]
    assert not drifted, drifted
    assert len(results) == 21


def test_correction_cost_breaches_the_declared_tolerance_only_on_nslkdd() -> None:
    # Given: the archived before/after correlation preservation per dataset
    # When: the cost of the correction is recomputed for both labelled benchmarks
    unsw_low, unsw_high = correction_cost(EVIDENCE, "unsw")
    nslkdd_low, nslkdd_high = correction_cost(EVIDENCE, "nslkdd")

    # Then: only NSL-KDD exceeds the pre-declared tolerance, as the paper admits
    assert unsw_low == pytest.approx(0.12, abs=0.005)
    assert unsw_high == pytest.approx(0.93, abs=0.005)
    assert nslkdd_low == pytest.approx(1.92, abs=0.005)
    assert nslkdd_high == pytest.approx(2.69, abs=0.005)
    assert unsw_high < CORRECTION_TOLERANCE < nslkdd_high


def test_correction_restores_off_default_recall_to_about_one_hundred_percent() -> None:
    # Given: the recall of non-modal discrete values after correction
    # When: the extremes across every dataset and seed are taken
    lowest, highest = restored_off_default_recall(EVIDENCE)

    # Then: the restoration lands on 100% rather than merely improving
    assert lowest == pytest.approx(100.0, abs=0.05)
    assert highest == pytest.approx(100.0, abs=2.0)


def test_augmentation_costs_adversarial_robustness_on_unsw() -> None:
    # Given: the archived FGSM evaluation on UNSW-NB15
    # When: the accuracy lost at the reported budget is measured for both arms
    augmented = fgsm_accuracy_drop(EVIDENCE, "unsw", "full_model")
    unaugmented = fgsm_accuracy_drop(EVIDENCE, "unsw", "baseline")

    # Then: the augmented detector is the more fragile of the two
    assert augmented > unaugmented
    assert augmented == pytest.approx(13.10, abs=0.005)
    assert unaugmented == pytest.approx(10.25, abs=0.005)


def test_cic_ddos_balancing_stops_at_the_expansion_cap() -> None:
    # Given: the balancing report for the most skewed benchmark
    # When: the post-balancing ratio and its cap are read back
    ratio, cap = residual_imbalance(EVIDENCE, "cicddos2019")

    # Then: the residual imbalance the paper discloses is confirmed
    assert cap == 15
    assert ratio == pytest.approx(1.07, abs=0.005)


def test_balancing_helps_almost_every_model_dataset_pair() -> None:
    # Given: every classical and deep baseline on both labelled benchmarks
    # When: balanced training is compared with untouched training
    improved, total, largest = balancing_effect(EVIDENCE)

    # Then: the model-independent finding holds with its reported maximum
    assert (improved, total) == (15, 16)
    assert largest == pytest.approx(17.70, abs=0.005)


def test_binary_ablation_covers_four_benchmarks_and_four_variants() -> None:
    # Given: the corrected main experiment
    # When: each benchmark's variant means are recomputed
    datasets = ("unsw", "nslkdd", "cicids2017", "cicddos2019")
    tables = {name: variant_means(EVIDENCE, "phase1_corrected", name) for name in datasets}

    # Then: every benchmark reports the same four ablation arms
    expected = {"full_model", "wo_diffusion", "wo_multiview", "baseline"}
    for name, table in tables.items():
        assert set(table) == expected, name
