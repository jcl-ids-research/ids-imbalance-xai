"""Read archived experiment results and reduce them to comparable numbers.

This module answers "what does the archive say", with no knowledge of what the
manuscript printed. `claims` holds the printed values and pairs them with
these readings, so a disagreement is always visible as data against prose rather
than buried inside one function.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path

from ids_diffusion.errors import DatasetFileError
from ids_diffusion.reproduction.archive_io import (
    JsonObject,
    as_float,
    as_list,
    as_object,
    nested_float,
    read_json,
)

PAPER_SEEDS: tuple[int, ...] = (42, 123, 456)
FGSM_EPSILON = "0.05"
MULTICLASS_DATASETS: tuple[str, ...] = ("unsw", "nslkdd")
BASELINE_FAMILIES: tuple[str, ...] = (
    "baseline_fair_multiclass",
    "baseline_deep_multiclass",
)
BINARY_DATASETS: tuple[str, ...] = ("unsw", "nslkdd", "cicids2017", "cicddos2019")
ABLATIONS: tuple[str, ...] = ("wo_diffusion", "wo_multiview")


def seed_paths(root: Path, family: str, dataset: str) -> Iterable[Path]:
    """Yield the archived file for each paper seed, in either stored layout."""
    for seed in PAPER_SEEDS:
        flat = root / family / f"{dataset}_seed{seed}.json"
        yield flat if flat.is_file() else root / family / dataset / f"seed{seed}" / "metrics.json"


def macro_f1(variants: JsonObject, name: str, path: Path | None = None) -> float:
    """Return one variant's macro F1 in percent, whichever key the run wrote."""
    where = path or Path("<archive>")
    entry = as_object(variants[name], where, f"variant {name}")
    score = entry.get("f1_macro", entry.get("f1"))
    return as_float(score, where, f"{name} macro F1") * 100


def _variant_scores(root: Path, family: str, dataset: str) -> dict[str, list[float]]:
    """Collect every variant's macro F1 across the paper seeds, in percent."""
    collected: dict[str, list[float]] = {}
    for path in seed_paths(root, family, dataset):
        variants = as_object(read_json(path)["variants"], path, "variants block")
        for name in variants:
            collected.setdefault(name, []).append(macro_f1(variants, name, path))
    return collected


def variant_means(root: Path, family: str, dataset: str) -> dict[str, float]:
    """Return the three-seed mean macro F1 of every variant, in percent."""
    return {
        name: statistics.mean(values)
        for name, values in _variant_scores(root, family, dataset).items()
    }


def variant_spread(root: Path, family: str, dataset: str) -> dict[str, float]:
    """Return the three-seed standard deviation of every variant, in percent."""
    return {
        name: statistics.stdev(values)
        for name, values in _variant_scores(root, family, dataset).items()
    }


def fgsm_accuracy_drop(root: Path, dataset: str, variant: str) -> float:
    """Return the mean accuracy lost at the reported FGSM budget, in points."""
    drops: list[float] = []
    for seed in PAPER_SEEDS:
        path = root / "adversarial" / f"adversarial_{dataset}_seed{seed}.json"
        variants = as_object(read_json(path)["variants"], path, "variants block")
        entry = as_object(variants[variant], path, f"variant {variant}")
        clean = nested_float(entry, path, "clean", "acc")
        attacked = nested_float(entry, path, "fgsm", FGSM_EPSILON, "acc")
        if clean is None or attacked is None:
            raise DatasetFileError(path=str(path), detail=f"no FGSM record for {variant}")
        drops.append((clean - attacked) * 100)
    return statistics.mean(drops)


def residual_imbalance(root: Path, dataset: str) -> tuple[float, int]:
    """Return the post-balancing ratio and the expansion cap that produced it."""
    path = root / "phase1_corrected" / f"{dataset}_seed42.json"
    report = as_object(read_json(path)["balance_report"], path, "balance report")
    ratio = as_float(report["imbalance_ratio_after"], path, "imbalance ratio")
    cap = as_float(report["expansion_cap"], path, "expansion cap")
    return ratio, int(cap)


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
            for model in sorted({name.split("__")[0] for name in means}):
                raw = means.get(f"{model}__raw")
                balanced = means.get(f"{model}__balanced")
                if raw is not None and balanced is not None:
                    gains.append(balanced - raw)
    return sum(1 for gain in gains if gain > 0), len(gains), max(gains)


def seed_consistent_effects(root: Path) -> tuple[int, int, int]:
    """Return how many component effects hold in sign across all three seeds.

    Reported as (agreeing, beneficial, harmful). The manuscript treats sign
    agreement as its directional bar, so the count behind that sentence is
    derived here; an effect that changes sign after a rerun then shows up as a
    claim mismatch rather than as prose that quietly stopped being true.
    """
    agreeing = beneficial = harmful = 0
    for ablation in ABLATIONS:
        for dataset in BINARY_DATASETS:
            deltas: list[float] = []
            for path in seed_paths(root, "phase1_corrected", dataset):
                variants = as_object(read_json(path)["variants"], path, "variants block")
                deltas.append(
                    macro_f1(variants, "full_model", path) - macro_f1(variants, ablation, path)
                )
            if len({delta > 0 for delta in deltas}) != 1:
                continue
            agreeing += 1
            if statistics.mean(deltas) > 0:
                beneficial += 1
            else:
                harmful += 1
    return agreeing, beneficial, harmful


def scarcity_recall(root: Path, variant: str, label: int) -> tuple[float, float]:
    """Return the mean and spread of one class's recall under training-side scarcity.

    The scarcity run thins UNSW-NB15 attack rows to five per cent of their
    original count and leaves the test partition as published, so the attack
    class is rare while the model learns and ordinary when it is scored. It is
    the one setting in which the proposed model leads a gradient-boosted
    baseline on the metric the augmentation targets.
    """
    values: list[float] = []
    for seed in PAPER_SEEDS:
        path = root / "scarcity" / f"keep005_seed{seed}.json"
        variants = as_object(read_json(path)["variants"], path, "variants block")
        entry = as_object(variants[variant], path, f"variant {variant}")
        per_class = as_list(entry["per_class"], path, "per_class")
        match = next(
            (
                item
                for item in per_class
                if isinstance(item, Mapping) and item.get("label") == label
            ),
            None,
        )
        if match is None:
            raise DatasetFileError(path=str(path), detail=f"class {label} missing for {variant}")
        values.append(as_float(match["recall"], path, "recall") * 100)
    return statistics.mean(values), statistics.stdev(values)


def scarcity_macro_f1(root: Path, variant: str) -> float:
    """Return the three-seed mean macro F1 of one variant in the scarcity run."""
    scores: list[float] = []
    for seed in PAPER_SEEDS:
        path = root / "scarcity" / f"keep005_seed{seed}.json"
        variants = as_object(read_json(path)["variants"], path, "variants block")
        entry = as_object(variants[variant], path, f"variant {variant}")
        scores.append(as_float(entry["macro_f1"], path, "macro F1") * 100)
    return statistics.mean(scores)


def scarcity_thinning(root: Path) -> tuple[int, int]:
    """Return the attack-row count before and after thinning."""
    path = root / "scarcity" / "keep005_seed42.json"
    report = as_object(read_json(path)["shift_report"], path, "shift report")
    before = as_float(report["attacks_before"], path, "attacks before")
    after = as_float(report["attacks_after"], path, "attacks after")
    return int(before), int(after)


__all__ = [
    "ABLATIONS",
    "BINARY_DATASETS",
    "PAPER_SEEDS",
    "balancing_effect",
    "fgsm_accuracy_drop",
    "macro_f1",
    "residual_imbalance",
    "scarcity_macro_f1",
    "scarcity_recall",
    "scarcity_thinning",
    "seed_consistent_effects",
    "seed_paths",
    "variant_means",
    "variant_spread",
]
