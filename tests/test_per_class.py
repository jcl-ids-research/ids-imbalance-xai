from __future__ import annotations

import numpy as np

from ids_diffusion.training.metrics import classification_metrics, mean_metrics


def test_per_class_recall_exposes_a_missed_minority() -> None:
    # Given: a detector that catches every normal row but only half the attacks
    labels = np.array([0] * 8 + [1] * 4, dtype=np.int64)
    predictions = np.array([0] * 8 + [1, 1, 0, 0], dtype=np.int64)

    # When: metrics are computed
    metrics = classification_metrics(labels, predictions)

    # Then: the averaged score stays high while the minority recall reveals the miss
    by_label = {entry.label: entry for entry in metrics.per_class}
    assert by_label[0].recall == 1.0
    assert by_label[1].recall == 0.5
    assert by_label[1].support == 4
    assert metrics.accuracy > by_label[1].recall


def test_per_class_covers_every_label_present() -> None:
    # Given: three classes in the ground truth
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    predictions = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)

    # When: metrics are computed
    metrics = classification_metrics(labels, predictions)

    # Then: every label is reported with its support
    assert {entry.label for entry in metrics.per_class} == {0, 1, 2}
    assert sum(entry.support for entry in metrics.per_class) == len(labels)


def test_mean_metrics_averages_per_class_recall() -> None:
    # Given: two runs whose minority recall differs
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    first = classification_metrics(labels, np.array([0, 0, 1, 1], dtype=np.int64))
    second = classification_metrics(labels, np.array([0, 0, 0, 0], dtype=np.int64))

    # When: the runs are averaged
    averaged = mean_metrics((first, second))

    # Then: the minority recall is the mean of 1.0 and 0.0
    by_label = {entry.label: entry for entry in averaged.per_class}
    assert by_label[1].recall == 0.5
