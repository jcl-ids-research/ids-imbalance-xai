from __future__ import annotations

from ids_diffusion.data.nslkdd import NSL_SEMANTIC_VIEWS
from ids_diffusion.data.registry import view_indices_for_dataset


def test_nslkdd_semantic_views_cover_each_feature_once() -> None:
    # Given: the complete NSL-KDD feature schema used by the paper
    feature_names = tuple(name for group in NSL_SEMANTIC_VIEWS for name in group)

    # When: the registry resolves the semantic partition
    views = view_indices_for_dataset("nslkdd", feature_names, view_count=3)

    # Then: every feature belongs to exactly one non-empty view
    flattened = tuple(index for view in views for index in view)
    assert all(views)
    assert sorted(flattened) == list(range(len(feature_names)))


def test_cic_datasets_use_deterministic_positional_views() -> None:
    # Given: a CIC feature vector without a published semantic grouping
    feature_names = tuple(f"feature_{index}" for index in range(8))

    # When: both CIC datasets resolve their view partition
    ids_views = view_indices_for_dataset("cicids2017", feature_names, view_count=3)
    ddos_views = view_indices_for_dataset("cicddos2019", feature_names, view_count=3)

    # Then: both use the same complete positional partition
    assert ids_views == ddos_views == ((0, 1, 2), (3, 4, 5), (6, 7))
