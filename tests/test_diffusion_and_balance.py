from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import QuantileTransformer

from ids_diffusion.config import DiffusionConfig, TrainingConfig
from ids_diffusion.models.diffusion import ClassConditionalDdpm
from ids_diffusion.training.balance import BalancingJob, rebalance
from ids_diffusion.training.diffusion import DiffusionTrainingJob, train_diffusion
from ids_diffusion.types import DatasetMatrix

CPU = torch.device("cpu")


def _config(timesteps: int = 8, epochs: int = 1, batch_size: int = 16) -> DiffusionConfig:
    return DiffusionConfig(
        timesteps=timesteps, hidden=(16, 16), epochs=epochs, batch_size=batch_size
    )


def test_the_noise_schedule_stays_inside_its_valid_range() -> None:
    # Given: the fixed schedule the forward process in the paper relies on
    model = ClassConditionalDdpm(input_dim=4, config=_config(timesteps=64))

    # Then: every beta is a usable variance and alpha-bar decays monotonically,
    # which is what makes the closed-form noising and the reverse step well posed
    assert torch.all(model.betas > 0)
    assert torch.all(model.betas < 1)
    assert torch.all(torch.diff(model.alpha_bar) <= 1e-6)
    assert float(model.alpha_bar[0]) > float(model.alpha_bar[-1])


def test_alpha_bar_matches_the_closed_form_the_manuscript_prints() -> None:
    # Given: the schedule buffers the model registers
    model = ClassConditionalDdpm(input_dim=3, config=_config(timesteps=32))

    # When: the cumulative product is recomputed from betas alone
    expected = torch.cumprod(1.0 - model.betas, dim=0)

    # Then: the buffers agree, so the equations added for comment m2 describe
    # the code rather than an idealised version of it
    assert torch.allclose(model.alpha_bar, expected, atol=1e-6)
    assert torch.allclose(model.sqrt_alpha_bar, torch.sqrt(expected), atol=1e-6)
    assert torch.allclose(model.sqrt_one_minus_alpha_bar, torch.sqrt(1.0 - expected), atol=1e-6)


def test_the_denoising_loss_is_a_finite_scalar() -> None:
    # Given: a small model and a batch of clean rows
    _ = torch.manual_seed(0)
    model = ClassConditionalDdpm(input_dim=6, config=_config())

    # When: the training objective is evaluated
    loss = model(torch.randn(12, 6))

    # Then: it reduces to one finite number that can be backpropagated
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0


def test_sampling_returns_bounded_rows_of_the_requested_shape() -> None:
    # Given: an untrained generator, which is the worst case for stability
    _ = torch.manual_seed(0)
    model = ClassConditionalDdpm(input_dim=5, config=_config())

    # When: rows are drawn by reversing the schedule
    samples = model.sample(7, CPU)

    # Then: the clamp in the reverse loop holds, so a diverging generator
    # cannot poison the balanced cache with infinities
    assert samples.shape == (7, 5)
    assert torch.isfinite(samples).all()
    assert float(samples.min()) >= -5.0
    assert float(samples.max()) <= 5.0


def test_training_reduces_the_objective_on_a_learnable_signal() -> None:
    # Given: rows drawn from one tight cluster, which a denoiser can fit
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(0)
    features = rng.normal(loc=0.5, scale=0.05, size=(128, 4)).astype(np.float32)
    model = ClassConditionalDdpm(input_dim=4, config=_config(epochs=40, batch_size=32))

    _ = torch.manual_seed(1)
    before = float(model(torch.from_numpy(features)).detach())
    trained = train_diffusion(
        DiffusionTrainingJob(
            model=model,
            features=features,
            config=_config(epochs=40, batch_size=32),
            device=CPU,
            class_label=1,
        )
    )
    _ = torch.manual_seed(1)
    after = float(trained(torch.from_numpy(features)).detach())

    # Then: fitting actually moves the model. Without this, every downstream
    # fidelity number would be measuring an untrained generator.
    assert after < before


def _job(labels: np.ndarray, features: np.ndarray, cap: int) -> BalancingJob:
    transformer = QuantileTransformer(
        n_quantiles=min(64, len(features)), output_distribution="normal"
    )
    _ = transformer.fit(features)
    generators = {
        int(label): ClassConditionalDdpm(input_dim=features.shape[1], config=_config())
        for label in np.unique(labels)
    }
    return BalancingJob(
        training=DatasetMatrix(features=features, labels=labels),
        generators=generators,
        transformer=transformer,
        config=TrainingConfig(expansion_cap=cap),
        device=CPU,
        seed=7,
    )


def test_balancing_lifts_the_rare_class_and_undersamples_the_common_one() -> None:
    # Given: one class that dominates and one that is scarce
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(220, 4)).astype(np.float32)
    labels = np.array([0] * 200 + [1] * 20, dtype=np.int64)

    # When: the partition is rebalanced
    result = rebalance(_job(labels, features, cap=15))

    # Then: both classes end at the same size, the majority by discarding rows
    # and the minority by generation, and the ratio improves
    sizes = {int(item.label): item.after for item in result.report.classes}
    assert sizes[0] == sizes[1]
    assert result.report.imbalance_after < result.report.imbalance_before
    assert result.report.imbalance_after == pytest.approx(1.0)


def test_the_expansion_cap_leaves_a_deliberate_residual_imbalance() -> None:
    # Given: a class so rare that reaching parity would need a large expansion
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(1)
    features = rng.normal(size=(410, 3)).astype(np.float32)
    labels = np.array([0] * 400 + [1] * 10, dtype=np.int64)

    # When: the cap is set below the expansion parity would require
    result = rebalance(_job(labels, features, cap=2))

    # Then: the rare class stops at the cap rather than reaching parity. This is
    # the mechanism behind the 1.07:1 residual the manuscript discloses.
    rare = next(item for item in result.report.classes if item.label == 1)
    assert rare.after == 10 * 2
    assert result.report.imbalance_after > 1.0


def test_synthetic_rows_stay_inside_the_observed_range_of_their_class() -> None:
    # Given: a scarce class whose real rows occupy a known range
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(2)
    features = rng.normal(size=(160, 4)).astype(np.float32)
    labels = np.array([0] * 140 + [1] * 20, dtype=np.int64)

    # When: rows are generated for it
    result = rebalance(_job(labels, features, cap=15))
    generated = result.synthetic.features[result.synthetic.labels == 1]
    real = features[labels == 1]

    # Then: generation cannot invent values outside what was observed, which is
    # what stops the augmentation fabricating impossible flow records
    assert len(generated) > 0
    assert np.all(generated >= real.min(axis=0) - 1e-4)
    assert np.all(generated <= real.max(axis=0) + 1e-4)
    assert np.isfinite(generated).all()


def test_an_already_balanced_partition_generates_nothing() -> None:
    # Given: three classes of equal size
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(3)
    features = rng.normal(size=(150, 3)).astype(np.float32)
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int64)

    # When: the partition is rebalanced
    result = rebalance(_job(labels, features, cap=15))

    # Then: balancing is inert, so it cannot be credited with gains on data
    # that never needed it
    assert result.synthetic.features.shape[0] == 0
    assert all(item.generated == 0 for item in result.report.classes)


def test_no_synthetic_row_carries_a_label_that_was_not_expanded() -> None:
    # Given: an imbalanced partition
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(4)
    features = rng.normal(size=(230, 3)).astype(np.float32)
    labels = np.array([0] * 200 + [1] * 30, dtype=np.int64)

    # When / Then: synthetic rows exist only for classes the report says were
    # expanded, so the balance report and the cache cannot disagree
    result = rebalance(_job(labels, features, cap=15))
    expanded = {int(item.label) for item in result.report.classes if item.generated > 0}
    produced = {int(label) for label in np.unique(result.synthetic.labels)}
    assert produced == expanded
