from __future__ import annotations

import itertools

import pytest

from ids_diffusion.config import ExperimentConfig
from ids_diffusion.tuning.search_space import (
    ARCHITECTURES,
    BATCH_SIZES,
    DIFFUSION_HIDDEN,
    FUSION_MODES,
    NARROW_BATCH_SIZES,
    suggest_classifier_trial,
    suggest_diffusion_trial,
    suggest_narrow_classifier_trial,
    suggest_trial,
)
from ids_diffusion.types import FusionMode


class ScriptedTrial:
    """A trial that returns fixed choices, so a search space can be tested alone.

    Optuna is not needed to check the invariants that matter here: that every
    suggested configuration is constructible, and that no combination of choices
    can produce an architecture the model layer would reject.
    """

    def __init__(
        self,
        *,
        categorical: dict[str, str] | None = None,
        integer: str = "low",
        floating: str = "low",
    ) -> None:
        self._categorical = categorical or {}
        self._integer = integer
        self._floating = floating
        self.asked: list[str] = []

    def suggest_categorical(self, name: str, choices: tuple[str, ...]) -> str:
        self.asked.append(name)
        return self._categorical.get(name, choices[0])

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.asked.append(name)
        return low if self._integer == "low" else high

    def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
        self.asked.append(name)
        _ = log
        return low if self._floating == "low" else high


def _base() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("fusion", FUSION_MODES)
@pytest.mark.parametrize("extreme", ["low", "high"])
def test_no_combination_of_choices_produces_an_invalid_architecture(
    architecture: str, fusion: str, extreme: str
) -> None:
    # Given: every architecture and fusion the space offers, at both ends of
    # each numeric range
    trial = ScriptedTrial(
        categorical={"architecture": architecture, "fusion_mode": fusion},
        integer=extreme,
        floating=extreme,
    )

    # When: a configuration is suggested
    config = suggest_classifier_trial(trial, _base())

    # Then: it is constructible and the head count divides the width. This is
    # the promise the module's docstring makes, and it is what stops a tuning
    # run dying hours in on an unbuildable proposal.
    assert config.model.d_model % config.model.n_heads == 0
    assert config.model.n_layers >= 2
    assert 0.0 <= config.model.dropout < 1.0
    assert config.training.batch_size in {int(size) for size in BATCH_SIZES}
    assert config.training.learning_rate > 0
    assert config.training.weight_decay > 0


def test_the_classifier_space_leaves_the_generator_untouched() -> None:
    # Given: a base configuration
    base = _base()

    # When: only the classifier is searched
    config = suggest_classifier_trial(ScriptedTrial(), base)

    # Then: the generator is carried through unchanged, which is what makes the
    # staged protocol in the paper valid -- classifier search runs against fixed
    # corrected caches
    assert config.diffusion == base.diffusion


def test_the_diffusion_space_leaves_the_classifier_untouched() -> None:
    # Given: a base configuration
    base = _base()

    # When: only the generator is searched
    config = suggest_diffusion_trial(ScriptedTrial(), base)

    # Then: the classifier is untouched, so a generator gain cannot be confused
    # with a classifier gain
    assert config.model == base.model


@pytest.mark.parametrize("hidden", DIFFUSION_HIDDEN)
def test_every_generator_width_string_parses_into_a_usable_shape(hidden: str) -> None:
    # Given: each hidden-width option the space offers
    trial = ScriptedTrial(categorical={"diffusion_hidden": hidden})

    # When: a generator configuration is suggested
    config = suggest_diffusion_trial(trial, _base())

    # Then: it becomes a tuple of positive widths rather than a string
    assert isinstance(config.diffusion.hidden, tuple)
    assert len(config.diffusion.hidden) >= 1
    assert all(isinstance(width, int) and width > 0 for width in config.diffusion.hidden)


@pytest.mark.parametrize("extreme", ["low", "high"])
def test_the_expansion_cap_stays_inside_the_declared_range(extreme: str) -> None:
    # Given: both ends of the integer ranges
    trial = ScriptedTrial(integer=extreme)

    # When: a generator configuration is suggested
    config = suggest_diffusion_trial(trial, _base())

    # Then: the cap remains within the bounds the protocol declares, so tuning
    # cannot quietly widen the residual-imbalance story the paper tells
    assert 5 <= config.training.expansion_cap <= 20
    assert 300 <= config.diffusion.timesteps <= 700


@pytest.mark.parametrize("extreme", ["low", "high"])
def test_the_narrow_space_fixes_what_the_first_study_settled(extreme: str) -> None:
    # Given: the narrowed space used after multi-seed confirmation
    trial = ScriptedTrial(integer=extreme, floating=extreme)

    # When: a configuration is suggested
    config = suggest_narrow_classifier_trial(trial, _base())

    # Then: width, heads and fusion are pinned, and only depth, capacity and
    # optimisation vary -- the budget is not spent re-exploring settled choices
    assert config.model.d_model == 256
    assert config.model.n_heads == 8
    assert config.model.fusion_mode is FusionMode.LINEAR
    assert 2 <= config.model.n_layers <= 3
    assert config.training.batch_size in {int(size) for size in NARROW_BATCH_SIZES}
    assert "architecture" not in trial.asked
    assert "fusion_mode" not in trial.asked


def test_the_joint_space_searches_both_halves() -> None:
    # Given: the joint space used for final narrowed-range experiments
    base = _base()
    trial = ScriptedTrial(integer="high", floating="high")

    # When: a configuration is suggested
    config = suggest_trial(trial, base)

    # Then: both halves moved, so the joint stage is not silently equivalent to
    # one of the staged searches
    assert config.model != base.model
    assert config.diffusion != base.diffusion
    assert "architecture" in trial.asked
    assert "diffusion_timesteps" in trial.asked


def test_repeated_identical_choices_produce_identical_configurations() -> None:
    # Given: two trials scripted the same way
    first = suggest_trial(ScriptedTrial(integer="low", floating="low"), _base())
    second = suggest_trial(ScriptedTrial(integer="low", floating="low"), _base())

    # Then: the space is a pure function of its choices, which is what makes a
    # tuning study reproducible from its recorded parameters
    assert first == second


def test_every_architecture_and_batch_size_pairing_is_constructible() -> None:
    # Given: the full cross product of categorical choices
    for architecture, fusion, batch in itertools.product(ARCHITECTURES, FUSION_MODES, BATCH_SIZES):
        trial = ScriptedTrial(
            categorical={
                "architecture": architecture,
                "fusion_mode": fusion,
                "batch_size": batch,
            }
        )

        # When / Then: none of the combinations raises, so the search space
        # cannot waste a trial on a configuration the model layer rejects
        config = suggest_classifier_trial(trial, _base())
        assert config.model.dim_feedforward >= config.model.d_model
