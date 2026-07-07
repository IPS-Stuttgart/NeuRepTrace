from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_prior_shift import (
    TARGET_PRIOR_SHIFT_CATEGORY,
    adapt_target_probabilities_prior_shift,
    normalize_initial_prior,
    target_prior_shift_config,
)


def test_target_prior_shift_returns_normalized_probabilities_and_metadata() -> None:
    probabilities = np.asarray(
        [
            [0.80, 0.20],
            [0.75, 0.25],
            [0.30, 0.70],
            [0.20, 0.80],
        ],
        dtype=float,
    )

    result = adapt_target_probabilities_prior_shift(probabilities, source_prior=[0.5, 0.5])

    assert result.probabilities.shape == probabilities.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.estimated_target_prior.sum(), 1.0)
    assert result.metadata["target_prior_shift_protocol_category"] == TARGET_PRIOR_SHIFT_CATEGORY
    assert result.metadata["target_prior_shift_uses_target_probabilities"] is True
    assert result.metadata["target_prior_shift_uses_target_labels"] is False
    assert result.metadata["target_prior_shift_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["target_prior_shift_valid_for_strict_source_only"] is False


def test_target_prior_shift_with_source_prior_changes_rows() -> None:
    probabilities = np.asarray([[0.60, 0.40], [0.55, 0.45], [0.50, 0.50]], dtype=float)

    result = adapt_target_probabilities_prior_shift(probabilities, source_prior=[0.9, 0.1])

    assert np.allclose(result.source_prior, np.asarray([0.9, 0.1], dtype=np.float32))
    assert np.all(result.prior_ratio > 0.0)
    assert not np.allclose(result.probabilities, result.original_probabilities)


def test_target_prior_shift_initial_prior_aliases() -> None:
    assert normalize_initial_prior("mean") == "mean_probability"
    assert normalize_initial_prior("source") == "source_prior"
    assert target_prior_shift_config(max_iter="5", initial_prior="uniform").max_iter == 5

    with pytest.raises(ValueError, match="initial_prior"):
        normalize_initial_prior("bad")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": True}, "max_iter"),
        ({"max_iter": np.asarray(3)}, "max_iter"),
        ({"tol": True}, "tol"),
        ({"tol": np.asarray(1e-8)}, "tol"),
        ({"epsilon": np.bool_(True)}, "epsilon"),
        ({"epsilon": np.asarray(1e-12)}, "epsilon"),
    ],
)
def test_target_prior_shift_config_rejects_boolean_and_array_scalar_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        target_prior_shift_config(**kwargs)


def test_target_prior_shift_config_accepts_numpy_numeric_scalar_controls() -> None:
    config = target_prior_shift_config(
        max_iter=np.int64(5),
        tol=np.float64(1e-7),
        epsilon=np.float64(1e-10),
    )

    assert config.max_iter == 5
    assert np.isclose(config.tol, 1e-7)
    assert np.isclose(config.epsilon, 1e-10)


@pytest.mark.parametrize(
    "probabilities",
    [
        [[True, False]],
        np.asarray([[True, False]], dtype=bool),
        np.asarray([[np.bool_(True), 0.0]], dtype=object),
        (row for row in [[True, False]]),
    ],
)
def test_target_prior_shift_rejects_boolean_probability_rows(probabilities: object) -> None:
    with pytest.raises(ValueError, match="probabilities.*boolean"):
        adapt_target_probabilities_prior_shift(probabilities)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "source_prior",
    [
        [True, False],
        np.asarray([True, False], dtype=bool),
        np.asarray([np.bool_(True), 0.0], dtype=object),
        (value for value in [True, False]),
    ],
)
def test_target_prior_shift_rejects_boolean_source_prior(source_prior: object) -> None:
    with pytest.raises(ValueError, match="source_prior.*boolean"):
        adapt_target_probabilities_prior_shift([[0.5, 0.5]], source_prior=source_prior)  # type: ignore[arg-type]


def test_target_prior_shift_rejects_bad_prior_length() -> None:
    with pytest.raises(ValueError, match="source_prior"):
        adapt_target_probabilities_prior_shift([[0.5, 0.5]], source_prior=[1.0, 0.0, 0.0])


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        adapt_target_probabilities_prior_shift(
            [[0.5, 0.5]],
            target_labels=[0],  # type: ignore[call-arg]
        )
