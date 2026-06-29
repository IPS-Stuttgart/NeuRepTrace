from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_probability_smoothing import (
    TARGET_PROBABILITY_SMOOTHING_CATEGORY,
    rbf_affinity,
    row_normalize,
    smooth_target_probabilities,
    target_probability_smoothing_config,
)


def test_target_probability_smoothing_preserves_probability_rows_and_metadata() -> None:
    features = np.asarray([[0.0], [0.1], [5.0], [5.1]], dtype=float)
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=float)

    result = smooth_target_probabilities(
        features,
        probabilities,
        config={"alpha": 0.5, "n_neighbors": 1, "max_iter": 25},
    )

    assert result.probabilities.shape == probabilities.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.affinity.shape == (4, 4)
    assert result.n_iter >= 1
    assert result.metadata["target_probability_smoothing_protocol_category"] == TARGET_PROBABILITY_SMOOTHING_CATEGORY
    assert result.metadata["target_probability_smoothing_uses_target_features"] is True
    assert result.metadata["target_probability_smoothing_uses_target_labels"] is False
    assert result.metadata["target_probability_smoothing_valid_for_strict_source_only"] is False
    assert result.metadata["target_probability_smoothing_valid_for_unlabeled_target_adaptation"] is True


def test_alpha_zero_returns_initial_probabilities() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    probabilities = np.asarray([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]], dtype=float)

    result = smooth_target_probabilities(features, probabilities, config={"alpha": 0.0})

    assert np.allclose(result.probabilities, probabilities)
    assert result.converged is True


def test_rbf_affinity_and_row_normalize_are_well_formed() -> None:
    features = np.asarray([[0.0], [0.1], [10.0]], dtype=float)
    affinity, gamma = rbf_affinity(features, gamma="auto", n_neighbors=1)
    transition = row_normalize(affinity)

    assert gamma > 0.0
    assert affinity.shape == (3, 3)
    assert np.allclose(affinity, affinity.T)
    assert np.allclose(transition.sum(axis=1), 1.0)


def test_smoothing_normalizes_unnormalized_initial_probabilities() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=float)
    probabilities = np.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=float)

    result = smooth_target_probabilities(features, probabilities, config={"alpha": 0.0})

    assert np.allclose(result.initial_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_config_aliases_and_validation() -> None:
    cfg = target_probability_smoothing_config(alpha="0.25", standardize="false", n_neighbors="2")
    assert cfg.alpha == 0.25
    assert cfg.standardize is False
    assert cfg.n_neighbors == 2

    with pytest.raises(ValueError, match="alpha"):
        target_probability_smoothing_config(alpha=1.5)

    with pytest.raises(ValueError, match="probabilities"):
        smooth_target_probabilities([[0.0], [1.0]], [[0.5, 0.5]])


def test_config_accepts_numpy_numeric_scalars() -> None:
    cfg = target_probability_smoothing_config(
        alpha=np.float64(0.25),
        gamma=np.float32(1.5),
        n_neighbors=np.int64(2),
        max_iter=np.int32(3),
        tol=np.float64(1e-4),
        epsilon=np.float32(1e-6),
        standardize=np.bool_(False),
    )

    assert cfg.alpha == 0.25
    assert cfg.gamma == pytest.approx(1.5)
    assert cfg.n_neighbors == 2
    assert cfg.max_iter == 3
    assert cfg.tol == pytest.approx(1e-4)
    assert cfg.epsilon == pytest.approx(1e-6)
    assert cfg.standardize is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", True),
        ("alpha", [0.5]),
        ("gamma", np.asarray([1.0])),
        ("n_neighbors", True),
        ("n_neighbors", [1]),
        ("max_iter", {"value": 2}),
        ("tol", (1e-3,)),
        ("epsilon", np.asarray(1e-12)),
    ],
)
def test_config_rejects_bool_and_array_like_numeric_controls(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        target_probability_smoothing_config(**{field: value})


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        smooth_target_probabilities(
            [[0.0], [1.0]],
            [[0.7, 0.3], [0.2, 0.8]],
            target_labels=[0, 1],  # type: ignore[call-arg]
        )
