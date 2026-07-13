from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_temperature_scaling import (
    TARGET_TEMPERATURE_CATEGORY,
    apply_temperature_to_probabilities,
    fit_target_temperature_scaling,
    negative_log_likelihood,
    target_temperature_config,
)


def test_target_temperature_scaling_uses_calibration_labels_only() -> None:
    cal_prob = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]], dtype=float)
    cal_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    score_prob = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)

    result = fit_target_temperature_scaling(
        calibration_probabilities=cal_prob,
        calibration_labels=cal_labels,
        probabilities=score_prob,
        classes=["a", "b"],
        config={"temperature_grid": [0.5, 1.0, 2.0]},
    )

    assert result.probabilities.shape == score_prob.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.temperature in {0.5, 1.0, 2.0}
    assert result.metadata["target_temperature_protocol_category"] == TARGET_TEMPERATURE_CATEGORY
    assert result.metadata["target_temperature_uses_target_calibration_labels"] is True
    assert result.metadata["target_temperature_uses_scored_target_labels"] is False
    assert result.metadata["target_temperature_valid_for_supervised_calibration"] is True


def test_target_temperature_scaling_preserves_composite_class_labels() -> None:
    classes = [("semantic", "animate"), ("semantic", "inanimate")]
    result = fit_target_temperature_scaling(
        calibration_probabilities=[[0.8, 0.2], [0.3, 0.7]],
        calibration_labels=classes,
        probabilities=[[0.6, 0.4], [0.45, 0.55]],
        classes=classes,
        config={"temperature_grid": [1.0]},
    )

    assert result.classes.tolist() == classes
    assert result.metadata["target_temperature_n_classes"] == 2
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_apply_temperature_changes_confidence() -> None:
    probabilities = np.asarray([[0.9, 0.1]], dtype=float)

    cold = apply_temperature_to_probabilities(probabilities, temperature=0.5)
    hot = apply_temperature_to_probabilities(probabilities, temperature=2.0)

    assert cold[0, 0] > probabilities[0, 0]
    assert hot[0, 0] < probabilities[0, 0]
    assert np.allclose(cold.sum(axis=1), 1.0)
    assert np.allclose(hot.sum(axis=1), 1.0)


def test_apply_temperature_rejects_zero_mass_rows() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        apply_temperature_to_probabilities([[0.0, 0.0]], temperature=1.0)


def test_target_temperature_normalizes_extreme_finite_rows_without_overflow() -> None:
    probabilities = np.asarray(
        [
            [1.0e308, 1.0e308],
            [1.0e308, 1.0e307],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        transformed = apply_temperature_to_probabilities(probabilities, temperature=1.0)
        fitted = fit_target_temperature_scaling(
            calibration_probabilities=probabilities,
            calibration_labels=[0, 1],
            probabilities=probabilities,
            classes=[0, 1],
            config={"temperature_grid": [1.0]},
        )
        nll = negative_log_likelihood(probabilities, [0, 1], classes=[0, 1])

    expected = np.asarray([[0.5, 0.5], [10.0 / 11.0, 1.0 / 11.0]])
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(fitted.probabilities, expected, rtol=1.0e-6)
    assert fitted.temperature == 1.0
    assert np.isfinite(nll)
    assert np.isfinite(fitted.calibration_nll_by_temperature[1.0])


def test_negative_log_likelihood_matches_manual_value() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=float)

    nll = negative_log_likelihood(probabilities, ["a", "b"], classes=["a", "b"])

    assert np.isclose(nll, -np.mean(np.log([0.8, 0.75])))


def test_negative_log_likelihood_accepts_numpy_composite_labels() -> None:
    classes = np.asarray([("left", 1), ("right", 2)], dtype=object)
    labels = np.asarray([("left", 1), ("right", 2)], dtype=object)

    nll = negative_log_likelihood([[0.9, 0.1], [0.25, 0.75]], labels, classes=classes)

    assert np.isclose(nll, -np.mean(np.log([0.9, 0.75])))


def test_target_temperature_config_parses_grid_string() -> None:
    cfg = target_temperature_config(temperature_grid="2,1,0.5")

    assert cfg.temperature_grid == (0.5, 1.0, 2.0)

    with pytest.raises(ValueError, match="temperature_grid"):
        target_temperature_config(temperature_grid="")


def test_scored_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_target_temperature_scaling(
            calibration_probabilities=[[0.5, 0.5]],
            calibration_labels=["a"],
            probabilities=[[0.5, 0.5]],
            classes=["a", "b"],
            scored_target_labels=["a"],  # type: ignore[call-arg]
        )
