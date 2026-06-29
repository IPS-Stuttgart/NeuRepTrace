from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import (
    SOURCE_TEMPERATURE_CATEGORY,
    apply_temperature,
    fit_source_temperature_scaling,
    negative_log_likelihood,
    source_temperature_config,
)


def test_fit_source_temperature_scaling_selects_source_temperature_and_scales_test_rows() -> None:
    source_probabilities = np.asarray(
        [
            [0.70, 0.30],
            [0.65, 0.35],
            [0.35, 0.65],
            [0.30, 0.70],
        ],
        dtype=float,
    )
    source_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    test_probabilities = np.asarray([[0.60, 0.40], [0.20, 0.80]], dtype=float)

    result = fit_source_temperature_scaling(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        test_probabilities=test_probabilities,
        classes=["a", "b"],
        config={"temperatures": [0.5, 1.0, 2.0]},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.temperature in {0.5, 1.0, 2.0}
    assert set(result.source_losses) == {0.5, 1.0, 2.0}
    assert result.metadata["source_temperature_protocol_category"] == SOURCE_TEMPERATURE_CATEGORY
    assert result.metadata["source_temperature_uses_test_probabilities_for_fitting"] is False
    assert result.metadata["source_temperature_uses_test_labels"] is False
    assert result.metadata["source_temperature_valid_for_strict_source_only"] is True


def test_apply_temperature_sharpens_and_smooths_probabilities() -> None:
    probabilities = np.asarray([[0.75, 0.25]], dtype=float)

    sharpened = apply_temperature(probabilities, temperature=0.5)
    smoothed = apply_temperature(probabilities, temperature=2.0)

    assert sharpened[0, 0] > probabilities[0, 0]
    assert smoothed[0, 0] < probabilities[0, 0]
    assert np.allclose(sharpened.sum(axis=1), 1.0)
    assert np.allclose(smoothed.sum(axis=1), 1.0)


def test_negative_log_likelihood_validates_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        negative_log_likelihood([[0.5, 0.5]], [2])


def test_source_temperature_config_parses_grid() -> None:
    cfg = source_temperature_config(temperatures="0.5,1,2", epsilon="1e-9")

    assert cfg.temperatures == (0.5, 1.0, 2.0)
    assert np.isclose(cfg.epsilon, 1e-9)

    with pytest.raises(ValueError, match="temperatures"):
        source_temperature_config(temperatures=[])


def test_source_temperature_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="class width"):
        fit_source_temperature_scaling(
            source_probabilities=[[0.5, 0.5]],
            source_labels=[0],
            test_probabilities=[[0.3, 0.3, 0.4]],
        )


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_temperature_scaling(
            source_probabilities=[[0.5, 0.5], [0.4, 0.6]],
            source_labels=[0, 1],
            test_probabilities=[[0.5, 0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
