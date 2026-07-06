from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import fit_source_temperature_scaling


def test_source_temperature_infers_integer_classes_in_probability_column_order() -> None:
    source_probabilities = np.asarray(
        [
            [0.05, 0.95],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.85, 0.15],
        ],
        dtype=float,
    )

    result = fit_source_temperature_scaling(
        source_probabilities=source_probabilities,
        source_labels=np.asarray([1, 0, 1, 0], dtype=np.int64),
        test_probabilities=[[0.40, 0.60]],
        config={"temperatures": [1.0]},
    )

    expected_loss = -np.mean(np.log([0.95, 0.90, 0.90, 0.85]))
    assert np.isclose(result.source_losses[1.0], expected_loss)


def test_source_temperature_allows_missing_integer_classes_when_width_is_known() -> None:
    source_probabilities = np.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.05, 0.90, 0.05],
            [0.75, 0.20, 0.05],
        ],
        dtype=float,
    )

    result = fit_source_temperature_scaling(
        source_probabilities=source_probabilities,
        source_labels=[0, 1, 0],
        test_probabilities=[[0.20, 0.30, 0.50]],
        config={"temperatures": [1.0]},
    )

    expected_loss = -np.mean(np.log([0.80, 0.90, 0.75]))
    assert np.isclose(result.source_losses[1.0], expected_loss)
    assert result.metadata["source_temperature_n_classes"] == 3


def test_source_temperature_rejects_out_of_range_inferred_integer_labels() -> None:
    with pytest.raises(ValueError, match="integer source_labels"):
        fit_source_temperature_scaling(
            source_probabilities=[[0.60, 0.40], [0.50, 0.50]],
            source_labels=[1, 2],
            test_probabilities=[[0.50, 0.50]],
        )
