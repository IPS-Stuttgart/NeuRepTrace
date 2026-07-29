from __future__ import annotations

from collections.abc import Callable

import pytest

from neureptrace.metrics import validate_probability_inputs, weighted_brier_score_multiclass


def _validate_unweighted(probabilities: object) -> object:
    return validate_probability_inputs(probabilities, [0, 1])


def _validate_weighted(probabilities: object) -> object:
    return weighted_brier_score_multiclass(probabilities, [0, 1], [1.0, 1.0])


@pytest.mark.parametrize(
    "validator",
    [_validate_unweighted, _validate_weighted],
    ids=["unweighted", "weighted"],
)
def test_probability_metrics_normalize_nonnumeric_cell_type_errors(validator: Callable[[object], object]) -> None:
    probabilities = [[0.8, object()], [0.3, 0.7]]

    with pytest.raises(ValueError, match="numeric probability values"):
        validator(probabilities)
