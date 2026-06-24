from __future__ import annotations

import numpy as np

from neureptrace.metrics import validate_sample_weight


def _flag(value: int) -> object:
    return np.asarray(value == 1).item()


def _expect_numeric_weight_error(callable_object) -> None:
    try:
        callable_object()
    except ValueError as exc:
        assert "numeric weights" in str(exc)
    else:
        raise AssertionError("expected sample-weight validation to fail")


def check_validate_sample_weight_requires_numeric_scalars() -> None:
    examples = [
        [_flag(1), 1.0],
        np.asarray([_flag(1), _flag(0)]),
        np.asarray([_flag(1), 1.0], dtype=object),
    ]
    for weights in examples:
        _expect_numeric_weight_error(lambda weights=weights: validate_sample_weight(weights, 2))


test_validate_sample_weight_requires_numeric_scalars = check_validate_sample_weight_requires_numeric_scalars
