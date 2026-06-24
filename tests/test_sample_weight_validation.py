from __future__ import annotations

import numpy as np

from neureptrace.metrics import validate_sample_weight


def _flag(value: int) -> object:
    return np.asarray(value == 1).item()


def check_validate_sample_weight_requires_numeric_scalars() -> None:
    try:
        validate_sample_weight([_flag(1), 1.0], 2)
    except ValueError as exc:
        assert "numeric weights" in str(exc)
    else:
        raise AssertionError("expected sample-weight validation to fail")


test_validate_sample_weight_requires_numeric_scalars = check_validate_sample_weight_requires_numeric_scalars
