from __future__ import annotations

import numpy as np
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        np.asarray(True),
        0,
        -1,
        1.5,
        "1.5",
        float("nan"),
        float("inf"),
        np.asarray([1]),
    ],
)
def test_validate_positive_limit_rejects_bool_fractional_and_nonfinite_values(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        all_protocols._validate_positive_limit("participant_limit", value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (np.int64(2), 2),
        (3.0, 3),
        ("4", 4),
        (np.asarray(5), 5),
    ],
)
def test_validate_positive_limit_accepts_integral_scalar_values(value, expected: int) -> None:
    assert all_protocols._validate_positive_limit("participant_limit", value) == expected


def test_limited_participant_ids_rejects_boolean_limit() -> None:
    with pytest.raises(ValueError, match="participant_limit must be a positive integer"):
        all_protocols._limited_participant_ids("1,2,3", participant_limit=True)


def test_apply_window_limit_rejects_fractional_limit() -> None:
    config = {"preprocessing": {"window_centers": [0.1, 0.2, 0.3]}}

    with pytest.raises(ValueError, match="window_limit must be a positive integer"):
        all_protocols._apply_window_limit(config, 1.5)
