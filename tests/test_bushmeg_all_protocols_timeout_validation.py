from __future__ import annotations

import math

import numpy as np
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols
from neureptrace.bushmeg_all_protocols import MethodProgress


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_all_protocol_timeout_validation_rejects_boolean_and_nonfinite_values(value) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        all_protocols._validate_timeout_seconds("method_timeout_seconds", value)


@pytest.mark.parametrize("field", ["method_timeout_seconds", "fold_timeout_seconds"])
def test_method_progress_rejects_invalid_timeout_controls(tmp_path, field: str) -> None:
    kwargs = {field: math.nan}

    with pytest.raises(ValueError, match=f"{field} must be a positive finite number"):
        MethodProgress(tmp_path / "methods" / "invalid_timeout", method="invalid_timeout", **kwargs)


def test_all_protocol_timeout_validation_still_accepts_positive_finite_values() -> None:
    assert all_protocols._validate_timeout_seconds("fold_timeout_seconds", "0.25") == pytest.approx(0.25)
