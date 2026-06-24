from __future__ import annotations

import numpy as np
import pytest

from neureptrace import response_window_ensemble


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_response_window_rejects_boolean_response_times(value: object) -> None:
    with pytest.raises(ValueError, match="Response-window times must be finite"):
        response_window_ensemble._normalize_response_times((value,))


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_response_window_rejects_boolean_output_time(value: object) -> None:
    with pytest.raises(ValueError, match="output_time must be finite"):
        response_window_ensemble._validate_optional_output_time(value)
