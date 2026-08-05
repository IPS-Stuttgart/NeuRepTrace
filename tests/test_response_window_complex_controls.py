from __future__ import annotations

import numpy as np
import pytest

from neureptrace.response_window_ensemble import (
    _normalize_response_times,
    _validate_optional_output_time,
    _validate_positive_finite_float,
    _validate_positive_integer,
    run_response_window_ensemble,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"response_times": (np.complex128(0.088 + 0.0j),)}, "Response-window times must be finite"),
        ({"output_time": np.complex64(0.184 + 0.1j)}, "output_time must be finite"),
        ({"weight_grid_step": np.complex128(0.1 + 1.0j)}, "weight_grid_step must be positive and finite"),
        ({"smoothing_stay_grid_size": np.asarray(200.0 + 1.0j)}, "smoothing_stay_grid_size must be a positive integer"),
    ],
)
def test_response_window_rejects_complex_scalar_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_response_window_ensemble([], **kwargs)


def test_response_window_accepts_real_numpy_scalar_controls() -> None:
    assert _normalize_response_times((np.float64(0.088),)) == (0.088,)
    assert _validate_optional_output_time(np.float32(0.184)) == pytest.approx(0.184)
    assert _validate_positive_finite_float(np.float64(0.1), name="weight_grid_step") == 0.1
    assert _validate_positive_integer(np.int64(200), name="smoothing_stay_grid_size") == 200
