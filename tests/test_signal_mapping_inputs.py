from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.signal.band import (
    average_phases,
    circular_mean_phase,
    validate_band_hz,
    validate_signal_values,
    validate_time_axis,
)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: validate_time_axis({0.0: "first", 0.01: "second"}),
        lambda: validate_signal_values({1.0: "first", 2.0: "second"}),
        lambda: validate_band_hz({8.0: "low", 12.0: "high"}, 200.0),
        lambda: circular_mean_phase({0.0: "first", np.pi / 2.0: "second"}),
        lambda: average_phases({0.0: "first", np.pi / 2.0: "second"}),
    ],
)
def test_signal_helpers_reject_mapping_inputs(operation: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="ordered values, not mappings"):
        operation()


def test_signal_validation_rejects_mapping_rows_in_one_pass_input() -> None:
    rows = (row for row in ({0: 1.0, 1: 2.0}, [3.0, 4.0]))

    with pytest.raises(ValueError, match="ordered values, not mappings"):
        validate_signal_values(rows)


def test_signal_validation_keeps_one_pass_numeric_inputs() -> None:
    signal = validate_signal_values(value for value in (1.0, 2.0, 3.0))

    np.testing.assert_array_equal(signal, np.asarray([1.0, 2.0, 3.0]))
