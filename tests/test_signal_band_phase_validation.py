from __future__ import annotations

import numpy as np
import pytest

from neureptrace.signal.band import average_phases, circular_mean_phase


def test_circular_mean_phase_rejects_boolean_phase_values() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        circular_mean_phase(np.asarray([True, False]))

    with pytest.raises(ValueError, match="not boolean"):
        circular_mean_phase([0.0, np.bool_(True)])


def test_average_phases_rejects_boolean_phase_arrays() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        average_phases([np.asarray([0.0, 1.0]), np.asarray([True, False])])

    with pytest.raises(ValueError, match="not boolean"):
        average_phases([False, True])


def test_average_phases_still_averages_numeric_arrays() -> None:
    first = np.asarray([0.0, np.pi / 2.0])
    second = np.asarray([0.0, np.pi / 2.0])

    averaged = average_phases([first, second])

    assert np.allclose(averaged, first)
