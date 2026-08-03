from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.array_stats import column_stats


@pytest.mark.parametrize(
    "scale_floor",
    [
        0.5 + 2.0j,
        np.complex64(0.5 + 2.0j),
        np.complex128(0.5 + 2.0j),
        np.asarray(0.5 + 2.0j),
    ],
)
def test_column_stats_rejects_complex_scale_floor(scale_floor) -> None:
    with pytest.raises(ValueError, match="scale_floor must be positive and finite"):
        column_stats([[1.0], [2.0]], scale_floor=scale_floor)
