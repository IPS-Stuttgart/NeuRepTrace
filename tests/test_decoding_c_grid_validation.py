from __future__ import annotations

import pytest

from neureptrace.decoding import parse_c_grid


def test_parse_c_grid_rejects_non_finite_values() -> None:
    for values in ("nan", "inf", "0.1,nan", [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="positive finite"):
            parse_c_grid(values)


def test_parse_c_grid_still_accepts_positive_finite_values() -> None:
    assert parse_c_grid("0.1,1,10") == (0.1, 1.0, 10.0)
