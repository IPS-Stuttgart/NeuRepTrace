from __future__ import annotations

from neureptrace.decoding import source_temperature


def test_source_temperature_rejects_indicator_rows() -> None:
    row = [[1 == 1, 1 == 0]]
    try:
        source_temperature.apply_temperature(row, temperature=1.0)
    except ValueError:
        return
    raise AssertionError("accepted indicator probability row")
