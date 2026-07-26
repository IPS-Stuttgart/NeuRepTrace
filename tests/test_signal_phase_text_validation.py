from __future__ import annotations

import pytest

from neureptrace.signal.band import average_phases, circular_mean_phase


@pytest.mark.parametrize(
    "phases",
    ["12", b"12", bytearray(b"12"), memoryview(b"12")],
    ids=["str", "bytes", "bytearray", "memoryview"],
)
def test_circular_mean_phase_rejects_textual_and_binary_inputs(phases) -> None:
    with pytest.raises(ValueError, match="not text or binary data"):
        circular_mean_phase(phases)


@pytest.mark.parametrize(
    "phases",
    ["12", b"12", bytearray(b"12"), memoryview(b"12")],
    ids=["str", "bytes", "bytearray", "memoryview"],
)
def test_average_phases_rejects_textual_and_binary_collections(phases) -> None:
    with pytest.raises(ValueError, match="not text or binary data"):
        average_phases(phases)
