from __future__ import annotations

import neureptrace  # noqa: F401
from neureptrace.openneuro_alignment_compare import _as_bool


def test_alignment_compare_as_bool_accepts_sequence_values() -> None:
    assert _as_bool(["yes"]) is True
    assert _as_bool(["no"]) is False
    assert _as_bool([]) is False
