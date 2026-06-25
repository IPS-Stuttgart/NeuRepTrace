from __future__ import annotations

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
    ],
)
def test_config_workflow_as_bool_accepts_numeric_flags(value, expected) -> None:
    assert _as_bool(value) is expected


@pytest.mark.parametrize("value", [2, -1, 0.5, float("nan")])
def test_config_workflow_as_bool_rejects_ambiguous_numeric_flags(value) -> None:
    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)
