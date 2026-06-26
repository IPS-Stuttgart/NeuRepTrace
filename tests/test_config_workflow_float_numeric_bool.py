from __future__ import annotations

import math

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_bool


@pytest.mark.parametrize(("value", "expected"), [(1.0, True), (0.0, False)])
def test_config_workflow_accepts_float_numeric_boolean_flags(value: float, expected: bool) -> None:
    assert _as_bool(value) is expected


@pytest.mark.parametrize("value", [0.5, -1.0, 2.0, math.inf, math.nan])
def test_config_workflow_rejects_ambiguous_float_numeric_boolean_flags(value: float) -> None:
    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)
