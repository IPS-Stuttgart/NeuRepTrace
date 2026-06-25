from __future__ import annotations

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_bool


@pytest.mark.parametrize(("value", "expected"), [(1, True), (0, False)])
def test_legacy_config_workflow_accepts_integer_boolean_flags(value, expected) -> None:
    assert _as_bool(value) is expected


@pytest.mark.parametrize("value", [2, -1, 0.5])
def test_legacy_config_workflow_rejects_ambiguous_numeric_boolean_flags(value) -> None:
    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)
