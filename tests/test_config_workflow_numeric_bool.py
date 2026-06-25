from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_bool


def test_config_workflow_accepts_numeric_boolean_flags() -> None:
    assert _as_bool(1) is True
    assert _as_bool(0) is False


@pytest.mark.parametrize("value", [2, -1, 0.5])
def test_config_workflow_rejects_ambiguous_numeric_boolean_flags(value: float) -> None:
    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)
