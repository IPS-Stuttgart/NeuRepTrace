from __future__ import annotations

import pytest

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.openneuro_meg import DATASET_SPECS, parse_runs


@pytest.mark.parametrize("runs", ["", "   ", ", ,", []])
def test_parse_runs_rejects_empty_run_selections(runs) -> None:
    with pytest.raises(ValueError, match="run selection must include at least one run"):
        parse_runs(DATASET_SPECS["ds006629"], runs)


def test_parse_runs_still_accepts_default_all_selection() -> None:
    assert parse_runs(DATASET_SPECS["ds006629"], "all") == ("0",)
