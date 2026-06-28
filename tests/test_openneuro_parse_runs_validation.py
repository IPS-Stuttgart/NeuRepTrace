from __future__ import annotations

import pytest

from neureptrace.openneuro_meg import DATASET_SPECS, expected_relative_files, parse_runs


def test_openneuro_parse_runs_rejects_empty_selections() -> None:
    spec = DATASET_SPECS["ds000117"]

    for runs in ("", " , ", []):
        with pytest.raises(ValueError, match="at least one run"):
            parse_runs(spec, runs)


def test_openneuro_parse_runs_rejects_boolean_selectors() -> None:
    spec = DATASET_SPECS["ds000117"]

    for runs in (True, False, "true", "false", "yes", "no"):
        with pytest.raises(ValueError, match="not booleans"):
            parse_runs(spec, runs)


def test_openneuro_parse_runs_accepts_numeric_zero_run_id() -> None:
    assert parse_runs(DATASET_SPECS["ds006629"], 0) == ("0",)
    assert expected_relative_files("ds006629", subjects="1", runs=0) == [
        "sub-01/meg/sub-01_task-MMNHCS_run-0_meg.fif",
        "sub-01/meg/sub-01_task-MMNHCS_run-0_events.tsv",
    ]
