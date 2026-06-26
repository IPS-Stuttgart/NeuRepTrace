from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.config_workflow import DatasetConfigError, _decode_kwargs


def _minimal_config(*, tuning_enabled, label_shuffle_control):
    return {
        "dataset": {"epochs": "subject-01-epo.fif"},
        "decoding": {
            "label_column": "stimulus",
            "label_shuffle_control": label_shuffle_control,
        },
        "outputs": {"metrics_csv": "results/metrics.csv"},
        "tuning": {"enabled": tuning_enabled},
    }


def test_config_workflow_accepts_integer_boolean_flags(tmp_path: Path) -> None:
    kwargs = _decode_kwargs(
        _minimal_config(tuning_enabled=0, label_shuffle_control=1),
        config_path=tmp_path / "workflow.json",
    )

    assert kwargs["tune_hyperparameters"] is False
    assert kwargs["label_shuffle_control"] is True


def test_config_workflow_rejects_ambiguous_integer_booleans(tmp_path: Path) -> None:
    with pytest.raises(DatasetConfigError, match="Cannot interpret 2 as a boolean"):
        _decode_kwargs(
            _minimal_config(tuning_enabled=2, label_shuffle_control=False),
            config_path=tmp_path / "workflow.json",
        )
