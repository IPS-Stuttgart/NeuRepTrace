from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace._fieldtrip_sampleinfo_validation_patch import parse_bool_config
from neureptrace.io import fieldtrip_mat


class _DummyEpochs:
    times = np.array([0.0, 1.0], dtype=float)
    ch_names = ["MEG001"]

    def get_data(self, *, copy: bool = True) -> np.ndarray:
        return np.zeros((1, 1, 2), dtype=float)


def test_parse_bool_config_respects_false_strings() -> None:
    assert parse_bool_config("false", name="flag") is False
    assert parse_bool_config("off", name="flag") is False
    assert parse_bool_config("0", name="flag") is False
    assert parse_bool_config("true", name="flag") is True
    assert parse_bool_config("yes", name="flag") is True

    with pytest.raises(ValueError, match="flag must be a boolean value"):
        parse_bool_config("sometimes", name="flag")


def test_fieldtrip_metadata_column_optional_false_string_is_false() -> None:
    columns = fieldtrip_mat._metadata_columns_from_config(
        {"metadata": {"columns": [{"name": "condition", "index": 0, "optional": "false"}]}}
    )

    assert len(columns) == 1
    assert columns[0].optional is False


def test_fieldtrip_epoch_loader_validation_false_strings_are_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def fake_load_fieldtrip_mat(path: str | Path, spec: fieldtrip_mat.FieldTripMatSpec):
        captured["path"] = path
        captured["spec"] = spec
        return _DummyEpochs(), pd.DataFrame({"trial": [0]})

    monkeypatch.setattr(fieldtrip_mat, "load_fieldtrip_mat", fake_load_fieldtrip_mat)

    dataset = fieldtrip_mat.load_fieldtrip_mat_epochs(
        tmp_path / "example.mat",
        {
            "validation": {
                "trim_channel_labels_to_data": "false",
                "require_equal_trial_time_lengths": "false",
                "require_trialinfo_rows_equal_trials": "false",
            }
        },
    )

    spec = captured["spec"]
    assert spec.trim_channel_labels_to_data is False
    assert spec.require_equal_trial_time_lengths is False
    assert spec.require_trialinfo_rows_equal_trials is False
    assert dataset.data.shape == (1, 1, 2)
