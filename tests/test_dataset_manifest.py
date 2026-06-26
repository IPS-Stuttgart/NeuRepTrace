from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from neureptrace.dataset_manifest import manifest_from_dataset_config, write_manifest_from_dataset_config


def test_manifest_from_dataset_config_expands_participants_runs_and_columns():
    frame = manifest_from_dataset_config(
        {
            "dataset": {
                "root": "D:/Uni-Data/Bush_MEG-Data/MEG-Data",
                "input_format": "fieldtrip-mat",
                "subject_template": "Part{participant}",
            },
            "participants": {"include": "9-11", "exclude": [10]},
            "files": {
                "main": "Part{participant}Data.mat",
                "cue": {"pattern": "Part{participant}CueData.mat"},
            },
            "metadata": {"label_column": "condition", "group_column": "trial"},
            "preprocessing": {"picks": "data", "window_ms": 100, "baseline_window": [-0.5, 0.0]},
            "decoding": {"decoder": "linear_svm", "emission_mode": "calibrated"},
            "fieldtrip": {"root_path": ["data", 0], "label_base": 1, "trim_overlong_labels": True},
            "runs": [
                {"name": "main", "file": "main"},
                {"name": "cue", "file": "cue", "columns": {"variant": "heldout_cue"}},
            ],
        }
    )

    assert frame["subject"].tolist() == ["Part9", "Part9", "Part11", "Part11"]
    assert frame["variant"].tolist() == ["main", "heldout_cue", "main", "heldout_cue"]
    assert frame["epochs"].tolist()[0] == "D:/Uni-Data/Bush_MEG-Data/MEG-Data/Part9Data.mat"
    assert frame["epochs"].tolist()[1] == "D:/Uni-Data/Bush_MEG-Data/MEG-Data/Part9CueData.mat"
    assert frame["input"].equals(frame["epochs"])
    assert frame["input_format"].tolist() == ["fieldtrip-mat"] * 4
    assert frame["label_column"].tolist() == ["condition"] * 4
    assert frame["fieldtrip_root_path"].tolist() == ["data,0"] * 4
    assert frame["fieldtrip_trim_overlong_labels"].tolist() == ["true"] * 4
    assert frame["baseline_window"].tolist() == ["-0.5,0.0"] * 4


@pytest.mark.parametrize(
    "participants",
    [
        True,
        {"include": True},
        {"include": [1, False]},
        {"include": ["true"]},
        {"include": [1], "exclude": [False]},
    ],
)
def test_manifest_from_dataset_config_rejects_boolean_participants(participants):
    config = {
        "dataset": {"root": "data"},
        "participants": participants,
        "files": {"main": "sub-{participant}_epo.fif"},
    }

    with pytest.raises(ValueError, match="boolean|booleans"):
        manifest_from_dataset_config(config)


@pytest.mark.parametrize(
    "participants",
    [
        {"include": {"subject": 1}},
        {"include": [1, {"subject": 2}]},
    ],
)
def test_manifest_from_dataset_config_rejects_mapping_participants(participants):
    config = {
        "dataset": {"root": "data"},
        "participants": participants,
        "files": {"main": "sub-{participant}_epo.fif"},
    }

    with pytest.raises(ValueError, match="mapping|mappings"):
        manifest_from_dataset_config(config)


def test_write_manifest_from_json_config_can_select_run(tmp_path: Path):
    config_path = tmp_path / "dataset.json"
    out_path = tmp_path / "manifest.csv"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"root": "data", "input_format": "mne-epochs"},
                "participants": {"include": [1, "3-4"]},
                "files": {"main": "sub-{participant}_epo.fif", "cue": "sub-{participant}_cue-epo.fif"},
                "metadata": {"label_column": "condition"},
                "runs": [{"name": "main", "file": "main"}, {"name": "cue", "file": "cue"}],
            }
        ),
        encoding="utf-8",
    )

    frame = write_manifest_from_dataset_config(config_path, out_path, run_names=("cue",))

    assert out_path.exists()
    assert frame["subject"].tolist() == ["Part1", "Part3", "Part4"]
    assert frame["epochs"].tolist() == ["data/sub-1_cue-epo.fif", "data/sub-3_cue-epo.fif", "data/sub-4_cue-epo.fif"]
    assert pd.read_csv(out_path)["variant"].tolist() == ["cue", "cue", "cue"]
