from __future__ import annotations

from pathlib import Path

from neureptrace.dataset_config import (
    apply_overrides,
    effective_config,
    iter_dataset_files,
    parse_participant_ids,
    validate_dataset_config,
)
from neureptrace.decode_from_config import _resolve_output
from neureptrace.io.dataset import EpochDataset

import numpy as np
import pandas as pd


def test_parse_participant_ids_supports_ranges_and_lists():
    assert parse_participant_ids("1-3,6,sub-09") == [1, 2, 3, 6, "sub-09"]
    assert parse_participant_ids(["2-1", 5]) == [2, 1, 5]


def test_apply_overrides_updates_nested_values():
    config = {"dataset": {"type": "fieldtrip_mat"}, "participants": {"ids": "1-2"}}

    updated = apply_overrides(
        config,
        ["participants.ids=[2,3]", "decoding.classifier=lda", "decoding.tune_hyperparameters=true"],
    )

    assert updated["participants"]["ids"] == [2, 3]
    assert updated["decoding"]["classifier"] == "lda"
    assert updated["decoding"]["tune_hyperparameters"] is True
    assert config["participants"]["ids"] == "1-2"


def test_validate_fieldtrip_config_and_iter_files(tmp_path: Path):
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": "data",
            "participant_file": "Part{participant}Data.mat",
        },
        "participants": {"ids": "2-3"},
        "decoding": {"label_column": "stimulus_class"},
    }

    assert validate_dataset_config(config, base_dir=tmp_path) == []
    assert iter_dataset_files(config, base_dir=tmp_path) == [
        tmp_path / "data" / "Part2Data.mat",
        tmp_path / "data" / "Part3Data.mat",
    ]


def test_iter_dataset_files_includes_mne_metadata_csv(tmp_path: Path):
    config = {
        "dataset": {
            "type": "mne_epochs",
            "root": "data",
            "epochs": "sub-01_epo.fif",
            "metadata_csv": "sub-01_events.csv",
        },
        "decoding": {"label_column": "condition"},
    }

    assert iter_dataset_files(config, base_dir=tmp_path) == [
        tmp_path / "data" / "sub-01_epo.fif",
        tmp_path / "data" / "sub-01_events.csv",
    ]


def test_effective_config_expands_participants_and_files(tmp_path: Path):
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": "data",
            "participant_file": "Part{participant}Data.mat",
        },
        "participants": {"ids": "2-3"},
        "decoding": {"label_column": "stimulus_class"},
    }

    rendered = effective_config(config, base_dir=tmp_path)

    assert rendered["participants"]["expanded_ids"] == [2, 3]
    assert rendered["resolved_input_files"] == [
        str(tmp_path / "data" / "Part2Data.mat"),
        str(tmp_path / "data" / "Part3Data.mat"),
    ]
    assert rendered["effective_config_hash"].startswith("sha256:")


def test_decode_outputs_are_cwd_relative_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "dataset": {"name": "demo", "type": "mne_epochs", "epochs": "epochs.fif"},
        "decoding": {"label_column": "condition"},
        "outputs": {"summary_csv": "results/{dataset}/summary.csv"},
    }

    assert _resolve_output(config, config_dir=tmp_path / "configs", key="summary_csv") == tmp_path / "results" / "demo" / "summary.csv"


def test_decode_outputs_can_use_output_base_dir(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "dataset": {"name": "demo", "type": "mne_epochs", "epochs": "epochs.fif"},
        "decoding": {"label_column": "condition"},
        "outputs": {"base_dir": "results/{dataset}", "summary_csv": "summary.csv"},
    }

    assert _resolve_output(config, config_dir=tmp_path / "configs", key="summary_csv") == tmp_path / "results" / "demo" / "summary.csv"


def test_epoch_dataset_concatenate_supports_channel_intersection():
    first = EpochDataset(
        data=np.ones((1, 3, 2)),
        times=np.array([0.0, 0.1]),
        channel_names=["A", "B", "C"],
        metadata=pd.DataFrame({"split": ["main"]}),
    )
    second = EpochDataset(
        data=np.ones((1, 3, 2)) * 2,
        times=np.array([0.0, 0.1]),
        channel_names=["B", "C", "D"],
        metadata=pd.DataFrame({"split": ["cue"]}),
    )

    merged = EpochDataset.concatenate([first, second], channel_policy="intersection")

    assert merged.channel_names == ["B", "C"]
    assert merged.data.shape == (2, 2, 2)
    assert merged.provenance["dropped_channels"] == ["A", "D"]
