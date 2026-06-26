from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.dataset_config import (
    ConfigValidationError,
    _fieldtrip_file_specs,
    apply_overrides,
    effective_config,
    iter_dataset_files,
    load_epoch_dataset_from_config,
    parse_participant_ids,
    validate_dataset_config,
)
from neureptrace.decode_from_config import _resolve_output
from neureptrace.io.dataset import EpochDataset


def test_parse_participant_ids_supports_ranges_and_lists():
    assert parse_participant_ids("1-3,6,sub-09") == [1, 2, 3, 6, "sub-09"]
    assert parse_participant_ids(["2-1", 5]) == [2, 1, 5]


def test_parse_participant_ids_rejects_boolean_values():
    for value in (True, False, [1, True], [False], ["true"], ["no"]):
        with pytest.raises(ValueError, match="booleans|boolean"):
            parse_participant_ids(value)


def test_parse_participant_ids_rejects_mapping_values():
    for value in ({"subject": 1}, [1, {"subject": 2}]):
        with pytest.raises(ValueError, match="mapping|mappings"):
            parse_participant_ids(value)


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


def test_fieldtrip_dataset_files_accepts_single_path_string(tmp_path: Path):
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": "data",
            "files": "Part10Data.mat",
        },
        "decoding": {"label_column": "stimulus_class"},
    }

    assert validate_dataset_config(config, base_dir=tmp_path) == []
    expected = tmp_path / "data" / "Part10Data.mat"
    assert iter_dataset_files(config, base_dir=tmp_path) == [expected]
    assert _fieldtrip_file_specs(config, base_dir=tmp_path) == [(expected, {})]


def test_fieldtrip_dataset_files_accepts_single_mapping_with_metadata(tmp_path: Path):
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": "data",
            "files": {"path": "Part11Data.mat", "split": "calibration"},
        },
        "decoding": {"label_column": "stimulus_class"},
    }

    assert validate_dataset_config(config, base_dir=tmp_path) == []
    expected = tmp_path / "data" / "Part11Data.mat"
    assert iter_dataset_files(config, base_dir=tmp_path) == [expected]
    assert _fieldtrip_file_specs(config, base_dir=tmp_path) == [(expected, {"split": "calibration"})]


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


def test_iter_dataset_files_expands_mne_epochs_template(tmp_path: Path):
    config = {
        "dataset": {
            "type": "mne_epochs",
            "root": "staged",
            "epochs_files": {"template": "ds004276/sub-{subject03d}_epo.fif"},
        },
        "participants": {"ids": "1-2"},
        "decoding": {"label_column": "condition"},
    }

    assert iter_dataset_files(config, base_dir=tmp_path) == [
        tmp_path / "staged" / "ds004276" / "sub-001_epo.fif",
        tmp_path / "staged" / "ds004276" / "sub-002_epo.fif",
    ]


def test_mne_epochs_template_requires_participants_ids(tmp_path: Path):
    config = {
        "dataset": {
            "type": "mne_epochs",
            "root": "staged",
            "epochs_files": {"template": "ds004276/sub-{subject03d}_epo.fif"},
        },
        "decoding": {"label_column": "condition"},
    }

    with pytest.raises(ConfigValidationError, match="participants.ids"):
        validate_dataset_config(config, base_dir=tmp_path)
    with pytest.raises(ConfigValidationError, match="participants.ids"):
        iter_dataset_files(config, base_dir=tmp_path)


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


def test_load_mne_epochs_dataset_concatenates_template_files(tmp_path: Path):
    import mne

    staged = tmp_path / "staged" / "demo"
    staged.mkdir(parents=True)
    for subject, offset in [(1, 0.0), (2, 1.0)]:
        info = mne.create_info(["MEG001", "MEG002"], sfreq=10.0, ch_types="mag")
        metadata = pd.DataFrame({"subject": [f"sub-{subject:03d}"], "condition": ["a" if subject == 1 else "b"]})
        epochs = mne.EpochsArray(
            np.ones((1, 2, 3)) + offset,
            info,
            events=np.array([[subject, 0, 1]]),
            event_id={"event": 1},
            tmin=-0.1,
            metadata=metadata,
            verbose="error",
        )
        epochs.save(staged / f"sub-{subject:03d}_epo.fif", overwrite=True)

    dataset = load_epoch_dataset_from_config(
        {
            "dataset": {
                "type": "mne_epochs",
                "root": "staged",
                "epochs_files": {"template": "demo/sub-{subject03d}_epo.fif"},
            },
            "participants": {"ids": "1-2"},
            "decoding": {"label_column": "condition"},
        },
        base_dir=tmp_path,
        check_files=True,
    )

    assert dataset.data.shape == (2, 2, 3)
    assert dataset.metadata["subject"].tolist() == ["sub-001", "sub-002"]
