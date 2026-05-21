from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from neureptrace.dataset_spec import expand_manifest, load_dataset_spec, load_split_dataset, parse_subjects, resolve_split, validate_dataset_spec
from neureptrace.datasets.spec import (
    build_dataset_file_table as build_dataset_file_table_v1,
    expand_participant_ids,
    validate_dataset_spec as validate_dataset_spec_v1,
)


def _write_spec(path: Path, root: Path) -> Path:
    payload = {
        "schema_version": "neureptrace.dataset.v1",
        "dataset_id": "toy",
        "root": {"path": str(root)},
        "subjects": {"include": "1-2,4"},
        "splits": {
            "epochs": {
                "loader": "mne_epochs",
                "path_template": "sub-{subject02d}_epo.fif",
                "metadata_template": "sub-{subject02d}_events.csv",
                "label_column": "condition",
                "group_column": "run",
                "manifest": {"n_splits": 3},
            }
        },
        "workflows": {
            "benchmark": {
                "split": "epochs",
                "manifest": {"decoder": "logistic", "window_ms": 20},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_subjects_ranges_lists_and_exclusions() -> None:
    assert parse_subjects("1-3,5") == ("1", "2", "3", "5")
    assert parse_subjects(["sub-01", 2, "4-5"]) == ("sub-01", "2", "4", "5")
    assert parse_subjects({"include": "1-4", "exclude": "2,4"}) == ("1", "3")


def test_json_spec_resolves_paths_and_expands_manifest(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "sub-01_epo.fif").write_text("placeholder", encoding="utf-8")
    pd.DataFrame({"condition": ["a"], "run": [1]}).to_csv(root / "sub-01_events.csv", index=False)
    spec = load_dataset_spec(_write_spec(tmp_path / "dataset.json", root))

    resolved = resolve_split(spec, "epochs", "1")
    assert resolved.data_path == (root / "sub-01_epo.fif").resolve()
    assert resolved.metadata_path == (root / "sub-01_events.csv").resolve()
    assert resolved.label_column == "condition"

    inventory = validate_dataset_spec(spec, subjects=("1",), require_files=True)
    assert inventory.loc[0, "data_exists"]

    manifest = expand_manifest(spec, workflow="benchmark", subjects=("1",))
    row = manifest.iloc[0].to_dict()
    assert row["epochs"] == str((root / "sub-01_epo.fif").resolve())
    assert row["metadata_csv"] == str((root / "sub-01_events.csv").resolve())
    assert row["decoder"] == "logistic"
    assert row["n_splits"] == 3


def test_yaml_pymegdec_style_config_loads() -> None:
    spec = load_dataset_spec(Path("examples/configs/pymegdec_bushmeg.yml"))
    assert spec.dataset_id == "bushmeg"
    assert spec.subjects[:4] == ("1", "2", "3", "4")
    assert spec.splits["main"].path_template == "Part{subject}Data.mat"
    assert spec.splits["cue"].path_template == "Part{subject}CueData.mat"


def test_validate_requires_existing_files(tmp_path: Path) -> None:
    spec = load_dataset_spec(_write_spec(tmp_path / "dataset.json", tmp_path / "missing-root"))
    with pytest.raises(FileNotFoundError, match="1:epochs"):
        validate_dataset_spec(spec, subjects=("1",), require_files=True)


def test_matlab_fieldtrip_loader_returns_trial_dataset(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    mat_path = root / "Part1Data.mat"
    times = np.array([-0.1, 0.0, 0.1])
    trial1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    trial2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    savemat(
        mat_path,
        {
            "data": {
                "trial": np.array([trial1, trial2], dtype=object),
                "time": np.array([times, times], dtype=object),
                "label": np.array(["MEG001", "MEG002"], dtype=object),
                "trialinfo": np.array([1, 2]),
            }
        },
    )
    spec_path = tmp_path / "dataset.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "neureptrace.dataset.v1",
                "dataset_id": "mat-toy",
                "root": {"path": str(root)},
                "subjects": [1],
                "splits": {
                    "main": {
                        "loader": "matlab_fieldtrip",
                        "path_template": "Part{subject}Data.mat",
                        "label_index_base": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    spec = load_dataset_spec(spec_path)
    dataset = load_split_dataset(spec, "main", 1)

    assert dataset.data.shape == (2, 2, 3)
    assert np.allclose(dataset.times, times)
    assert dataset.labels is not None
    assert dataset.labels.tolist() == [0, 1]
    assert dataset.channels == ("MEG001", "MEG002")


def test_expand_participant_ids_supports_ranges_and_deduplication() -> None:
    ids = expand_participant_ids(["1-3", 3, 6, {"range": ["08", "10"]}, {"id": "13"}])

    assert ids == ("1", "2", "3", "6", "08", "09", "10", "13")


def test_v1_dataset_spec_resolves_env_root_and_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "meg"
    data_dir.mkdir()
    (data_dir / "Part2Data.mat").write_text("placeholder", encoding="utf-8")
    (data_dir / "Part2CueData.mat").write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("PYMEGDEC_DATA_DIR", str(data_dir))

    spec_path = tmp_path / "dataset.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "pymegdec_meg", "root": "${PYMEGDEC_DATA_DIR}", "format": "matlab_struct"},
                "participants": {"ids": [2], "files": {"main": "Part{participant}Data.mat", "cue": "Part{participant}CueData.mat"}},
                "roles": {"train": {"file_role": "main"}, "validation": {"file_role": "cue"}},
            }
        ),
        encoding="utf-8",
    )

    validations = validate_dataset_spec_v1(spec_path)
    table = build_dataset_file_table_v1(spec_path)

    assert all(validation.ok for validation in validations)
    assert list(table["role"]) == ["train", "validation"]
    assert table["exists"].all()


def test_v1_dataset_spec_reports_bad_role_reference(tmp_path: Path) -> None:
    data_dir = tmp_path / "meg"
    data_dir.mkdir()
    spec_path = tmp_path / "dataset.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "example", "root": str(data_dir), "format": "matlab_struct"},
                "participants": {"ids": [1], "files": {"main": "Part{participant}Data.mat"}},
                "roles": {"validation": {"file_role": "cue"}},
            }
        ),
        encoding="utf-8",
    )

    validations = validate_dataset_spec_v1(spec_path, check_exists=False)
    role_validation = next(validation for validation in validations if validation.scope == "roles")

    assert not role_validation.ok
    assert "not defined in participants.files" in " ".join(role_validation.messages)


def test_v1_dataset_spec_can_skip_existence_checks(tmp_path: Path) -> None:
    spec_path = tmp_path / "dataset.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "example", "root": str(tmp_path), "format": "matlab_struct"},
                "participants": {"ids": ["1-2"], "files": {"main": "Part{participant}Data.mat"}},
            }
        ),
        encoding="utf-8",
    )

    validations = validate_dataset_spec_v1(spec_path, check_exists=False)

    assert all(validation.ok for validation in validations)
