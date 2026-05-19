from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.dataset_config import (
    DatasetConfigError,
    expand_participants,
    load_dataset_config,
    main,
    to_benchmark_manifest_frame,
    validate_dataset_config,
)


def _write_nod_yaml(path: Path) -> None:
    pytest.importorskip("yaml")
    path.write_text(
        """
version: 1
dataset:
  id: nod
  root: data
  participants: "01-02"
  subject_template: "sub-{participant}"
  files:
    epochs:
      pattern: "{subject}_epo.fif"
      loader: mne_epochs
    events:
      pattern: "{subject}_events.csv"
      loader: csv_events
workflow:
  name: animate
  kind: mne_time_decode
  epochs: epochs
  events: events
  label_column: condition
  group_column: run
  source_column: category
  positive_pattern: "face|person"
  positive_label: face
  negative_label: object
  options:
    n_splits: 2
    window_ms: 20
""".lstrip(),
        encoding="utf-8",
    )


def test_expand_participants_preserves_zero_padding() -> None:
    assert expand_participants("01-03,10") == ("01", "02", "03", "10")
    assert expand_participants([1, "3-4"]) == ("1", "3", "4")


def test_yaml_config_validates_and_compiles_to_benchmark_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yml"
    _write_nod_yaml(config_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for subject in ("sub-01", "sub-02"):
        (data_dir / f"{subject}_epo.fif").write_text("placeholder", encoding="utf-8")
        (data_dir / f"{subject}_events.csv").write_text("category,run\nface,1\nchair,1\n", encoding="utf-8")

    config = load_dataset_config(config_path)
    validation = validate_dataset_config(config, check_files=True)

    assert validation.ok, validation.errors
    frame = to_benchmark_manifest_frame(config, relative_to=tmp_path)
    assert list(frame["subject"]) == ["sub-01", "sub-02"]
    assert list(frame["participant"]) == ["01", "02"]
    assert frame.loc[0, "epochs"] == "data/sub-01_epo.fif"
    assert frame.loc[0, "events_csv"] == "data/sub-01_events.csv"
    assert frame.loc[0, "label_column"] == "condition"
    assert frame.loc[0, "positive_pattern"] == "face|person"
    assert frame.loc[0, "n_splits"] == 2


def test_json_config_expands_environment_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "staged"
    data_root.mkdir()
    monkeypatch.setenv("NRT_DATA_ROOT", str(data_root))
    config_path = tmp_path / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": {
                    "id": "env_dataset",
                    "root": "${NRT_DATA_ROOT}",
                    "participants": ["001"],
                    "subject_template": "sub-{participant}",
                    "files": {"epochs": {"pattern": "{subject}_epo.fif", "loader": "mne_epochs"}},
                },
                "workflow": {"name": "decode", "epochs": "epochs", "label_column": "condition"},
            }
        ),
        encoding="utf-8",
    )
    (data_root / "sub-001_epo.fif").write_text("placeholder", encoding="utf-8")

    config = load_dataset_config(config_path)
    assert validate_dataset_config(config, check_files=True).ok
    frame = to_benchmark_manifest_frame(config)
    assert frame.loc[0, "epochs"] == str(data_root / "sub-001_epo.fif")


def test_validate_reports_unknown_workflow_role(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "id": "broken",
                    "participants": "1",
                    "files": {"epochs": {"pattern": "sub-{participant}_epo.fif"}},
                },
                "workflow": {"name": "decode", "epochs": "missing", "label_column": "condition"},
            }
        ),
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)
    validation = validate_dataset_config(config)

    assert not validation.ok
    assert "unknown epochs role" in " ".join(validation.errors)


def test_compile_rejects_non_mne_loader(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "id": "mat_dataset",
                    "participants": "1",
                    "files": {"main": {"pattern": "Part{participant}Data.mat", "loader": "fieldtrip_mat"}},
                },
                "workflow": {"name": "decode", "kind": "mne_time_decode", "epochs": "main", "label_column": "stimulus_id"},
            }
        ),
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)
    with pytest.raises(DatasetConfigError, match="expected one of"):
        to_benchmark_manifest_frame(config)


def test_cli_writes_benchmark_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yml"
    _write_nod_yaml(config_path)
    out_path = tmp_path / "benchmarks" / "nod_animate.csv"

    result = main([str(config_path), "--write-benchmark-manifest", str(out_path)])

    assert result == 0
    assert out_path.read_text(encoding="utf-8").startswith("subject,participant,epochs,events_csv")
