from __future__ import annotations

import json

from neureptrace.config import ConfigSpec, load_config, parse_participants


def test_parse_participants_expands_ranges_and_preserves_padding() -> None:
    assert parse_participants("1-3,3,5") == ("1", "2", "3", "5")
    assert parse_participants("01-03,7") == ("01", "02", "03", "7")


def test_load_json_config_materializes_participant_files(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("NEUREPTRACE_DATA_ROOT", str(data_root))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": {
                    "id": "example",
                    "root_env": "NEUREPTRACE_DATA_ROOT",
                    "participants": "1-2",
                    "metadata": {"label_column": "condition", "group_column": "subject"},
                    "files": {
                        "main": {
                            "pattern": "sub-{participant}/epochs.fif",
                            "loader": "mne_epochs",
                            "metadata": "sub-{participant}/metadata.csv",
                        }
                    },
                },
                "workflow": {"name": "decode", "train": "main", "test": "main", "label": "condition"},
            }
        )
    )

    config = load_config(config_path)
    assert config.validate() == []
    files = config.dataset.iter_files()
    assert [item.participant for item in files] == ["1", "2"]
    assert files[0].path == data_root / "sub-1" / "epochs.fif"
    assert files[0].metadata_path == data_root / "sub-1" / "metadata.csv"


def test_load_yaml_config_accepts_workflows_mapping(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
version: 1
dataset:
  id: yaml-example
  root: data
  participants: [sub-01]
  metadata:
    label_column: condition
  files:
    main:
      path: sub-01/epochs.fif
      loader: mne_epochs
workflows:
  time_decode:
    train: main
    test: main
    label: condition
""".strip()
    )

    config = load_config(config_path)
    assert config.dataset.id == "yaml-example"
    assert config.workflows[0].name == "time_decode"
    assert config.dataset.iter_files()[0].path == tmp_path / "data" / "sub-01" / "epochs.fif"


def test_validate_reports_unknown_workflow_role() -> None:
    config = ConfigSpec.from_mapping(
        {
            "version": 1,
            "dataset": {
                "id": "example",
                "metadata": {"label_column": "condition"},
                "files": {"main": {"path": "epochs.fif", "loader": "mne_epochs"}},
            },
            "workflow": {"name": "decode", "train": "main", "test": "cue", "label": "condition"},
        }
    )

    assert config.validate() == ["workflow.decode references unknown dataset file role: cue"]
