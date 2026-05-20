from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from neureptrace.dataset_config import load_epoch_dataset_from_config
from neureptrace.io.fieldtrip_mat import load_fieldtrip_mat_epochs


def _write_fieldtrip_mat(path: Path) -> None:
    data = {
        "trial": np.array(
            [
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            ],
            dtype=object,
        ),
        "time": np.array(
            [
                np.array([0.0, 0.01, 0.02]),
                np.array([0.0, 0.01, 0.02]),
            ],
            dtype=object,
        ),
        "label": np.array(["MEG001", "MEG002"], dtype=object),
        "trialinfo": np.array([[1, 10], [2, 20]]),
    }
    savemat(path, {"data": data})


def test_load_fieldtrip_mat_epochs_maps_trialinfo_columns(tmp_path: Path):
    mat_path = tmp_path / "Part2Data.mat"
    _write_fieldtrip_mat(mat_path)

    dataset = load_fieldtrip_mat_epochs(
        mat_path,
        {
            "variable": "data",
            "metadata": {
                "columns": [
                    {"name": "stimulus_class", "index": 0},
                    {"name": "condition", "index": 1},
                ]
            },
        },
        extra_metadata={"participant": 2},
    )

    assert dataset.data.shape == (2, 2, 3)
    assert dataset.channel_names == ["MEG001", "MEG002"]
    assert dataset.times.tolist() == [0.0, 0.01, 0.02]
    assert dataset.metadata["stimulus_class"].tolist() == [1, 2]
    assert dataset.metadata["condition"].tolist() == [10, 20]
    assert dataset.metadata["participant"].tolist() == [2, 2]


def test_load_epoch_dataset_from_yaml_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_fieldtrip_mat(data_dir / "Part2Data.mat")

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
schema_version: neureptrace.dataset.v1
dataset:
  type: fieldtrip_mat
  root: data
  participant_file: "Part{participant}Data.mat"
  variable: data
participants:
  ids: 2
metadata:
  columns:
    - name: stimulus_class
      index: 0
decoding:
  label_column: stimulus_class
""",
        encoding="utf-8",
    )

    dataset = load_epoch_dataset_from_config(config_path, check_files=True)

    assert dataset.data.shape == (2, 2, 3)
    assert dataset.metadata["participant"].tolist() == [2, 2]
    assert dataset.metadata["stimulus_class"].tolist() == [1, 2]
