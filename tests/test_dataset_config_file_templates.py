from __future__ import annotations

import numpy as np
import scipy.io

from neureptrace.dataset_config import iter_dataset_files, load_epoch_dataset_from_config


def _write_fieldtrip_mat(path, *, label_offset=0):
    time = np.arange(20, dtype=float) / 100.0 - 0.05
    trials = np.empty((1, 2), dtype=object)
    trials[0, 0] = np.vstack([np.sin(2 * np.pi * 10 * time), np.cos(2 * np.pi * 10 * time)])
    trials[0, 1] = np.vstack([np.sin(2 * np.pi * 11 * time), np.cos(2 * np.pi * 11 * time)])
    times = np.empty((1, 2), dtype=object)
    times[0, 0] = time
    times[0, 1] = time
    data = {
        "trial": trials,
        "time": times,
        "label": np.array(["MLO11", "MRO11"], dtype=object),
        "trialinfo": np.array([[1 + label_offset], [2 + label_offset]], dtype=int),
    }
    scipy.io.savemat(path, {"data": data})


def test_iter_dataset_files_expands_split_file_templates(tmp_path):
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": str(tmp_path),
            "file_templates": {
                "main": "Part{participant}Data.mat",
                "cue": "Part{participant}CueData.mat",
            },
        },
        "participants": {"ids": "1-2"},
    }

    paths = [path.name for path in iter_dataset_files(config)]

    assert paths == ["Part1Data.mat", "Part1CueData.mat", "Part2Data.mat", "Part2CueData.mat"]


def test_load_epoch_dataset_from_config_adds_split_and_participant_metadata(tmp_path):
    _write_fieldtrip_mat(tmp_path / "Part2Data.mat")
    _write_fieldtrip_mat(tmp_path / "Part2CueData.mat", label_offset=10)
    config = {
        "schema_version": "neureptrace.dataset.v1",
        "dataset": {
            "name": "bushmeg_smoke_test",
            "type": "fieldtrip_mat",
            "root": str(tmp_path),
            "variable": "data",
            "file_templates": {
                "main": "Part{participant}Data.mat",
                "cue": "Part{participant}CueData.mat",
            },
        },
        "participants": {"ids": "2"},
        "metadata": {"columns": [{"name": "stimulus", "index": 0}]},
        "validation": {"channel_policy": "exact"},
    }

    dataset = load_epoch_dataset_from_config(config)

    assert dataset.data.shape == (4, 2, 20)
    assert dataset.metadata["participant"].tolist() == ["2", "2", "2", "2"]
    assert dataset.metadata["split"].tolist() == ["main", "main", "cue", "cue"]
    assert dataset.metadata["stimulus"].tolist() == [1, 2, 11, 12]
