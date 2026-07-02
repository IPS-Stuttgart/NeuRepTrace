from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neureptrace.dataset_config import iter_dataset_files, parse_participant_ids


def test_parse_participant_ids_accepts_numpy_integer_scalar():
    assert parse_participant_ids(np.int64(2)) == [2]
    assert parse_participant_ids(np.uint32(3)) == [3]


def test_parse_participant_ids_rejects_numpy_boolean_scalar():
    with pytest.raises(ValueError, match="boolean|booleans"):
        parse_participant_ids(np.bool_(True))
    with pytest.raises(ValueError, match="boolean|booleans"):
        parse_participant_ids([np.bool_(False)])


def test_iter_dataset_files_accepts_numpy_integer_scalar_participant(tmp_path: Path):
    config = {
        "dataset": {
            "type": "mne_epochs",
            "root": "data",
            "epochs_files": {"template": "sub-{subject03d}_epo.fif"},
        },
        "participants": {"ids": np.int64(7)},
        "decoding": {"label_column": "condition"},
    }

    assert iter_dataset_files(config, base_dir=tmp_path) == [tmp_path / "data" / "sub-007_epo.fif"]
