from __future__ import annotations

from pathlib import Path

from neureptrace.dataset_config import iter_dataset_files, validate_dataset_config


def test_fieldtrip_dataset_files_accepts_single_string_path(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": "data",
            "files": "subject01.mat",
        },
        "decoding": {"label_column": "stimulus_class"},
    }

    assert validate_dataset_config(config, base_dir=tmp_path) == []
    assert iter_dataset_files(config, base_dir=tmp_path) == [tmp_path / "data" / "subject01.mat"]
