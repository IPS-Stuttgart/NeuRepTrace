from __future__ import annotations

from pathlib import Path

from neureptrace.dataset_config import _fieldtrip_file_specs, iter_dataset_files, validate_dataset_config


def test_single_fieldtrip_template_supports_padded_participant_placeholders(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "type": "fieldtrip_mat",
            "root": "data",
            "file_template": "sub-{participant02d}/ses-{subject02d}/Part{participant}Data.mat",
        },
        "participants": {"ids": "1,12"},
        "decoding": {"label_column": "stimulus_class"},
    }
    expected_paths = [
        tmp_path / "data" / "sub-01" / "ses-01" / "Part1Data.mat",
        tmp_path / "data" / "sub-12" / "ses-12" / "Part12Data.mat",
    ]

    assert validate_dataset_config(config, base_dir=tmp_path) == []
    assert iter_dataset_files(config, base_dir=tmp_path) == expected_paths

    specs = _fieldtrip_file_specs(config, base_dir=tmp_path)
    assert [path for path, _extra in specs] == expected_paths
    assert [extra["participant"] for _path, extra in specs] == [1, 12]
