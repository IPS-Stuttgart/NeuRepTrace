from __future__ import annotations

from pathlib import Path

from neureptrace.dataset_config import apply_overrides, iter_dataset_files, parse_participant_ids, validate_dataset_config


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
