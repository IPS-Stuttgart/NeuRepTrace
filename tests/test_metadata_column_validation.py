from __future__ import annotations

import pytest

from neureptrace.dataset_config import ConfigValidationError, validate_dataset_config
from neureptrace.io.fieldtrip_mat import _metadata_columns_from_config


def _fieldtrip_config_with_metadata_index(index):
    return {
        "dataset": {"type": "fieldtrip_mat", "files": ["demo.mat"]},
        "metadata": {"columns": [{"name": "condition", "index": index}]},
        "decoding": {"label_column": "condition"},
    }


def test_validate_dataset_config_accepts_non_negative_metadata_column_indices(tmp_path):
    assert validate_dataset_config(_fieldtrip_config_with_metadata_index(0), base_dir=tmp_path) == []
    assert validate_dataset_config(_fieldtrip_config_with_metadata_index("1"), base_dir=tmp_path) == []


def test_validate_dataset_config_rejects_invalid_metadata_column_indices(tmp_path):
    for index in (True, False, -1, "-1", 1.5, "1.5", float("nan"), "condition"):
        with pytest.raises(ConfigValidationError, match="metadata.columns.index|non-negative"):
            validate_dataset_config(_fieldtrip_config_with_metadata_index(index), base_dir=tmp_path)


def test_validate_dataset_config_requires_metadata_columns_list(tmp_path):
    config = {
        "dataset": {"type": "fieldtrip_mat", "files": ["demo.mat"]},
        "metadata": {"columns": {"name": "condition", "index": 0}},
        "decoding": {"label_column": "condition"},
    }

    with pytest.raises(ConfigValidationError, match="metadata.columns must be a list"):
        validate_dataset_config(config, base_dir=tmp_path)


def test_fieldtrip_metadata_column_parser_rejects_reverse_indexing():
    with pytest.raises(ValueError, match="non-negative"):
        _metadata_columns_from_config(_fieldtrip_config_with_metadata_index(-1))


def test_fieldtrip_metadata_column_parser_coerces_valid_string_indices():
    specs = _metadata_columns_from_config(_fieldtrip_config_with_metadata_index("2"))

    assert len(specs) == 1
    assert specs[0].name == "condition"
    assert specs[0].index == 2
