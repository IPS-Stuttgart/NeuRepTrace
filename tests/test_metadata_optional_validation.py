from __future__ import annotations

import pytest

from neureptrace.dataset_config import ConfigValidationError, validate_dataset_config
from neureptrace.io.fieldtrip_mat import _metadata_columns_from_config


def _config(optional):
    return {
        "dataset": {"type": "fieldtrip_mat", "files": ["demo.mat"]},
        "metadata": {"columns": [{"name": "condition", "index": 0, "optional": optional}]},
        "decoding": {"label_column": "condition"},
    }


def test_fieldtrip_metadata_optional_string_false_stays_false():
    specs = _metadata_columns_from_config(_config("false"))

    assert len(specs) == 1
    assert specs[0].optional is False


def test_fieldtrip_metadata_optional_string_true_stays_true():
    specs = _metadata_columns_from_config(_config("true"))

    assert len(specs) == 1
    assert specs[0].optional is True


def test_validate_dataset_config_rejects_non_boolean_metadata_optional(tmp_path):
    with pytest.raises(ConfigValidationError, match="metadata.columns.optional|boolean"):
        validate_dataset_config(_config("maybe"), base_dir=tmp_path)
