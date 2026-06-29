from __future__ import annotations

import pytest

from neureptrace.dataset_config import ConfigValidationError, validate_dataset_config


def _base_config() -> dict:
    return {
        "dataset": {"type": "mne_epochs", "epochs": "subject-01-epo.fif"},
        "decoding": {"label_column": "condition"},
    }


@pytest.mark.parametrize("section_name", ["metadata", "validation", "participants", "decoding", "workflow"])
@pytest.mark.parametrize(
    "section_value",
    ["", False, 0, []],
    ids=["empty-string", "false", "zero", "empty-list"],
)
def test_dataset_config_rejects_falsey_non_mapping_optional_sections(section_name: str, section_value) -> None:
    config = _base_config()
    config[section_name] = section_value

    with pytest.raises(ConfigValidationError, match=f"Config section '{section_name}' must be a mapping"):
        validate_dataset_config(config, check_files=False)
