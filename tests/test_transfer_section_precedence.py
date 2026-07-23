from __future__ import annotations

import pytest

from neureptrace.transfer_from_config import _transfer_section


def test_transfer_section_prefers_explicit_empty_mapping_over_legacy_workflow() -> None:
    config = {
        "transfer": {},
        "workflow": {"label_column": "legacy_condition"},
    }

    assert _transfer_section(config) == {}


def test_transfer_section_treats_explicit_null_as_empty() -> None:
    config = {
        "transfer": None,
        "workflow": {"label_column": "legacy_condition"},
    }

    assert _transfer_section(config) == {}


@pytest.mark.parametrize("value", [[], "", False, 0])
def test_transfer_section_rejects_explicit_falsy_non_mapping(value: object) -> None:
    config = {
        "transfer": value,
        "workflow": {"label_column": "legacy_condition"},
    }

    with pytest.raises(ValueError, match="must be a mapping"):
        _transfer_section(config)
