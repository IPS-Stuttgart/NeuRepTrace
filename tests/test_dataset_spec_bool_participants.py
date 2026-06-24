from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.datasets.spec import expand_participant_ids, validate_loaded_dataset_spec


@pytest.mark.parametrize("ids", ([True], [{"id": False}], [{"range": [True, 2]}]))
def test_expand_participant_ids_rejects_boolean_identifiers(ids: list[object]) -> None:
    with pytest.raises(ValueError, match="boolean identifiers"):
        expand_participant_ids(ids)


def test_dataset_spec_validation_reports_boolean_participant_id(tmp_path: Path) -> None:
    spec = {
        "schema_version": 1,
        "dataset": {"id": "toy", "root": str(tmp_path), "format": "matlab_struct"},
        "participants": {"ids": [True], "files": {"main": "Part{participant}Data.mat"}},
    }

    validations = validate_loaded_dataset_spec(spec, spec_dir=tmp_path, check_exists=False)
    participant_validation = next(validation for validation in validations if validation.scope == "participants")

    assert not participant_validation.ok
    assert "boolean identifiers" in " ".join(participant_validation.messages)
