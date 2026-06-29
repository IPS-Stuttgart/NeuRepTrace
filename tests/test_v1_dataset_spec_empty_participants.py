from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.datasets.spec import expand_participant_ids, validate_dataset_spec as validate_dataset_spec_v1


@pytest.mark.parametrize(
    "ids",
    [
        [{"id": ""}],
        [{"id": "   "}],
        [{"range": ""}],
        [{"range": "   "}],
        [{"range": ["", "2"]}],
        [{"range": ["1", " "]}],
    ],
)
def test_expand_participant_ids_rejects_empty_mapping_values(ids: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="empty identifier"):
        expand_participant_ids(ids)


def test_v1_dataset_spec_reports_empty_mapping_participant_id(tmp_path: Path) -> None:
    spec_path = tmp_path / "dataset.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "example", "root": str(tmp_path), "format": "matlab_struct"},
                "participants": {"ids": [{"id": ""}], "files": {"main": "Part{participant}Data.mat"}},
            }
        ),
        encoding="utf-8",
    )

    validations = validate_dataset_spec_v1(spec_path, check_exists=False)
    participant_validation = next(validation for validation in validations if validation.scope == "participants")

    assert not participant_validation.ok
    assert "empty identifier" in " ".join(participant_validation.messages)
