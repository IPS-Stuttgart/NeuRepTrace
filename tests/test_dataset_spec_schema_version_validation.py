from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.datasets.spec import validate_dataset_spec


def _write_v1_spec(path: Path, root: Path, schema_version: object) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "dataset": {"id": "example", "root": str(root), "format": "matlab_struct"},
                "participants": {"ids": [1], "files": {"main": "Part{participant}Data.mat"}},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("bad_version", [True, 1.5, "1.0", "latest"])
def test_v1_dataset_spec_reports_malformed_schema_version(tmp_path: Path, bad_version: object) -> None:
    spec_path = _write_v1_spec(tmp_path / "dataset.json", tmp_path, bad_version)

    validations = validate_dataset_spec(spec_path, check_exists=False)

    schema_validation = next(validation for validation in validations if validation.scope == "schema")
    assert not schema_validation.ok
    assert "schema_version must be an integer value" in " ".join(schema_validation.messages)
