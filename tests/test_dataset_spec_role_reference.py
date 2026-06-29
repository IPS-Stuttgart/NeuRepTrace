from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.datasets.spec import build_dataset_file_table, resolve_dataset_files


def _spec_with_bad_role_reference(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": {"id": "example", "root": str(root), "format": "matlab_struct"},
        "participants": {"ids": [1], "files": {"main": "Part{participant}Data.mat"}},
        "roles": {"validation": {"file_role": "cue"}},
    }


def test_v1_dataset_file_table_reports_bad_role_reference(tmp_path: Path) -> None:
    spec_path = tmp_path / "dataset.json"
    spec_path.write_text(json.dumps(_spec_with_bad_role_reference(tmp_path)), encoding="utf-8")

    with pytest.raises(ValueError, match=r"roles\.validation\.file_role='cue' is not defined in participants\.files"):
        build_dataset_file_table(spec_path)


def test_v1_dataset_resolver_reports_bad_role_reference(tmp_path: Path) -> None:
    spec = _spec_with_bad_role_reference(tmp_path)

    with pytest.raises(ValueError, match=r"roles\.validation\.file_role='cue' is not defined in participants\.files"):
        resolve_dataset_files(spec, spec_dir=tmp_path)
