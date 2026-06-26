from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neureptrace.datasets.loaders import matlab_struct
from neureptrace.datasets.spec import DatasetFile


def _dataset_file(tmp_path: Path) -> DatasetFile:
    return DatasetFile(
        participant="1",
        role="main",
        file_role="main",
        path=tmp_path / "data.mat",
        exists=True,
    )


def _capture_loader(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_load_matlab_struct(
        path: Path,
        *,
        variable: str,
        index_path: tuple[int, ...],
        squeeze_me: bool,
        struct_as_record: bool,
    ) -> str:
        calls.update(
            {
                "path": path,
                "variable": variable,
                "index_path": index_path,
                "squeeze_me": squeeze_me,
                "struct_as_record": struct_as_record,
            }
        )
        return "loaded"

    monkeypatch.setattr(matlab_struct, "load_matlab_struct", fake_load_matlab_struct)
    return calls


def test_matlab_struct_recording_preserves_scalar_zero_index_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _capture_loader(monkeypatch)

    result = matlab_struct.load_matlab_struct_recording(
        _dataset_file(tmp_path),
        {"matlab": {"variable": "recording", "index_path": 0}},
    )

    assert result == "loaded"
    assert calls["index_path"] == (0,)
    assert calls["variable"] == "recording"


def test_matlab_struct_recording_preserves_sequence_index_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _capture_loader(monkeypatch)

    result = matlab_struct.load_matlab_struct_recording(
        _dataset_file(tmp_path),
        {"matlab": {"index_path": [0, 2]}},
    )

    assert result == "loaded"
    assert calls["index_path"] == (0, 2)


@pytest.mark.parametrize("bad_index_path", ["10", b"0", True, [0, False], [1.5], 1.5])
def test_matlab_struct_recording_rejects_ambiguous_index_path_values(bad_index_path: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="matlab.index_path"):
        matlab_struct.load_matlab_struct_recording(
            _dataset_file(tmp_path),
            {"matlab": {"index_path": bad_index_path}},
        )
