import sys
from pathlib import Path

import pytest

from neureptrace.inference import main


def test_main_rejects_colliding_output_paths_before_input_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_path = tmp_path / "inference.csv"
    aliased_path = tmp_path / "nested" / ".." / "inference.csv"
    missing_input = tmp_path / "missing.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neureptrace.inference",
            str(missing_input),
            "--out-time",
            str(aliased_path),
            "--out-clusters",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="output paths must be distinct"):
        main()

    assert not output_path.exists()
    assert not (tmp_path / "nested").exists()
