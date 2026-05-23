from __future__ import annotations

from pathlib import Path

from neureptrace.dataset_spec_cli import main as dataset_spec_main
from neureptrace.datasets.pymegdec import (
    build_pymegdec_bushmeg_dataset_spec_text,
    write_pymegdec_bushmeg_dataset_spec,
)


def test_build_pymegdec_bushmeg_dataset_spec_text_contains_legacy_paths() -> None:
    text = build_pymegdec_bushmeg_dataset_spec_text(participants="2", env_var="CUSTOM_DATA", data_dir="/tmp/meg")

    assert "path: \"/tmp/meg\"" in text
    assert "env: CUSTOM_DATA" in text
    assert "include: \"2\"" in text
    assert "Part{subject}Data.mat" in text
    assert "Part{subject}CueData.mat" in text
    assert "loader: matlab_fieldtrip" in text


def test_write_pymegdec_bushmeg_dataset_spec(tmp_path: Path) -> None:
    out = tmp_path / "configs" / "bushmeg.yml"

    assert write_pymegdec_bushmeg_dataset_spec(["--out", str(out), "--participants", "1-2"]) == 0

    text = out.read_text(encoding="utf-8")
    assert "dataset_id: bushmeg" in text
    assert "include: \"1-2\"" in text
    assert "paired_split: cue" in text


def test_dataset_cli_writes_pymegdec_bushmeg_spec(tmp_path: Path) -> None:
    out = tmp_path / "bushmeg.yml"

    assert dataset_spec_main(["pymegdec-bushmeg", "--out", str(out), "--participants", "2"]) == 0

    text = out.read_text(encoding="utf-8")
    assert "dataset_id: bushmeg" in text
    assert "include: \"2\"" in text
    assert "Part{subject}CueData.mat" in text
