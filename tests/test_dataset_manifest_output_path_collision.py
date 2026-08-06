from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.dataset_manifest import write_manifest_from_dataset_config


@pytest.mark.parametrize("alias", [False, True])
def test_manifest_writer_rejects_output_that_overwrites_config(
    tmp_path: Path,
    alias: bool,
) -> None:
    config_path = tmp_path / "dataset.json"
    original = json.dumps(
        {
            "dataset": {"root": "data"},
            "participants": {"include": [1]},
            "files": {"main": "sub-{participant}_epo.fif"},
        }
    )
    config_path.write_text(original, encoding="utf-8")
    out_path = (
        config_path.parent / "unused" / ".." / config_path.name
        if alias
        else config_path
    )

    with pytest.raises(
        ValueError,
        match="output path must differ from the input config path",
    ):
        write_manifest_from_dataset_config(config_path, out_path)

    assert config_path.read_text(encoding="utf-8") == original
