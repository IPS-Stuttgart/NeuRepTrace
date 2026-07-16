from __future__ import annotations

import json
from pathlib import Path

from neureptrace.dataset_manifest import write_manifest_from_dataset_config


def test_write_manifest_absolute_paths_resolves_relative_config_path(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"root": "data"},
                "participants": {"include": [1]},
                "files": {"main": "sub-{participant}_epo.fif"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    frame = write_manifest_from_dataset_config(
        Path("configs/dataset.json"),
        Path("manifest.csv"),
        absolute_paths=True,
    )

    expected = (config_dir / "data/sub-1_epo.fif").resolve()
    assert Path(frame.loc[0, "epochs"]) == expected
    assert Path(frame.loc[0, "epochs"]).is_absolute()
    assert frame.loc[0, "input"] == str(expected)
