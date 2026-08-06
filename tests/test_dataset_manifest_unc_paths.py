from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.dataset_manifest import manifest_from_dataset_config


@pytest.mark.parametrize("absolute_paths", [False, True])
def test_manifest_preserves_absolute_unc_file_patterns(tmp_path: Path, absolute_paths: bool):
    pattern = r"\\server\share\Part{participant}Data.mat"
    expected = r"\\server\share\Part7Data.mat"

    frame = manifest_from_dataset_config(
        {
            "dataset": {"root": "fallback-root"},
            "participants": {"include": [7]},
            "files": {"main": pattern},
        },
        config_dir=tmp_path,
        absolute_paths=absolute_paths,
    )

    assert frame["epochs"].tolist() == [expected]
    assert frame["input"].tolist() == [expected]
