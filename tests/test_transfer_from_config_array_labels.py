from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neureptrace.transfer_from_config import run_transfer_from_config


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "transfer.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"name": "synthetic"},
                "preprocessing": {},
                "decoding": {"label_column": "condition"},
                "transfer": {},
                "outputs": {"base_dir": tmp_path.as_posix(), "summary_csv": "transfer.csv"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _array_label_vector(values: list[tuple[str, str]]) -> np.ndarray:
    labels = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        labels[index] = np.asarray(value, dtype=object)
    return labels


def _fake_dataset_with_array_labels():
    labels = _array_label_vector(
        [
            ("face", "left"),
            ("object", "left"),
            ("face", "left"),
            ("object", "left"),
        ]
    )
    data = np.zeros((4, 2, 11), dtype=float)
    data[:, 0, 4:7] = np.asarray([0.0, 1.0, 0.0, 1.0])[:, None]
    data[:, 1, 4:7] = np.asarray([1.0, 0.0, 1.0, 0.0])[:, None]
    return SimpleNamespace(
        data=data,
        times=np.linspace(-0.05, 0.05, 11),
        metadata=pd.DataFrame(
            {
                "condition": pd.Series(labels, dtype=object),
                "split": ["main", "main", "cue", "cue"],
            }
        ),
    )


def test_transfer_from_config_handles_array_valued_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "neureptrace.transfer_from_config.load_epoch_dataset_from_config",
        lambda *args, **kwargs: _fake_dataset_with_array_labels(),
    )
    config_path = _write_config(tmp_path)

    results = run_transfer_from_config(config_path)

    assert results["n_classes"].unique().tolist() == [2]
    assert results["n_train"].unique().tolist() == [2]
    class_names = results["class_names"].unique().tolist()
    assert len(class_names) == 1
    assert "face" in class_names[0]
    assert "object" in class_names[0]
