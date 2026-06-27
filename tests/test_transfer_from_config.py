from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neureptrace.transfer_from_config import run_transfer_from_config


def _write_config(
    tmp_path: Path,
    *,
    preprocessing: dict | None = None,
    transfer: dict | None = None,
) -> Path:
    config_path = tmp_path / "transfer.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"name": "synthetic"},
                "preprocessing": preprocessing or {},
                "decoding": {"label_column": "condition"},
                "transfer": transfer or {},
                "outputs": {"base_dir": tmp_path.as_posix(), "summary_csv": "transfer.csv"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _fake_dataset():
    labels = np.asarray(["face", "object", "face", "object"], dtype=object)
    data = np.zeros((4, 2, 11), dtype=float)
    data[:, 0, 4:7] = np.asarray([0.0, 1.0, 0.0, 1.0])[:, None]
    data[:, 1, 4:7] = np.asarray([1.0, 0.0, 1.0, 0.0])[:, None]
    return SimpleNamespace(
        data=data,
        times=np.linspace(-0.05, 0.05, 11),
        metadata=pd.DataFrame(
            {
                "condition": labels,
                "split": ["main", "main", "cue", "cue"],
            }
        ),
    )


def _fake_dataset_with_unused_missing_row():
    base = _fake_dataset()
    return SimpleNamespace(
        data=np.concatenate([base.data, base.data[:1]], axis=0),
        times=base.times,
        metadata=pd.concat(
            [
                base.metadata,
                pd.DataFrame({"condition": [pd.NA], "split": ["unused"]}),
            ],
            ignore_index=True,
        ),
    )


def _fake_dataset_with_missing_test_row():
    base = _fake_dataset()
    metadata = base.metadata.copy()
    metadata.loc[3, "condition"] = pd.NA
    return SimpleNamespace(data=base.data, times=base.times, metadata=metadata)


def test_transfer_from_config_ignores_missing_labels_outside_selected_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "neureptrace.transfer_from_config.load_epoch_dataset_from_config",
        lambda *args, **kwargs: _fake_dataset_with_unused_missing_row(),
    )
    config_path = _write_config(tmp_path)

    results = run_transfer_from_config(config_path)

    assert results["n_classes"].unique().tolist() == [2]
    assert results["class_names"].unique().tolist() == ["face|object"]


def test_transfer_from_config_rejects_missing_labels_inside_selected_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "neureptrace.transfer_from_config.load_epoch_dataset_from_config",
        lambda *args, **kwargs: _fake_dataset_with_missing_test_row(),
    )
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="selected"):
        run_transfer_from_config(config_path)


@pytest.mark.parametrize(
    ("preprocessing", "transfer", "message"),
    [
        ({}, {"max_iter": True}, "transfer.max_iter"),
        ({}, {"max_iter": 100.5}, "transfer.max_iter"),
        ({"window_ms": True}, {}, "preprocessing.window_ms"),
        ({"step_ms": 0}, {}, "preprocessing.step_ms"),
        ({"tmin": True}, {}, "preprocessing.tmin"),
        ({"tmax": float("inf")}, {}, "preprocessing.tmax"),
        ({"tmin": 0.02, "tmax": 0.01}, {}, "preprocessing.tmax"),
    ],
)
def test_transfer_from_config_rejects_malformed_result_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preprocessing: dict,
    transfer: dict,
    message: str,
) -> None:
    monkeypatch.setattr(
        "neureptrace.transfer_from_config.load_epoch_dataset_from_config",
        lambda *args, **kwargs: _fake_dataset(),
    )
    config_path = _write_config(tmp_path, preprocessing=preprocessing, transfer=transfer)

    with pytest.raises(ValueError, match=message):
        run_transfer_from_config(config_path)


def test_transfer_from_config_rejects_malformed_write_provenance(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="write_provenance"):
        run_transfer_from_config(config_path, write_provenance="maybe")
