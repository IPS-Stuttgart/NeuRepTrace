from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.loso_time_decode import run_loso_time_decode


@dataclass
class FakeEpochDataset:
    data: np.ndarray
    times: np.ndarray
    metadata: pd.DataFrame
    name: str = "numeric_group_loso"


def _numeric_group_dataset() -> FakeEpochDataset:
    rng = np.random.default_rng(42)
    groups = np.repeat([0, 1, 2], 4)
    labels = np.tile(["class_a", "class_b", "class_a", "class_b"], 3)
    data = rng.normal(scale=0.05, size=(len(labels), 2, 5))
    sign = np.where(labels == "class_a", -1.0, 1.0)
    data[:, 0, :2] += 2.0 * sign[:, None]
    metadata = pd.DataFrame({"subject": groups, "condition": labels})
    return FakeEpochDataset(
        data=data,
        times=np.array([0.00, 0.01, 0.02, 0.03, 0.04]),
        metadata=metadata,
    )


def test_loso_preserves_explicit_zero_test_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "dataset": {"name": "numeric_group_loso"},
        "preprocessing": {"window_ms": 20.0, "step_ms": 20.0, "normalization": "none"},
        "loso": {
            "label_column": "condition",
            "group_column": "subject",
            "test_groups": 0,
            "heldout_groups": [2],
            "decoder": "logistic",
            "emission_mode": "uncalibrated",
            "normalization_scope": "per_group",
            "max_iter": 500,
        },
        "outputs": {"summary_csv": str(tmp_path / "loso_zero_group.csv"), "provenance": False},
    }
    monkeypatch.setattr("neureptrace.loso_time_decode.load_config", lambda _path: config)
    monkeypatch.setattr(
        "neureptrace.loso_time_decode.load_epoch_dataset_from_config",
        lambda *_args, **_kwargs: _numeric_group_dataset(),
    )

    results = run_loso_time_decode(tmp_path / "config.yml")

    assert results["outer_group"].unique().tolist() == [0]
    assert results["n_test"].unique().tolist() == [4]
