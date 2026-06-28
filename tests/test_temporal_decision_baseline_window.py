from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.io.dataset import EpochDataset
from neureptrace.temporal_decision_decode import run_temporal_decision_decode_dataset


def _synthetic_grouped_dataset() -> EpochDataset:
    rng = np.random.default_rng(13)
    times = np.arange(-0.1, 0.31, 0.05)
    n_groups = 3
    n_classes = 3
    trials_per_class_and_group = 2
    n_channels = 4
    rows = []
    labels = []
    for group in range(n_groups):
        for class_label in range(n_classes):
            for _trial in range(trials_per_class_and_group):
                rows.append({"stimulus": class_label, "participant": f"p{group}"})
                labels.append(class_label)
    data = rng.normal(scale=0.05, size=(len(rows), n_channels, len(times)))
    decision_mask = (times >= 0.05) & (times <= 0.2)
    patterns = np.eye(n_classes, n_channels)
    for trial_index, class_label in enumerate(labels):
        data[trial_index, :, decision_mask] += patterns[class_label] * 1.5
    return EpochDataset(
        data=data,
        times=times,
        channel_names=[f"MEG{channel}" for channel in range(n_channels)],
        metadata=pd.DataFrame(rows),
        name="synthetic_bush_like",
    )


def test_temporal_decision_decode_accepts_string_baseline_window(tmp_path: Path) -> None:
    results = run_temporal_decision_decode_dataset(
        _synthetic_grouped_dataset(),
        label_column="stimulus",
        group_column="participant",
        out_path=tmp_path / "summary.csv",
        decoders=["logistic"],
        window_ms=100.0,
        step_ms=50.0,
        test_window=(0.075, 0.175),
        baseline_window="[-0.10, -0.05]",
        emission_mode="uncalibrated",
        max_iter=1000,
    )

    np.testing.assert_allclose(results["baseline_window_start"], -0.10)
    np.testing.assert_allclose(results["baseline_window_stop"], -0.05)
