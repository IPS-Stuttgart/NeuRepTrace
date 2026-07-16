from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.inference import _t_statistic, sign_flip_time_inference


def test_t_statistic_preserves_exact_constant_effects() -> None:
    effects = np.asarray(
        [
            [0.125, -0.25, 0.0],
            [0.125, -0.25, 0.0],
            [0.125, -0.25, 0.0],
        ]
    )

    statistics = _t_statistic(effects)

    assert np.isposinf(statistics[0])
    assert np.isneginf(statistics[1])
    assert statistics[2] == 0.0


def test_sign_flip_inference_detects_identical_positive_effects(tmp_path: Path) -> None:
    csv_paths: list[Path] = []
    for subject_index in range(8):
        subject = f"sub-{subject_index + 1:02d}"
        csv_path = tmp_path / f"{subject}_time_decode.csv"
        pd.DataFrame(
            {
                "subject": [subject],
                "fold": [0],
                "time": [0.1],
                "accuracy": [0.625],
                "log_loss": [0.5],
                "brier": [0.5],
                "ece": [0.1],
            }
        ).to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

    time_table, cluster_table = sign_flip_time_inference(
        csv_paths,
        chance=0.5,
        n_permutations=4096,
        random_state=7,
    )

    assert np.isposinf(time_table.loc[0, "statistic"])
    assert time_table.loc[0, "pointwise_p"] < 0.05
    assert not cluster_table.empty
    assert cluster_table.loc[0, "cluster_p"] < 0.05
