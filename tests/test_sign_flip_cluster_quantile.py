from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.inference import sign_flip_time_inference


def test_sign_flip_cluster_threshold_handles_infinite_null_statistics(tmp_path: Path) -> None:
    csv_paths: list[Path] = []
    for subject_index in range(2):
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

    with np.errstate(invalid="raise"):
        time_table, cluster_table = sign_flip_time_inference(
            csv_paths,
            chance=0.5,
            n_permutations=64,
            random_state=7,
            cluster_alpha=0.05,
        )

    assert np.isposinf(time_table.loc[0, "statistic"])
    assert np.isposinf(time_table.loc[0, "cluster_threshold"])
    assert time_table.loc[0, "cluster_id"] == 1
    assert not cluster_table.empty
    assert np.isposinf(cluster_table.loc[0, "cluster_mass"])
    assert 0.0 < cluster_table.loc[0, "cluster_p"] <= 1.0
