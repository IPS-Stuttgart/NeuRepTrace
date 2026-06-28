from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.observation_ensemble import DEFAULT_ENSEMBLE_DECODER, main


def _source_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decoder_specs = {
        "logistic": {
            "baseline": (0.70, 0.30),
            "effect": (0.80, 0.20),
        },
        "linear_svm": {
            "baseline": (0.80, 0.20),
            "effect": (0.95, 0.05),
        },
    }
    for decoder, probabilities_by_window in decoder_specs.items():
        for sample_index, true_label in enumerate([0, 1]):
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "split_id": "split-0",
                    "seed": 13,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": -0.20,
                    "test_time": -0.20,
                    "time": -0.20,
                    "window_start": -0.21,
                    "window_stop": -0.19,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": true_label,
                    "true_class": "zero" if true_label == 0 else "one",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": probabilities_by_window["baseline"][0],
                    "prob_class_1": probabilities_by_window["baseline"][1],
                }
            )
        for sample_index in [0, 1]:
            rows.append(
                {
                    "subject": "sub-01",
                    "fold": 0,
                    "split_id": "split-0",
                    "seed": 13,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": 0.10,
                    "test_time": 0.10,
                    "time": 0.10,
                    "window_start": 0.09,
                    "window_stop": 0.11,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": 0,
                    "true_class": "zero",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": probabilities_by_window["effect"][0],
                    "prob_class_1": probabilities_by_window["effect"][1],
                }
            )
    return pd.DataFrame(rows)


def test_observation_ensemble_cli_forwards_probability_tolerance(tmp_path: Path) -> None:
    observations = _source_observations()
    logistic_rows = observations["decoder"].eq("logistic")
    observations.loc[logistic_rows, "prob_class_1"] += 0.05

    source_path = tmp_path / "source_observations.csv"
    ensemble_path = tmp_path / "ensemble_observations.csv"
    observations.to_csv(source_path, index=False)

    exit_code = main(
        [
            str(source_path),
            "--out",
            str(ensemble_path),
            "--baseline-window",
            "-0.25",
            "-0.15",
            "--probability-tolerance",
            "0.1",
        ]
    )

    assert exit_code == 0
    ensemble = pd.read_csv(ensemble_path)
    assert ensemble["decoder"].unique().tolist() == [DEFAULT_ENSEMBLE_DECODER]
    assert np.allclose(ensemble[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)
