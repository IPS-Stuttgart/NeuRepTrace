from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.observation_ensemble import DEFAULT_ENSEMBLE_DECODER, main


def test_observation_ensemble_cli_forwards_probability_tolerance(tmp_path: Path) -> None:
    observations = pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": "logistic",
                "backend": "sklearn",
                "emission_mode": "calibrated",
                "time": 0.10,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 0,
                "true_class": "zero",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": 0.55,
                "prob_class_1": 0.50,
            },
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": "linear_svm",
                "backend": "sklearn",
                "emission_mode": "calibrated",
                "time": 0.10,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 0,
                "true_class": "zero",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": 0.60,
                "prob_class_1": 0.40,
            },
        ]
    )

    source_path = tmp_path / "source_observations.csv"
    ensemble_path = tmp_path / "ensemble_observations.csv"
    observations.to_csv(source_path, index=False)

    exit_code = main(
        [
            str(source_path),
            "--out",
            str(ensemble_path),
            "--no-baseline-debiasing",
            "--probability-tolerance",
            "0.1",
        ]
    )

    assert exit_code == 0
    ensemble = pd.read_csv(ensemble_path)
    assert ensemble["decoder"].unique().tolist() == [DEFAULT_ENSEMBLE_DECODER]
    assert np.allclose(ensemble[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)
