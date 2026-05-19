from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.mne_time_decode import _resolve_observation_out_path
from neureptrace.observations import summarize_decoding_observations


def test_summarize_decoding_observations_rebuilds_metrics():
    observations = pd.DataFrame(
        {
            "fold": [0, 0, 0, 1, 1, 1],
            "decoder": ["logistic"] * 6,
            "emission_mode": ["calibrated"] * 6,
            "time": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "test_time": [0.1] * 6,
            "train_time": [0.1] * 6,
            "true_label": [0, 1, 1, 0, 0, 1],
            "class_0": ["left"] * 6,
            "class_1": ["right"] * 6,
            "prob_class_0": [0.8, 0.3, 0.4, 0.7, 0.6, 0.2],
            "prob_class_1": [0.2, 0.7, 0.6, 0.3, 0.4, 0.8],
        }
    )

    summary = summarize_decoding_observations(observations)

    assert summary["fold"].tolist() == [0, 1]
    assert summary["n_test"].tolist() == [3, 3]
    assert summary["n_classes"].tolist() == [2, 2]
    assert summary["class_names"].tolist() == ["left|right", "left|right"]
    np.testing.assert_allclose(summary["accuracy"], [1.0, 2.0 / 3.0])
    assert {"log_loss", "brier", "ece"}.issubset(summary.columns)


def test_cli_resolves_default_canonical_observation_path(tmp_path: Path):
    out = tmp_path / "decode.csv"
    explicit = tmp_path / "custom_observations.csv"

    assert _resolve_observation_out_path(out, None, False) == tmp_path / "decode_observations.csv"
    assert _resolve_observation_out_path(out, explicit, False) == explicit
    assert _resolve_observation_out_path(out, explicit, True) is None
