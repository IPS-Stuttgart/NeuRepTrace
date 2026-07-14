from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.observation_ensemble import ensemble_probability_observations


def _two_decoder_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 0,
                "true_class": "zero",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": probability_zero,
                "prob_class_1": 1.0 - probability_zero,
            }
            for decoder, probability_zero in (("source_a", 0.9), ("source_b", 0.1))
        ]
    )


def test_observation_ensemble_normalizes_extreme_finite_weights_without_overflow() -> None:
    observations = _two_decoder_observations()
    maximum = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        extreme = ensemble_probability_observations(
            observations,
            decoders=("source_a", "source_b"),
            weights=(maximum, maximum / 3.0),
            baseline_window=None,
            score_mode="probability",
        )

    ordinary = ensemble_probability_observations(
        observations,
        decoders=("source_a", "source_b"),
        weights=(3.0, 1.0),
        baseline_window=None,
        score_mode="probability",
    )

    np.testing.assert_allclose(
        extreme[["prob_class_0", "prob_class_1"]],
        ordinary[["prob_class_0", "prob_class_1"]],
    )
    assert extreme["ensemble_weights"].unique().tolist() == ["0.75|0.25"]
