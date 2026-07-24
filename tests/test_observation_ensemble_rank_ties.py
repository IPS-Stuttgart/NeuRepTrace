from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.observation_ensemble import ensemble_probability_observations


def test_rank_ensemble_preserves_uniform_class_ties() -> None:
    observations = pd.DataFrame(
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
                "prob_class_0": 0.5,
                "prob_class_1": 0.5,
            }
            for decoder in ("source_a", "source_b")
        ]
    )

    ensemble = ensemble_probability_observations(
        observations,
        decoders=("source_a", "source_b"),
        weights=(1.0, 1.0),
        baseline_window=None,
        score_mode="rank",
    )

    np.testing.assert_allclose(
        ensemble[["prob_class_0", "prob_class_1"]].to_numpy(dtype=float),
        np.asarray([[0.5, 0.5]], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
