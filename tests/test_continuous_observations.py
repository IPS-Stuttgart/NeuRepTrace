from __future__ import annotations

import pandas as pd

from neureptrace.continuous_observations import standardize_continuous_observations


def test_standardize_continuous_observations_preserves_typed_sequence_identity() -> None:
    observations = pd.DataFrame(
        {
            "stream_id": pd.Series([1, "1"], dtype=object),
            "sample_index": [7, 7],
            "time": [0.0, 0.0],
            "prob_class_0": [1.0, 1.0],
        }
    )

    standardized = standardize_continuous_observations(
        observations,
        subject="subject",
        split_id="split",
        slice_seed=13,
        decoder="logistic",
        emission_mode="calibrated",
        train_time=0.15,
        preprocessing_hash="preprocessing",
        model_hash="model",
    )

    assert standardized["sequence_id"].tolist() == [(1, 7), ("1", 7)]
    assert standardized["sequence_id"].is_unique
