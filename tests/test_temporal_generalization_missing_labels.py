from __future__ import annotations

import numpy as np

from neureptrace.decoding.temporal_generalization import (
    TemporalFeatureWindow,
    compute_temporal_generalization_matrix,
)


def test_temporal_generalization_matches_missing_class_labels() -> None:
    train_window = TemporalFeatureWindow(
        center=0.0,
        features=np.zeros((2, 1)),
        labels=[float("nan"), "known"],
    )
    test_window = TemporalFeatureWindow(
        center=0.0,
        features=np.zeros((2, 1)),
        labels=[float("nan"), "known"],
    )

    rows = compute_temporal_generalization_matrix(
        [train_window],
        [test_window],
        fit_model=lambda window: window,
        predict_labels=lambda _model, _window: [np.float64("nan"), "known"],
    )

    assert rows.loc[0, "accuracy"] == 1.0
    assert rows.loc[0, "chance_accuracy"] == 0.5
    assert rows.loc[0, "n_train_classes"] == 2
    assert rows.loc[0, "n_validation_classes"] == 2
