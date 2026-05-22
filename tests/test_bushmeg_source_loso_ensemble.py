from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_source_loso_ensemble import (
    _fit_stacking_weights,
    _normalize_ensemble_class_bias,
    _normalize_weighting,
)


def test_stacked_weighting_is_configurable():
    assert _normalize_weighting("stacked") == "stacked"
    assert _normalize_ensemble_class_bias("balanced-acc") == "balanced_accuracy"


def test_fit_stacking_weights_prefers_source_oof_winner():
    labels = np.array([0, 1, 0, 1, 0, 1])
    good = np.array(
        [
            [0.90, 0.10],
            [0.15, 0.85],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.75, 0.25],
            [0.25, 0.75],
        ]
    )
    bad = 1.0 - good
    cube = np.stack([good, bad], axis=0)

    weights = _fit_stacking_weights(cube, labels, n_classes=2, max_iter=200)

    assert weights.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > 0.95
    assert weights[1] < 0.05
