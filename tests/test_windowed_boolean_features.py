from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.windowed import fit_window_model, score_windowed_decoding


def _unexpected_fit_model(features: np.ndarray, labels: np.ndarray) -> object:
    raise AssertionError("feature validation must run before model fitting")


@pytest.mark.parametrize(
    "train_features",
    [
        np.asarray([[False], [True]], dtype=bool),
        [[False], [True]],
        np.asarray([[np.asarray(False)], [np.asarray(True)]], dtype=object),
    ],
)
def test_fit_window_model_rejects_boolean_feature_matrices(train_features: object) -> None:
    with pytest.raises(ValueError, match="train_features .* non-boolean"):
        fit_window_model(
            train_features,
            np.asarray([0, 1]),
            fit_model=_unexpected_fit_model,
        )


def test_fit_window_model_rejects_nested_boolean_generators() -> None:
    train_features = ((value for value in row) for row in [[False], [True]])

    with pytest.raises(ValueError, match="train_features .* non-boolean"):
        fit_window_model(
            train_features,
            np.asarray([0, 1]),
            fit_model=_unexpected_fit_model,
        )


def test_score_windowed_decoding_rejects_boolean_validation_features() -> None:
    with pytest.raises(ValueError, match="validation_features .* non-boolean"):
        score_windowed_decoding(
            train_features=np.asarray([[-1.0], [1.0]]),
            train_labels=np.asarray([0, 1]),
            validation_features=np.asarray([[False], [True]], dtype=bool),
            validation_labels=np.asarray([0, 1]),
            fit_model=_unexpected_fit_model,
        )
