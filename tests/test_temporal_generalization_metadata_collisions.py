from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.temporal_generalization import TemporalFeatureWindow, compute_temporal_generalization_matrix


def _window(*, metadata=None) -> TemporalFeatureWindow:
    return TemporalFeatureWindow(
        center=0.0,
        features=np.asarray([[-1.0], [1.0]]),
        labels=np.asarray([0, 1]),
        metadata=metadata,
    )


def _compute(*, metadata=None, train_metadata=None, test_metadata=None, model_metadata=None):
    return compute_temporal_generalization_matrix(
        [_window(metadata=train_metadata)],
        [_window(metadata=test_metadata)],
        fit_model=lambda window: window,
        predict_labels=lambda _model, window: window.labels,
        metadata=metadata,
        model_metadata=model_metadata,
    )


def test_temporal_generalization_rejects_reserved_base_metadata_columns() -> None:
    with pytest.raises(ValueError, match=r"metadata.*reserved result columns: accuracy"):
        _compute(metadata={"accuracy": 0.0})


def test_temporal_generalization_rejects_reserved_train_window_metadata_columns() -> None:
    with pytest.raises(ValueError, match=r"train window metadata.*reserved result columns: train_window_center_s"):
        _compute(train_metadata={"train_window_center_s": 99.0})


def test_temporal_generalization_rejects_reserved_test_window_metadata_columns() -> None:
    with pytest.raises(ValueError, match=r"test window metadata.*reserved result columns: chance_accuracy"):
        _compute(test_metadata={"chance_accuracy": 0.0})


def test_temporal_generalization_rejects_reserved_model_metadata_columns() -> None:
    with pytest.raises(ValueError, match=r"model metadata.*reserved result columns: n_validation_trials"):
        _compute(model_metadata=lambda _model: {"n_validation_trials": 0})
