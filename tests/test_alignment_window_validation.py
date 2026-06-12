from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from neureptrace.decoding.alignment_window import transform_with_alignment_projection


@dataclass(frozen=True)
class DummyFeatureSet:
    features: np.ndarray
    labels: np.ndarray
    n_channels: int
    n_window_samples: int
    feature_order: str = "channel_time"


def _feature_set(width: int = 2) -> DummyFeatureSet:
    return DummyFeatureSet(
        features=np.zeros((1, width), dtype=float),
        labels=np.array([1]),
        n_channels=width,
        n_window_samples=1,
    )


def test_cross_window_alignment_rejects_nonfinite_features() -> None:
    feature_set = _feature_set()

    with pytest.raises(ValueError, match="features contains non-finite values"):
        transform_with_alignment_projection(
            np.array([[np.nan, 1.0]]),
            decode_feature_set=feature_set,
            projection=np.eye(2),
            projection_feature_mean=np.zeros(2),
            projection_feature_set=feature_set,
        )


def test_cross_window_alignment_rejects_nonfinite_projection_mean() -> None:
    feature_set = _feature_set()

    with pytest.raises(ValueError, match="projection_feature_mean contains non-finite values"):
        transform_with_alignment_projection(
            np.array([[1.0, 2.0]]),
            decode_feature_set=feature_set,
            projection=np.eye(2),
            projection_feature_mean=np.array([0.0, np.inf]),
            projection_feature_set=feature_set,
        )


def test_cross_window_alignment_rejects_empty_projection() -> None:
    feature_set = _feature_set()

    with pytest.raises(ValueError, match="projection must have at least one row and one column"):
        transform_with_alignment_projection(
            np.array([[1.0, 2.0]]),
            decode_feature_set=feature_set,
            projection=np.empty((0, 2)),
            projection_feature_mean=np.zeros(2),
            projection_feature_set=feature_set,
        )
