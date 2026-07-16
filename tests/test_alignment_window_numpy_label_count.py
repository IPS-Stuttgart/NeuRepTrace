from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from neureptrace.decoding.alignment_window import validate_paired_feature_sets


@dataclass(frozen=True)
class DummyFeatureSet:
    features: np.ndarray
    labels: np.ndarray
    n_channels: int
    n_window_samples: int


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["face", "scrambled"], dtype=object),
        np.array([["face"], ["scrambled"]], dtype=object),
    ],
)
def test_validate_paired_feature_sets_rejects_short_numpy_label_vectors(labels: np.ndarray) -> None:
    decode = DummyFeatureSet(
        features=np.zeros((3, 4)),
        labels=labels,
        n_channels=2,
        n_window_samples=2,
    )
    alignment = DummyFeatureSet(
        features=np.zeros((3, 6)),
        labels=labels.copy(),
        n_channels=2,
        n_window_samples=3,
    )

    with pytest.raises(ValueError, match="Decoding labels row count differs from feature rows for participant 7"):
        validate_paired_feature_sets(decode, alignment, participant=7)
