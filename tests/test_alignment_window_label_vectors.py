from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from neureptrace.decoding.alignment_window import validate_paired_feature_sets


@dataclass(frozen=True)
class DummyFeatureSet:
    features: np.ndarray
    labels: object
    n_channels: int
    n_window_samples: int


def test_validate_paired_feature_sets_preserves_tuple_labels() -> None:
    labels = [("run-01", "famous"), ("run-01", "scrambled")]
    decode = DummyFeatureSet(np.zeros((2, 4)), labels, n_channels=2, n_window_samples=2)
    alignment = DummyFeatureSet(np.zeros((2, 6)), list(labels), n_channels=2, n_window_samples=3)

    validate_paired_feature_sets(decode, alignment)


def test_validate_paired_feature_sets_rejects_label_row_count_mismatch() -> None:
    decode = DummyFeatureSet(
        np.zeros((3, 4)),
        [("run-01", "famous"), ("run-01", "scrambled")],
        n_channels=2,
        n_window_samples=2,
    )
    alignment = DummyFeatureSet(
        np.zeros((3, 6)),
        [("run-01", "famous"), ("run-01", "scrambled")],
        n_channels=2,
        n_window_samples=3,
    )

    with pytest.raises(ValueError, match="Decoding labels row count differs from feature rows"):
        validate_paired_feature_sets(decode, alignment, participant=3)
