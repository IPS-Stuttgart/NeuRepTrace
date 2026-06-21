from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neureptrace.decoding.alignment_window import validate_paired_feature_sets


@dataclass(frozen=True)
class DummyFeatureSet:
    features: np.ndarray
    labels: np.ndarray
    n_channels: int
    n_window_samples: int


def test_validate_paired_feature_sets_flattens_single_column_label_arrays() -> None:
    decode = DummyFeatureSet(
        np.zeros((3, 2)),
        np.array([["trial-a"], ["trial-b"], ["trial-c"]], dtype=object),
        n_channels=2,
        n_window_samples=1,
    )
    alignment = DummyFeatureSet(
        np.zeros((3, 4)),
        np.array(["trial-a", "trial-b", "trial-c"], dtype=object),
        n_channels=2,
        n_window_samples=2,
    )

    validate_paired_feature_sets(decode, alignment)


def test_validate_paired_feature_sets_preserves_multi_column_anchor_rows() -> None:
    decode = DummyFeatureSet(
        np.zeros((3, 2)),
        np.array([("run-1", "stim-a"), ("run-1", "stim-b"), ("run-2", "stim-a")], dtype=object),
        n_channels=2,
        n_window_samples=1,
    )
    alignment = DummyFeatureSet(
        np.zeros((3, 4)),
        np.array([["run-1", "stim-a"], ["run-1", "stim-b"], ["run-2", "stim-a"]], dtype=object),
        n_channels=2,
        n_window_samples=2,
    )

    validate_paired_feature_sets(decode, alignment)
