from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_alignment import _target_alignment_matrix, source_alignment_config


def _target_features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    features = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1])
    return features, labels


@pytest.mark.parametrize(
    "bad_offsets",
    [
        {0: np.array([0.0, 1.5]), 1: np.array([0, 1])},
        {0: np.array([0, np.nan]), 1: np.array([0, 1])},
        {0: np.array([False, True]), 1: np.array([0, 1])},
    ],
)
def test_target_repetition_offsets_reject_noninteger_offsets(bad_offsets) -> None:
    features, labels = _target_features_and_labels()

    with pytest.raises(ValueError, match="integer offsets"):
        _target_alignment_matrix(
            features,
            labels,
            classes=np.array([0, 1]),
            config=source_alignment_config(method="mcca", anchor_mode="class_repetition"),
            n_repetitions_per_class=2,
            selected_offsets_by_class=bad_offsets,
        )


def test_target_repetition_offsets_keep_integer_strings_compatible() -> None:
    features, labels = _target_features_and_labels()

    matrix = _target_alignment_matrix(
        features,
        labels,
        classes=np.array([0, 1]),
        config=source_alignment_config(method="mcca", anchor_mode="class_repetition"),
        n_repetitions_per_class=2,
        selected_offsets_by_class={0: np.array(["0", "2"]), 1: np.array(["1", "2"])},
    )

    np.testing.assert_array_equal(matrix, np.vstack([features[[0, 2]], features[[4, 5]]]))
