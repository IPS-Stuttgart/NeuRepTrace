from __future__ import annotations

import numpy as np

from neureptrace.decoding._domain_ids import values_equal
from neureptrace.decoding.class_balanced_jda import class_balanced_source_indices, fit_class_balanced_jda


_SOURCE_FEATURES = np.asarray(
    [
        [0.0, 0.0],
        [0.1, 0.0],
        [3.0, 3.0],
        [3.1, 3.0],
    ],
    dtype=float,
)
_SOURCE_LABELS = [float("nan"), float("nan"), "right", "right"]
_TARGET_FEATURES = np.asarray([[0.05, 0.0], [3.05, 3.0]], dtype=float)


def test_class_balanced_source_indices_treats_distinct_nan_labels_as_one_class() -> None:
    result = class_balanced_source_indices(_SOURCE_LABELS, random_state=0)

    assert len(result.classes) == 2
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "right"
    assert result.original_counts == (2, 2)
    assert result.balanced_counts == (2, 2)
    assert result.indices.shape == (4,)


def test_fit_class_balanced_jda_round_trips_missing_class_labels() -> None:
    result = fit_class_balanced_jda(
        _SOURCE_FEATURES,
        _SOURCE_LABELS,
        _TARGET_FEATURES,
        balance_random_state=0,
        n_components=1,
        max_iterations=2,
    )

    assert len(result.classes) == 2
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "right"
    assert result.source_features.shape == (4, 1)
    assert result.target_features.shape == (2, 1)
    assert np.all(np.isfinite(result.source_features))
    assert np.all(np.isfinite(result.target_features))
    assert all(any(values_equal(label, class_label) for class_label in result.classes) for label in result.target_pseudo_labels)
    assert result.metadata["class_balanced_jda_original_counts"] == "2|2"
    assert result.metadata["class_balanced_jda_balanced_counts"] == "2|2"
