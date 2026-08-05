from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.joint_distribution_adaptation import fit_joint_distribution_adaptation


_SOURCE_FEATURES = np.asarray(
    [
        [0.0, 0.1],
        [0.2, 0.0],
        [1.0, 1.1],
        [1.2, 1.0],
    ],
    dtype=float,
)
_SOURCE_LABELS = np.asarray(["left", "left", "right", "right"], dtype=object)
_TARGET_FEATURES = np.asarray([[0.1, 0.2], [1.1, 0.9]], dtype=float)
_CLASSES_WITHOUT_SOURCE_ROWS = ("left", "right", "unused")


def _fit_with_extra_class(**kwargs: object) -> None:
    fit_joint_distribution_adaptation(
        _SOURCE_FEATURES,
        _SOURCE_LABELS,
        _TARGET_FEATURES,
        classes=_CLASSES_WITHOUT_SOURCE_ROWS,
        n_components=1,
        max_iterations=1,
        **kwargs,
    )


def test_jda_rejects_supplied_class_without_source_rows_before_centroid_initialization() -> None:
    with pytest.raises(ValueError, match="absent from source_labels"):
        _fit_with_extra_class()


def test_jda_rejects_supplied_class_without_source_rows_with_initial_probabilities() -> None:
    target_probabilities = np.asarray(
        [
            [0.45, 0.45, 0.10],
            [0.10, 0.45, 0.45],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="absent from source_labels"):
        _fit_with_extra_class(target_probabilities=target_probabilities)
