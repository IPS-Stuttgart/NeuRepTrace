from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_selection import select_target_confident_predictions


def test_composite_classes_are_preserved_as_labels() -> None:
    tuple_classes = [("face", "upright"), ("face", "inverted")]
    tuple_result = select_target_confident_predictions(
        [[0.8, 0.2], [0.1, 0.9]],
        classes=tuple_classes,
    )

    assert tuple_result.classes.tolist() == tuple_classes
    assert tuple_result.predictions.tolist() == tuple_classes

    matrix_result = select_target_confident_predictions(
        [[0.8, 0.2], [0.1, 0.9]],
        classes=np.asarray([[1, 10], [2, 20]]),
    )

    assert matrix_result.classes.tolist() == [(1, 10), (2, 20)]
    assert matrix_result.predictions.tolist() == [(1, 10), (2, 20)]


def test_duplicate_nan_classes_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_target_confident_predictions([[0.8, 0.2]], classes=[np.nan, float("nan")])
