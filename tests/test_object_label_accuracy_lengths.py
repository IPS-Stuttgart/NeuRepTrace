from __future__ import annotations

import numpy as np
import pytest

from neureptrace._object_label_utils import label_accuracy


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([], ["unexpected"]),
        (["expected"], []),
    ],
)
def test_label_accuracy_rejects_length_mismatches(labels, predictions) -> None:
    with pytest.raises(ValueError, match="labels and predictions must have the same length"):
        label_accuracy(labels, predictions)


def test_label_accuracy_keeps_empty_empty_result_undefined() -> None:
    assert np.isnan(label_accuracy([], []))
