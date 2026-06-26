from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401  # installs runtime compatibility patches
import neureptrace.decoding.few_shot as few_shot


def test_few_shot_column_vector_labels_remain_scalar_labels() -> None:
    labels = np.asarray([["face"], ["face"], ["face"], ["standard"], ["standard"], ["standard"]], dtype=object)

    label_vector = few_shot._as_1d_object_array(labels, name="labels")

    assert label_vector.shape == (6,)
    assert label_vector.tolist() == ["face", "face", "face", "standard", "standard", "standard"]


def test_few_shot_probability_alignment_accepts_column_vector_class_order() -> None:
    class Model:
        pass

    model = Model()
    model.classes_ = np.asarray(["rare", "standard"], dtype=object)

    aligned = few_shot._align_probability_columns(
        np.asarray([[0.25, 0.75]]),
        model=model,
        classes=np.asarray([["rare"], ["standard"]], dtype=object),
    )

    assert np.allclose(aligned, [[0.25, 0.75]])
