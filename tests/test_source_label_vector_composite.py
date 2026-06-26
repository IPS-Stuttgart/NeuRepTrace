from __future__ import annotations

import numpy as np
import pytest

from neureptrace._source_label_vector_patch import _as_label_vector


def test_source_label_vector_keeps_composite_rows_atomic() -> None:
    labels = np.asarray([
        ("face", "left"),
        ("house", "right"),
        ("face", "left"),
    ], dtype=object)

    normalized = _as_label_vector(labels, n_rows=3, row_error="row mismatch", shape_error="shape mismatch")

    assert normalized.shape == (3,)
    assert normalized.dtype == object
    assert normalized.tolist() == [("face", "left"), ("house", "right"), ("face", "left")]


def test_source_label_vector_flattens_single_column_labels() -> None:
    labels = np.asarray([["face"], ["house"], ["face"]], dtype=object)

    normalized = _as_label_vector(labels, n_rows=3, row_error="row mismatch", shape_error="shape mismatch")

    assert normalized.shape == (3,)
    assert normalized.tolist() == ["face", "house", "face"]


def test_source_label_vector_rejects_row_mismatch() -> None:
    labels = np.asarray([("face", "left"), ("house", "right")], dtype=object)

    with pytest.raises(ValueError, match="row mismatch"):
        _as_label_vector(labels, n_rows=3, row_error="row mismatch", shape_error="shape mismatch")
