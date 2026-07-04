from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.class_prior import compute_source_class_prior, normalize_prior_mode


def test_compute_source_class_prior_preserves_rectangular_numpy_composite_labels() -> None:
    source_labels = np.asarray([("left", 1), ("right", 2), ("left", 1)], dtype=object)
    classes = np.asarray([("left", 1), ("right", 2)], dtype=object)

    result = compute_source_class_prior(source_labels, classes=classes)

    assert result.classes.tolist() == [("left", 1), ("right", 2)]
    assert result.counts.tolist() == [2, 1]
    assert np.allclose(result.prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))
    assert result.metadata["source_prior_n_rows"] == 3
    assert result.metadata["source_prior_n_classes"] == 2
    assert "('left', 1):2" in result.metadata["source_prior_class_counts"]


def test_compute_source_class_prior_infers_composite_classes_in_first_seen_order() -> None:
    source_labels = np.asarray([("right", 2), ("left", 1), ("right", 2)], dtype=object)

    result = compute_source_class_prior(source_labels)

    assert result.classes.tolist() == [("right", 2), ("left", 1)]
    assert result.counts.tolist() == [2, 1]


def test_compute_source_class_prior_rejects_duplicate_composite_classes() -> None:
    source_labels = np.asarray([("left", 1), ("right", 2)], dtype=object)
    classes = np.asarray([("left", 1), ("left", 1)], dtype=object)

    with pytest.raises(ValueError, match="classes must be unique"):
        compute_source_class_prior(source_labels, classes=classes)


def test_normalize_prior_mode_aliases() -> None:
    assert normalize_prior_mode("flat") == "uniform"
    assert normalize_prior_mode("counts") == "empirical"
