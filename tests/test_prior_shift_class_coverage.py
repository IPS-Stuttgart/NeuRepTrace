from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.prior_shift import prior_from_labels


def test_prior_from_labels_rejects_observed_labels_missing_from_explicit_classes() -> None:
    with pytest.raises(ValueError, match="labels contain labels absent from classes: 'other'"):
        prior_from_labels(
            ["left", "left", "right", "other"],
            classes=["left", "right"],
        )


def test_prior_from_labels_rejects_duplicate_explicit_classes() -> None:
    with pytest.raises(ValueError, match="classes must be unique"):
        prior_from_labels(
            ["left", "right"],
            classes=["left", "left", "right"],
        )


def test_prior_from_labels_preserves_complete_explicit_class_order() -> None:
    prior, classes = prior_from_labels(
        ["left", "left", "right", "left"],
        classes=["right", "left"],
    )

    assert classes == ("right", "left")
    np.testing.assert_allclose(prior, [0.25, 0.75])
