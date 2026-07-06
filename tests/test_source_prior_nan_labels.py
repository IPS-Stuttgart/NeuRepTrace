from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior, estimate_source_class_prior


def test_source_prior_matches_explicit_nan_class_labels() -> None:
    source_labels = [np.nan, "seen", np.float64("nan")]
    classes = [np.nan, "seen"]

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)
    result = adjust_probabilities_to_source_prior(
        [[0.8, 0.2]],
        source_labels=source_labels,
        classes=classes,
        config={"target_prior": "source"},
    )

    assert inferred_classes.shape == (2,)
    assert np.isnan(inferred_classes[0])
    assert inferred_classes[1] == "seen"
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))
    np.testing.assert_allclose(result.source_prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


def test_source_prior_collapses_repeated_nan_labels() -> None:
    prior, classes = estimate_source_class_prior([np.nan, np.float64("nan"), "other"])

    assert classes.shape == (2,)
    assert np.isnan(classes[0])
    assert classes[1] == "other"
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


def test_source_prior_matches_composite_nan_labels() -> None:
    nan_class = ("missing", np.nan)
    other_class = ("other", 1.0)

    prior, classes = estimate_source_class_prior(
        [("missing", np.nan), ("missing", np.float64("nan")), other_class],
        classes=[nan_class, other_class],
    )

    assert classes.tolist()[0][0] == "missing"
    assert np.isnan(classes.tolist()[0][1])
    assert classes.tolist()[1] == other_class
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))
