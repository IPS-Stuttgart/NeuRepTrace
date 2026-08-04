from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior, estimate_source_class_prior


def _label(*parts: str):
    return (part for part in parts)


def test_source_prior_groups_equal_generator_backed_labels() -> None:
    prior, classes = estimate_source_class_prior(
        [_label("subject-1", "run-1"), _label("subject-1", "run-1"), _label("subject-2", "run-2")]
    )

    assert classes.tolist() == [("subject-1", "run-1"), ("subject-2", "run-2")]
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32))


def test_source_prior_rejects_duplicate_generator_backed_classes() -> None:
    with pytest.raises(ValueError, match="classes must be unique"):
        estimate_source_class_prior(
            [_label("subject-1", "run-1")],
            classes=[_label("subject-1", "run-1"), _label("subject-1", "run-1")],
        )


def test_source_prior_adjustment_retains_materialized_generator_classes() -> None:
    result = adjust_probabilities_to_source_prior(
        [[0.8, 0.2]],
        source_labels=[_label("subject-1", "run-1"), _label("subject-2", "run-2")],
        classes=[_label("subject-1", "run-1"), _label("subject-2", "run-2")],
        config={"target_prior": "source"},
    )

    assert result.classes.tolist() == [("subject-1", "run-1"), ("subject-2", "run-2")]
    np.testing.assert_allclose(result.probabilities, np.asarray([[0.8, 0.2]], dtype=np.float32))
