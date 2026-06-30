from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.label_shift import adapt_label_shift_probabilities


def test_label_shift_source_prior_mapping_rejects_missing_class_key() -> None:
    with pytest.raises(ValueError, match="source_prior mapping must provide a prior for every class"):
        adapt_label_shift_probabilities(
            [[0.9, 0.1], [0.8, 0.2]],
            source_prior={"yes": 1.0},
            classes=["yes", "no"],
        )


def test_label_shift_source_prior_mapping_preserves_explicit_class_order() -> None:
    result = adapt_label_shift_probabilities(
        [[0.9, 0.1], [0.8, 0.2]],
        source_prior={"yes": 0.25, "no": 0.75},
        classes=["yes", "no"],
    )

    assert np.allclose(result.source_prior, (0.25, 0.75))


def test_label_shift_rejects_boolean_target_probabilities() -> None:
    with pytest.raises(ValueError, match="target_probabilities must be numeric probability values, not boolean"):
        adapt_label_shift_probabilities(
            np.array([[True, False], [False, True]]),
            source_prior=[0.5, 0.5],
        )


def test_label_shift_rejects_boolean_bbse_validation_probabilities() -> None:
    with pytest.raises(ValueError, match="source_validation_probabilities must be numeric probability values, not boolean"):
        adapt_label_shift_probabilities(
            [[0.55, 0.45], [0.35, 0.65]],
            method="bbse",
            source_prior=[0.5, 0.5],
            source_validation_probabilities=np.array([[True, False], [False, True]]),
            source_validation_labels=[0, 1],
            classes=[0, 1],
        )


def test_label_shift_rejects_zero_mass_target_probability_rows() -> None:
    with pytest.raises(ValueError, match="Probability rows must have positive mass"):
        adapt_label_shift_probabilities(
            [[0.0, 0.0], [0.7, 0.3]],
            source_prior=[0.5, 0.5],
        )


def test_label_shift_rejects_zero_mass_source_prior() -> None:
    with pytest.raises(ValueError, match="Prior vectors must have positive mass"):
        adapt_label_shift_probabilities(
            [[0.8, 0.2], [0.6, 0.4]],
            source_prior=[0.0, 0.0],
        )
