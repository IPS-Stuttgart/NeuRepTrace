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
