from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401 - installs compatibility patches
from neureptrace.decoding.source_free import SourceFreeSubjectAdapter, _normalize_probability_rows


class BooleanProbabilitySourceModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([[True, False]] * features.shape[0], dtype=bool)


def test_source_free_rejects_boolean_source_probabilities() -> None:
    adapter = SourceFreeSubjectAdapter(
        source_model=BooleanProbabilitySourceModel(),
        max_iterations=0,
        prototype_weight=0.0,
    )

    with pytest.raises(ValueError, match="boolean"):
        adapter.fit(np.ones((3, 2), dtype=float))


def test_source_free_rejects_boolean_probability_rows_directly() -> None:
    with pytest.raises(ValueError, match="boolean"):
        _normalize_probability_rows([[True, False], [False, True]])
