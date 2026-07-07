from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401 - installs compatibility patches
from neureptrace.decoding.source_free import SourceFreeSubjectAdapter, _normalize_probability_rows


class BooleanProbabilitySourceModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([[True, False]] * features.shape[0], dtype=bool)


class CountingProbabilitySourceModel:
    classes_ = np.asarray([0, 1])

    def __init__(self) -> None:
        self.calls = 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.tile(np.asarray([[0.25, 0.75]], dtype=float), (features.shape[0], 1))


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


def test_source_free_probability_validation_preserves_single_predict_call() -> None:
    model = CountingProbabilitySourceModel()
    adapter = SourceFreeSubjectAdapter(
        source_model=model,
        max_iterations=0,
        prototype_weight=0.0,
    )

    adapter.fit(np.ones((3, 2), dtype=float))

    assert model.calls == 1
    assert np.allclose(adapter.probabilities_, np.asarray([[0.25, 0.75]] * 3))


def test_source_free_probability_validation_preserves_one_pass_iterables() -> None:
    def rows():
        yield (0.25, 0.75)
        yield (0.60, 0.40)

    normalized = _normalize_probability_rows(rows())

    assert np.allclose(normalized, np.asarray([[0.25, 0.75], [0.60, 0.40]]))
