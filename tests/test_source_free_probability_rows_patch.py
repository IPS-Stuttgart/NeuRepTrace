from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free import SourceFreeSubjectAdapter, fit_source_free_predict_proba


class _ShortProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([[0.65, 0.35]], dtype=float)


class _ChangingProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self) -> None:
        self.calls = 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.calls += 1
        rows = features.shape[0] if self.calls == 1 else max(1, features.shape[0] - 1)
        return np.tile(np.array([[0.60, 0.40]], dtype=float), (rows, 1))


class _NegativeProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[-0.20, 1.20]], dtype=float), (features.shape[0], 1))


class _TinyNegativeResidueModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[-1e-12, 1.0 + 1e-12]], dtype=float), (features.shape[0], 1))


def test_source_free_fit_rejects_probability_row_mismatch() -> None:
    with pytest.raises(ValueError, match="probability rows must match target_features rows"):
        fit_source_free_predict_proba(
            source_model=_ShortProbabilityModel(),
            target_features=np.zeros((3, 2), dtype=float),
            max_iterations=0,
        )


def test_source_free_predict_rejects_probability_row_mismatch() -> None:
    model = _ChangingProbabilityModel()
    adapter = SourceFreeSubjectAdapter(source_model=model, max_iterations=0).fit(np.zeros((3, 2), dtype=float))

    with pytest.raises(ValueError, match="probability rows must match target_features rows"):
        adapter.predict_proba(np.ones((3, 2), dtype=float))


def test_source_free_fit_rejects_invalid_negative_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities must be non-negative"):
        fit_source_free_predict_proba(
            source_model=_NegativeProbabilityModel(),
            target_features=np.zeros((3, 2), dtype=float),
            max_iterations=0,
        )


def test_source_free_fit_allows_tiny_negative_probability_residue() -> None:
    result = fit_source_free_predict_proba(
        source_model=_TinyNegativeResidueModel(),
        target_features=np.zeros((3, 2), dtype=float),
        max_iterations=0,
    )

    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.all(result.probabilities >= 0.0)
