from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_consensus import SourceFreeConsensusVariant, fit_source_free_consensus_predict_proba


class _ToySourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.90, 0.10], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.35, 0.65]
        return probabilities


@pytest.mark.parametrize(
    ("variants", "expected_name"),
    [
        ("source_raw", "source_raw"),
        ({"name": "raw", "kwargs": {"max_iterations": 0, "target_prior_correction": "none"}}, "raw"),
        (SourceFreeConsensusVariant("raw", {"max_iterations": 0, "target_prior_correction": "none"}), "raw"),
    ],
)
def test_fit_source_free_consensus_accepts_single_variant_selector(variants, expected_name) -> None:
    result = fit_source_free_consensus_predict_proba(
        source_model=_ToySourceModel(),
        target_features=np.array([[-1.0], [0.5], [1.0]], dtype=float),
        variants=variants,
    )

    assert result.probabilities.shape == (3, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["source_free_consensus_n_variants"] == 1
    assert result.metadata["source_free_consensus_variants"] == expected_name
    assert np.allclose(result.weights, [1.0])
