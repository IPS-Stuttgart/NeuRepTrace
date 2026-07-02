from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_consensus import SourceFreeConsensusVariant, combine_probability_variants, fit_source_free_consensus_predict_proba


class _ToySourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.90, 0.10], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.35, 0.65]
        return probabilities


def _one_pass(rows):
    return (row for row in rows)


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


def test_combine_probability_variants_accepts_one_pass_rows_and_weights() -> None:
    first = _one_pass(([0.80, 0.20], [0.40, 0.60]))
    second = _one_pass(([0.60, 0.40], [0.20, 0.80]))
    variants = (matrix for matrix in (first, second))
    weights = (weight for weight in (0.25, 0.75))

    result = combine_probability_variants(variants, weights=weights, mode="arithmetic_mean")

    assert np.allclose(result, [[0.65, 0.35], [0.25, 0.75]])
    assert np.allclose(result.sum(axis=1), 1.0)


def test_combine_probability_variants_rejects_empty_one_pass_variants() -> None:
    with pytest.raises(ValueError, match="At least one probability matrix"):
        combine_probability_variants((matrix for matrix in ()))
