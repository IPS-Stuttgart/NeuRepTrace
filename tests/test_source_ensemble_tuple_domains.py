from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_ensemble import fit_source_domain_probability_ensemble


def test_tuple_domains_are_masked_atomically() -> None:
    x = np.asarray([[-2.0, 0.0], [-1.6, 0.2], [1.7, -0.1], [2.1, 0.1], [-1.8, 3.0], [-1.4, 3.2], [1.8, 2.8], [2.2, 3.1]])
    y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=object)
    domains = [("a", 1), ("a", 1), ("a", 1), ("a", 1), ("b", 2), ("b", 2), ("b", 2), ("b", 2)]
    target = np.asarray([[-1.7, 0.1], [1.9, 0.0]])

    result = fit_source_domain_probability_ensemble(
        source_features=x,
        source_labels=y,
        source_domains=domains,
        target_features=target,
        weighting="target_feature_similarity",
    )

    assert set(result.models) == {("a", 1), ("b", 2)}
    assert result.models[("a", 1)].n_rows == 4
    assert result.models[("b", 2)].n_rows == 4
    assert result.metadata["source_domain_ensemble_n_source_domains"] == 2
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
