from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_ensemble import fit_source_domain_probability_ensemble


@pytest.mark.parametrize("temporal_nat", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_source_ensemble_keeps_temporal_nat_class_separate_from_none(temporal_nat: object) -> None:
    result = fit_source_domain_probability_ensemble(
        source_features=[[-2.0], [-1.5], [1.5], [2.0]],
        source_labels=[None, None, temporal_nat, temporal_nat],
        source_domains=["source"] * 4,
        target_features=[[-1.8], [1.8]],
    )

    assert result.classes.shape == (2,)
    assert result.classes[0] is None
    assert type(result.classes[1]) is type(temporal_nat)
    assert np.isnat(result.classes[1])
    assert result.metadata["source_domain_ensemble_n_classes"] == 2
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("temporal_nat", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_source_ensemble_keeps_temporal_nat_domain_separate_from_none(temporal_nat: object) -> None:
    result = fit_source_domain_probability_ensemble(
        source_features=[
            [-2.0],
            [-1.5],
            [1.5],
            [2.0],
            [-2.2],
            [-1.7],
            [1.7],
            [2.2],
        ],
        source_labels=[0, 0, 1, 1, 0, 0, 1, 1],
        source_domains=[None, None, None, None, temporal_nat, temporal_nat, temporal_nat, temporal_nat],
        target_features=[[-1.8], [1.8]],
    )

    domains = list(result.models)
    assert len(domains) == 2
    assert domains[0] is None
    assert type(domains[1]) is type(temporal_nat)
    assert np.isnat(domains[1])
    assert result.models[domains[0]].n_rows == 4
    assert result.models[domains[1]].n_rows == 4
    assert result.metadata["source_domain_ensemble_n_source_domains"] == 2
    assert result.metadata["source_domain_ensemble_n_trained_domains"] == 2
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
