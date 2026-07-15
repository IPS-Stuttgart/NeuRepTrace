from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity


def test_mmd_source_selection_preserves_large_finite_domain_distances() -> None:
    source_features = np.asarray(
        [
            [-1.0e200, -1.0e200],
            [-1.0e200, -1.0e200],
            [1.0e200, 1.0e200],
            [1.0e200, 1.0e200],
        ]
    )
    source_domains = np.asarray(["a_wrong", "a_wrong", "z_match", "z_match"], dtype=object)
    target_features = np.asarray([[1.0e200, 1.0e200], [1.0e200, 1.0e200]])

    with np.errstate(over="raise", invalid="raise"):
        result = select_source_domains_by_target_similarity(
            source_features,
            source_domains,
            target_features,
            metric="mmd",
            top_k=1,
        )

    assert result.selected_domains == ("z_match",)
    assert result.domain_distances["z_match"] == 0.0
    assert result.domain_distances["a_wrong"] > result.domain_distances["z_match"]
    assert all(np.isfinite(distance) for distance in result.domain_distances.values())
