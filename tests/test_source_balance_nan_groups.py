from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_balance import compute_source_balance_weights, summarize_source_groups


def test_source_balance_groups_nan_labels_together() -> None:
    labels = np.asarray([float("nan"), np.float64(np.nan), "target"], dtype=object)

    result = summarize_source_groups(labels, strategy="class")

    assert sorted(result.group_counts.values()) == [1, 2]
    assert len(result.group_counts) == 2
    nan_keys = [key for key in result.group_counts if isinstance(key, float) and np.isnan(key)]
    assert len(nan_keys) == 1
    assert result.group_counts[nan_keys[0]] == 2

    weights = compute_source_balance_weights(labels, config={"strategy": "class", "target": "max"})

    np.testing.assert_allclose(weights.sample_weights, np.asarray([0.75, 0.75, 1.5], dtype=np.float32))


def test_source_balance_groups_nan_domains_inside_class_domain_keys() -> None:
    labels = np.asarray(["target", "target", "target"], dtype=object)
    domains = np.asarray([float("nan"), np.float64(np.nan), "known"], dtype=object)

    result = summarize_source_groups(labels, source_domains=domains, strategy="class_domain")

    assert sorted(result.group_counts.values()) == [1, 2]
    assert len(result.group_counts) == 2
    nan_domain_keys = [
        key
        for key in result.group_counts
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], float) and np.isnan(key[1])
    ]
    assert len(nan_domain_keys) == 1
    assert result.group_counts[nan_domain_keys[0]] == 2
