from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_mixstyle import augment_source_domains_mixstyle


def test_source_mixstyle_preserves_tuple_labels_and_tuple_domains_atomically() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [10.0, 10.0],
            [11.0, 10.5],
        ],
        dtype=float,
    )
    labels = [("face", "seen"), ("house", "novel"), ("face", "seen"), ("house", "novel")]
    domains = [("s1", "run1"), ("s1", "run1"), ("s2", "run1"), ("s2", "run1")]

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "random_state": 3},
    )

    assert result.features.shape == (8, 2)
    assert result.labels.shape == (8,)
    assert result.domain_ids.shape == (8,)
    assert result.n_original == 4
    assert result.n_synthetic == 4
    assert result.labels[:4].tolist() == labels
    assert result.labels[4:].tolist() == labels
    assert result.domain_ids[:4].tolist() == domains
    assert result.domain_ids[4:].tolist() == domains
    assert np.all(np.isfinite(result.features))


def test_source_mixstyle_zero_mixes_preserves_tuple_labels_and_domains() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=float)
    labels = [("left", 1), ("right", 2)]
    domains = [("source", "a"), ("source", "b")]

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 0},
    )

    assert result.labels.shape == (2,)
    assert result.domain_ids.shape == (2,)
    assert result.labels.tolist() == labels
    assert result.domain_ids.tolist() == domains
    assert result.n_synthetic == 0
