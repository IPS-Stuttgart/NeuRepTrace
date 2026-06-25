from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_mixstyle import augment_source_domains_mixstyle


def test_source_mixstyle_preserves_tuple_labels_atomically() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [10.0, 10.0],
            [11.0, 10.5],
        ],
        dtype=float,
    )
    labels = [("face", "left"), ("house", "right"), ("face", "left"), ("house", "right")]
    domains = np.asarray(["s1", "s1", "s2", "s2"], dtype=object)

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "random_state": 3},
    )

    assert result.labels.shape == (8,)
    assert result.labels[:4].tolist() == labels
    assert result.labels[4:].tolist() == labels
    assert all(isinstance(label, tuple) for label in result.labels.tolist())


def test_source_mixstyle_preserves_tuple_labels_in_synthetic_only_mode() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [10.0, 10.0],
            [11.0, 10.5],
        ],
        dtype=float,
    )
    labels = [("face", "left"), ("house", "right"), ("face", "left"), ("house", "right")]
    domains = np.asarray(["s1", "s1", "s2", "s2"], dtype=object)

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "include_original": False, "random_state": 3},
    )

    assert result.labels.shape == (4,)
    assert result.labels.tolist() == labels
    assert all(isinstance(label, tuple) for label in result.labels.tolist())


def test_source_mixstyle_preserves_tuple_domains_atomically() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [10.0, 10.0],
            [11.0, 10.5],
        ],
        dtype=float,
    )
    labels = np.asarray(["face", "house", "face", "house"], dtype=object)
    domains = [("s1", "run1"), ("s1", "run1"), ("s2", "run1"), ("s2", "run1")]

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "random_state": 3},
    )

    assert result.domain_ids.shape == (8,)
    assert result.domain_ids[:4].tolist() == domains
    assert result.domain_ids[4:].tolist() == domains
    assert result.metadata["source_mixstyle_n_source_domains"] == 2
    assert all(isinstance(domain, tuple) for domain in result.domain_ids.tolist())


def test_source_mixstyle_preserves_tuple_domains_without_synthetic_rows() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=float)
    labels = np.asarray(["face", "house"], dtype=object)
    domains = [("s1", "run1"), ("s2", "run1")]

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 0},
    )

    assert result.domain_ids.shape == (2,)
    assert result.domain_ids.tolist() == domains
    assert result.n_synthetic == 0
