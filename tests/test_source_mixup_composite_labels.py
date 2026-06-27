from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_mixup import augment_source_with_mixup


def test_source_mixup_preserves_tuple_labels_and_tuple_domains() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = [
        ("cue", "left"),
        ("cue", "left"),
        ("cue", "right"),
        ("cue", "right"),
    ]
    domains = [
        ("subject1", "run1"),
        ("subject2", "run1"),
        ("subject1", "run1"),
        ("subject2", "run1"),
    ]

    result = augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "same_class_partner": True, "cross_domain_partner": True, "random_state": 3},
    )

    assert result.labels.shape == (6,)
    assert result.classes.tolist() == [("cue", "left"), ("cue", "right")]
    assert result.metadata["source_mixup_n_classes"] == 2
    assert result.metadata["source_mixup_n_source_domains"] == 2
    assert np.allclose(result.label_distributions.sum(axis=1), 1.0)
    assert all(isinstance(label, tuple) for label in result.labels.tolist())
    for content_index, partner_index in zip(result.content_indices, result.partner_indices, strict=True):
        assert domains[int(content_index)] != domains[int(partner_index)]


def test_source_mixup_treats_domain_matrix_rows_as_atomic_ids() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    domains = np.asarray(
        [
            ["subject1", "run1"],
            ["subject2", "run1"],
            ["subject1", "run1"],
            ["subject2", "run1"],
        ],
        dtype=object,
    )

    result = augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "same_class_partner": True, "cross_domain_partner": True, "random_state": 5},
    )

    assert result.metadata["source_mixup_n_source_domains"] == 2
    for content_index, partner_index in zip(result.content_indices, result.partner_indices, strict=True):
        assert tuple(domains[int(content_index)]) != tuple(domains[int(partner_index)])


def test_source_mixup_accepts_yaml_style_list_domain_ids() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    domains = [
        ["subject1", "run1"],
        ["subject2", "run1"],
        ["subject1", "run1"],
        ["subject2", "run1"],
    ]

    result = augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "same_class_partner": True, "cross_domain_partner": True, "random_state": 11},
    )

    assert result.metadata["source_mixup_n_source_domains"] == 2
    for content_index, partner_index in zip(result.content_indices, result.partner_indices, strict=True):
        assert tuple(domains[int(content_index)]) != tuple(domains[int(partner_index)])
