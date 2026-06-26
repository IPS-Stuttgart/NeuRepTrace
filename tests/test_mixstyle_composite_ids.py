from __future__ import annotations

import numpy as np

from neureptrace.decoding.mixstyle import augment_source_mixstyle


def test_mixstyle_preserves_composite_labels_and_domains() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [5.0, 5.0],
            [6.0, 5.2],
        ],
        dtype=float,
    )
    labels = [("left", 1), ("right", 1), ("left", 1), ("right", 1)]
    domains = [("s1", "run1"), ("s1", "run1"), ("s2", "run1"), ("s2", "run1")]

    result = augment_source_mixstyle(
        features,
        labels,
        domains,
        augmentations_per_row=1,
        class_conditional=True,
        random_state=0,
    )

    assert result.labels.shape == (8,)
    assert result.domains.shape == (8,)
    assert result.labels[:4].tolist() == labels
    assert result.labels[4:].tolist() == labels
    assert result.domains[:4].tolist() == domains
    assert result.metadata["source_mixstyle_n_classes"] == 2


def test_mixstyle_single_composite_label_and_domain_are_atomic_original_only() -> None:
    result = augment_source_mixstyle(
        [[1.0, 2.0]],
        ("class", 1),
        ("subject", "run1"),
        augmentations_per_row=0,
    )

    assert result.labels.shape == (1,)
    assert result.domains.shape == (1,)
    assert result.labels.tolist() == [("class", 1)]
    assert result.domains.tolist() == [("subject", "run1")]
