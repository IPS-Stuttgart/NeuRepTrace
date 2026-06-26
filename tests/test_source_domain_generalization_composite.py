from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.decoding import source_domain_generalization as sdg


def test_source_domain_generalization_encoder_preserves_tuple_labels_and_domains() -> None:
    features = np.arange(18, dtype=float).reshape(6, 3)
    labels = [
        ("cue", "left"),
        ("cue", "left"),
        ("cue", "right"),
        ("cue", "right"),
        ("cue", "left"),
        ("cue", "right"),
    ]
    domains = [
        ("subject", "01"),
        ("subject", "01"),
        ("subject", "01"),
        ("subject", "02"),
        ("subject", "02"),
        ("subject", "02"),
    ]
    assert np.asarray(labels, dtype=object).ndim == 2
    assert np.asarray(domains, dtype=object).ndim == 2

    x, classes, encoded_labels, domain_names, encoded_domains = sdg._encode_inputs(
        features,
        labels,
        domains,
        name="source_domain_generalization",
    )

    assert x.shape == features.shape
    assert classes.tolist() == [("cue", "left"), ("cue", "right")]
    assert encoded_labels.tolist() == [0, 0, 1, 1, 0, 1]
    assert domain_names.tolist() == [("subject", "01"), ("subject", "02")]
    assert encoded_domains.tolist() == [0, 0, 0, 1, 1, 1]


def test_source_domain_generalization_encoder_counts_composite_classes_not_fields() -> None:
    features = np.ones((4, 2), dtype=float)
    labels = [("phase", "early"), ("phase", "early"), ("phase", "late"), ("phase", "late")]
    domains = ["sub-01", "sub-01", "sub-02", "sub-02"]

    _x, classes, encoded_labels, _domain_names, _encoded_domains = sdg._encode_inputs(
        features,
        labels,
        domains,
        name="source_domain_generalization",
    )

    assert classes.shape == (2,)
    assert encoded_labels.shape == (4,)
    assert set(encoded_labels.tolist()) == {0, 1}
