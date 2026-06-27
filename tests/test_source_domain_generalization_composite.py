from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_source_domain_generalization_missing_domain_detector_handles_pandas_and_composite_values() -> None:
    values = np.empty(4, dtype=object)
    values[0] = "sub-01"
    values[1] = pd.NA
    values[2] = ("subject", pd.NA)
    values[3] = ("subject", "02")

    mask = sdg._is_missing_domain_array(values)

    assert mask.tolist() == [False, True, True, False]


def test_source_domain_generalization_encoder_rejects_pandas_missing_composite_domains() -> None:
    features = np.ones((4, 2), dtype=float)
    labels = [0, 1, 0, 1]
    domains = [
        ("subject", "01"),
        ("subject", "01"),
        ("subject", pd.NA),
        ("subject", "02"),
    ]

    with pytest.raises(ValueError, match="source_domains must not contain missing values"):
        sdg._encode_inputs(features, labels, domains, name="source_domain_generalization")
