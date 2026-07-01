from __future__ import annotations

import numpy as np

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.decoding import source_domain_generalization as sdg


def _array_value_vector(*values: tuple[str, str]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = np.asarray(value, dtype=object)
    return vector


def test_source_domain_generalization_encoder_groups_array_valued_labels_and_domains() -> None:
    features = np.arange(18, dtype=float).reshape(6, 3)
    labels = _array_value_vector(
        ("cue", "left"),
        ("cue", "left"),
        ("cue", "right"),
        ("cue", "right"),
        ("cue", "left"),
        ("cue", "right"),
    )
    domains = _array_value_vector(
        ("sub", "01"),
        ("sub", "01"),
        ("sub", "01"),
        ("sub", "02"),
        ("sub", "02"),
        ("sub", "02"),
    )

    _x, classes, encoded_labels, domain_names, encoded_domains = sdg._encode_inputs(
        features,
        labels,
        domains,
        name="source_domain_generalization",
    )

    assert [tuple(value.tolist()) for value in classes] == [("cue", "left"), ("cue", "right")]
    assert encoded_labels.tolist() == [0, 0, 1, 1, 0, 1]
    assert [tuple(value.tolist()) for value in domain_names] == [("sub", "01"), ("sub", "02")]
    assert encoded_domains.tolist() == [0, 0, 0, 1, 1, 1]
