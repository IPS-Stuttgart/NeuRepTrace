from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_domain_subset import apply_source_domain_subset, source_domain_subset_mask


@pytest.mark.parametrize("omit_fraction", [True, [0.5], (0.5,), np.asarray([0.5])])
def test_source_domain_subset_omit_fraction_rejects_non_scalar_values(omit_fraction: object) -> None:
    with pytest.raises(ValueError, match="omit_fraction"):
        source_domain_subset_mask(["a", "b"], omit_fraction=omit_fraction)


@pytest.mark.parametrize("min_domains", [True, [1], (1,), np.asarray([1])])
def test_source_domain_subset_min_domains_rejects_non_scalar_values(min_domains: object) -> None:
    with pytest.raises(ValueError, match="min_domains"):
        source_domain_subset_mask(["a", "b"], min_domains=min_domains)


def test_apply_source_domain_subset_rejects_non_finite_features() -> None:
    with pytest.raises(ValueError, match="features"):
        apply_source_domain_subset([[0.0], [float("nan")]], [0, 1], ["a", "b"])


def test_apply_source_domain_subset_rejects_zero_width_features() -> None:
    with pytest.raises(ValueError, match="features"):
        apply_source_domain_subset(np.empty((2, 0)), [0, 1], ["a", "b"])


@pytest.mark.parametrize(
    "features",
    [
        [[True], [False]],
        np.asarray([[True], [False]], dtype=bool),
        np.asarray([[0.0], [np.bool_(True)]], dtype=object),
        (iter(row) for row in [[0.0], [True]]),
    ],
)
def test_apply_source_domain_subset_rejects_boolean_features(features: object) -> None:
    with pytest.raises(ValueError, match="features.*booleans"):
        apply_source_domain_subset(features, [0, 1], ["a", "b"])


def test_apply_source_domain_subset_keeps_numeric_binary_features() -> None:
    selected_features, selected_labels, result = apply_source_domain_subset(
        [[0.0], [1.0]],
        ["zero", "one"],
        ["a", "b"],
        omit_fraction=0.0,
    )

    assert selected_features.dtype == np.float32
    assert selected_features.tolist() == [[0.0], [1.0]]
    assert selected_labels.tolist() == ["zero", "one"]
    assert result.omitted_domains == ()


def test_apply_source_domain_subset_materializes_generator_feature_rows() -> None:
    features = (iter(row) for row in [[0.0, 1.0], [2.0, 3.0]])

    selected_features, selected_labels, result = apply_source_domain_subset(
        features,
        ["first", "second"],
        ["a", "b"],
        omit_fraction=0.0,
    )

    assert selected_features.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert selected_labels.tolist() == ["first", "second"]
    assert result.selected_mask.tolist() == [True, True]
