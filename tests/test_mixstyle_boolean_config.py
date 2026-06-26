from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mixstyle import augment_source_mixstyle, source_mixstyle_config as feature_mixstyle_config
from neureptrace.decoding.source_mixstyle import augment_source_domains_mixstyle, source_mixstyle_config as domain_mixstyle_config


def _toy_sources():
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, -0.1],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.8, 5.2],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "b", "a", "a", "b", "a"], dtype=object)
    domains = np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"], dtype=object)
    return features, labels, domains


def test_feature_mixstyle_string_false_flags_are_false() -> None:
    features, labels, domains = _toy_sources()

    config = feature_mixstyle_config(
        include_original="false",
        preserve_domain_mean="false",
        class_conditional="false",
    )

    assert config.include_original is False
    assert config.preserve_domain_mean is False
    assert config.class_conditional is False

    result = augment_source_mixstyle(
        features,
        labels,
        domains,
        augmentations_per_row=1,
        include_original="false",
        preserve_domain_mean="false",
        class_conditional="false",
        random_state=1,
    )

    assert result.n_original == 0
    assert result.n_synthetic == features.shape[0]
    assert np.all(result.synthetic_mask)
    assert result.metadata["source_mixstyle_include_original"] is False
    assert result.metadata["source_mixstyle_preserve_domain_mean"] is False
    assert result.metadata["source_mixstyle_class_conditional"] is False


def test_domain_mixstyle_string_false_include_original_is_false() -> None:
    features, labels, domains = _toy_sources()

    config = domain_mixstyle_config(include_original="false")
    assert config.include_original is False

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "include_original": "false", "random_state": 1},
    )

    assert result.n_original == 0
    assert result.n_synthetic == features.shape[0]
    assert np.all(result.synthetic_mask)
    assert result.metadata["source_mixstyle_include_original"] is False


def test_mixstyle_boolean_options_reject_ambiguous_strings() -> None:
    features, labels, domains = _toy_sources()

    with pytest.raises(ValueError, match="include_original"):
        feature_mixstyle_config(include_original="maybe")
    with pytest.raises(ValueError, match="preserve_domain_mean"):
        augment_source_mixstyle(features, labels, domains, preserve_domain_mean="maybe")
    with pytest.raises(ValueError, match="include_original"):
        domain_mixstyle_config(include_original="maybe")
