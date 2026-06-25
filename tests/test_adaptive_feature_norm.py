from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.adaptive_feature_norm import (
    ADAPTIVE_FEATURE_NORM_CATEGORY,
    adaptive_feature_normalize,
    apply_adaptive_feature_norm,
    normalize_adaptive_feature_norm_method,
)


def _features():
    train = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
    test = np.asarray([[10.0, -2.0], [12.0, -1.0], [14.0, 0.0]], dtype=float)
    return train, test


def test_target_zscore_metadata_and_centering() -> None:
    train, test = _features()
    result = adaptive_feature_normalize(train, test, method="target-zscore")
    assert result.train_features.shape == train.shape
    assert result.test_features.shape == test.shape
    assert np.allclose(result.test_features.mean(axis=0), 0.0, atol=1e-6)
    assert result.metadata["adaptive_feature_norm_protocol_category"] == ADAPTIVE_FEATURE_NORM_CATEGORY
    assert result.metadata["adaptive_feature_norm_uses_target_features"] is True
    assert result.metadata["adaptive_feature_norm_uses_target_labels"] is False


def test_domain_zscore_and_moment_match() -> None:
    train, test = _features()
    domain = adaptive_feature_normalize(train, test, method="adabn")
    matched = adaptive_feature_normalize(train, test, method="source-to-target")
    assert domain.metadata["adaptive_feature_norm_method"] == "domain_zscore"
    assert np.allclose(domain.train_features.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(domain.test_features.mean(axis=0), 0.0, atol=1e-6)
    assert matched.metadata["adaptive_feature_norm_method"] == "moment_match"
    assert np.allclose(matched.train_features.mean(axis=0), test.mean(axis=0), atol=1e-6)
    assert np.allclose(matched.test_features, test)


def test_none_is_strict_source_only_and_transform_reuses_stats() -> None:
    train, test = _features()
    none = adaptive_feature_normalize(train, test, method="none")
    z = adaptive_feature_normalize(train, test, method="target_zscore")
    transformed = apply_adaptive_feature_norm(test[:1], mean=z.target_mean, scale=z.target_scale)
    assert np.allclose(none.train_features, train)
    assert np.allclose(none.test_features, test)
    assert none.metadata["adaptive_feature_norm_uses_target_features"] is False
    assert none.metadata["adaptive_feature_norm_valid_for_strict_source_only"] is True
    assert np.allclose(transformed, z.test_features[:1])


def test_aliases_and_guardrails() -> None:
    train, test = _features()
    assert normalize_adaptive_feature_norm_method("adaptive-batch-norm") == "domain_zscore"
    assert normalize_adaptive_feature_norm_method("target-standardize") == "target_zscore"
    with pytest.raises(ValueError, match="same feature width"):
        adaptive_feature_normalize(train, test[:, :1])
    with pytest.raises(ValueError, match="Unknown"):
        adaptive_feature_normalize(train, test, method="unknown")
    with pytest.raises(ValueError, match="positive"):
        adaptive_feature_normalize(train, test, scale_floor=0)
