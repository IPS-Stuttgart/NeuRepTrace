from __future__ import annotations

import numpy as np

from neureptrace.decoding.adaptive_feature_norm import adaptive_feature_normalize, normalize_adaptive_feature_norm_method


def test_robust_feature_norm_uses_target_median() -> None:
    train = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
    test = np.asarray([[10.0, -1.0], [12.0, 0.0], [14.0, 1.0], [200.0, 2.0]], dtype=float)

    result = adaptive_feature_normalize(train, test, method="robust")

    assert result.metadata["adaptive_feature_norm_method"] == "robust_zscore"
    assert result.metadata["adaptive_feature_norm_uses_target_features"] is True
    assert result.metadata["adaptive_feature_norm_uses_target_labels"] is False
    assert np.allclose(result.target_mean, np.median(test, axis=0))
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))


def test_robust_feature_norm_aliases() -> None:
    assert normalize_adaptive_feature_norm_method("target-iqr") == "robust_zscore"
    assert normalize_adaptive_feature_norm_method("iqr-zscore") == "robust_zscore"
