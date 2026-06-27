from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_bagging import SOURCE_BAGGING_CATEGORY, fit_source_bagging_decoder, source_bagging_config


def test_source_bagging_predicts_separated_classes() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 5, "random_state": 7},
    )

    assert result.predictions.tolist() == ["left", "right"]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.n_estimators == 5
    assert result.metadata["source_bagging_protocol_category"] == SOURCE_BAGGING_CATEGORY
    assert result.metadata["source_bagging_valid_for_strict_source_only"] is True


def test_source_bagging_feature_fraction_subsamples_features() -> None:
    source_features = np.asarray([[-2.0, 0.0, 1.0, 0.0], [-1.5, 0.2, 1.1, 0.0], [1.5, 0.1, -1.0, 0.0], [2.0, -0.1, -1.1, 0.0]], dtype=float)
    source_labels = np.asarray([0, 0, 1, 1], dtype=object)
    test_features = np.asarray([[-1.8, 0.0, 1.0, 0.0], [1.8, 0.0, -1.0, 0.0]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 3, "feature_fraction": 0.5, "random_state": 11},
    )

    assert len(result.feature_indices) == 3
    assert all(indices.size == 2 for indices in result.feature_indices)


def test_source_bagging_config_validation() -> None:
    cfg = source_bagging_config(n_estimators="3", sample_fraction="0.75", feature_fraction="0.5")
    assert cfg.n_estimators == 3
    assert cfg.sample_fraction == 0.75
    assert cfg.feature_fraction == 0.5

    with pytest.raises(ValueError, match="n_estimators"):
        source_bagging_config(n_estimators=0)
