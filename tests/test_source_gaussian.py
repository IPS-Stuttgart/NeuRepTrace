from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_gaussian import (
    SOURCE_GAUSSIAN_CATEGORY,
    fit_source_gaussian_decoder,
    gaussian_log_likelihoods,
    normalize_covariance_type,
    normalize_prior_mode,
    source_gaussian_config,
)


def test_source_gaussian_predicts_separated_classes() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [2.0], [1.5]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_gaussian_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"covariance_type": "tied_spherical", "prior": "uniform"},
    )

    assert result.predictions.tolist() == ["left", "right"]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["source_gaussian_protocol_category"] == SOURCE_GAUSSIAN_CATEGORY
    assert result.metadata["source_gaussian_valid_for_strict_source_only"] is True
    assert result.metadata["source_gaussian_uses_test_features_for_fitting"] is False


def test_covariance_modes_change_variance_shape_not_dimensions() -> None:
    source_features = np.asarray([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0], [7.0, 6.0]], dtype=float)
    source_labels = np.asarray([0, 0, 1, 1], dtype=object)
    test_features = np.asarray([[0.5, 0.5]], dtype=float)

    diagonal = fit_source_gaussian_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"covariance_type": "diagonal"},
    )
    tied = fit_source_gaussian_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"covariance_type": "tied_diagonal"},
    )

    assert diagonal.variances.shape == tied.variances.shape == (2, 2)
    assert np.all(tied.variances[0] == tied.variances[1])
    assert not np.allclose(diagonal.variances[0], diagonal.variances[1])


def test_gaussian_log_likelihoods_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="feature width"):
        gaussian_log_likelihoods([[0.0, 1.0]], means=[[0.0]], variances=[[1.0]])


def test_aliases_and_config_validation() -> None:
    assert normalize_covariance_type("tied-diag") == "tied_diagonal"
    assert normalize_prior_mode("balanced") == "uniform"
    cfg = source_gaussian_config(temperature="2.0", variance_floor="1e-5")
    assert cfg.temperature == 2.0
    assert np.isclose(cfg.variance_floor, 1e-5)

    with pytest.raises(ValueError, match="covariance_type"):
        normalize_covariance_type("full")


def test_source_gaussian_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_gaussian_decoder(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            source_labels=[0, 1],
            test_features=[[0.0]],
        )
