from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_gaussian import (
    SOURCE_GAUSSIAN_CATEGORY,
    SourceGaussianConfig,
    fit_source_gaussian_decoder,
    gaussian_log_likelihoods,
    normalize_covariance_type,
    normalize_prior_mode,
    source_gaussian_config,
)
from neureptrace.decoding.source_mahalanobis import (
    SOURCE_MAHALANOBIS_CATEGORY,
    SourceMahalanobisConfig,
    fit_source_mahalanobis_decoder,
    source_mahalanobis_config,
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


def test_source_gaussian_revalidates_direct_dataclass_config() -> None:
    source = np.asarray([[0.0], [0.2], [3.0], [3.2]], dtype=float)
    test = np.asarray([[0.1], [3.1]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)

    result = fit_source_gaussian_decoder(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config=SourceGaussianConfig(covariance_type="diag", prior="flat", variance_floor="1e-5", temperature="2.0"),  # type: ignore[arg-type]
    )

    assert result.metadata["source_gaussian_covariance_type"] == "diagonal"
    assert result.metadata["source_gaussian_prior"] == "uniform"
    assert result.metadata["source_gaussian_variance_floor"] == pytest.approx(1e-5)
    assert result.metadata["source_gaussian_temperature"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="temperature"):
        fit_source_gaussian_decoder(
            source_features=source,
            source_labels=labels,
            test_features=test,
            config=SourceGaussianConfig(temperature=float("nan")),
        )


@pytest.mark.parametrize("value", [True, np.bool_(True), [], {"variance_floor": 1}, np.asarray(1.0), np.asarray([1.0])])
def test_source_gaussian_rejects_invalid_variance_floor_values(value: object) -> None:
    with pytest.raises(ValueError, match="variance_floor"):
        source_gaussian_config(variance_floor=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, np.bool_(True), [], {"temperature": 1}, np.asarray(1.0), np.asarray([1.0])])
def test_source_gaussian_rejects_invalid_temperature_values(value: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        source_gaussian_config(temperature=value)  # type: ignore[arg-type]


def test_source_gaussian_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_gaussian_decoder(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            source_labels=[0, 1],
            test_features=[[0.0]],
        )


def test_source_mahalanobis_fits_source_only_decoder() -> None:
    source = np.asarray([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]], dtype=float)
    test = np.asarray([[0.1, 0.0], [3.1, 3.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)

    result = fit_source_mahalanobis_decoder(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"prior": "uniform", "regularization": "0.01", "temperature": "1.5"},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.predictions.tolist() == ["left", "right"]
    assert result.metadata["source_mahalanobis_protocol_category"] == SOURCE_MAHALANOBIS_CATEGORY
    assert result.metadata["source_mahalanobis_uses_source_features"] is True
    assert result.metadata["source_mahalanobis_uses_test_features_for_fitting"] is False
    assert result.metadata["source_mahalanobis_uses_test_labels"] is False
    assert result.metadata["source_mahalanobis_valid_for_strict_source_only"] is True


def test_source_mahalanobis_preserves_composite_source_labels() -> None:
    source = np.asarray([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]], dtype=float)
    test = np.asarray([[0.1, 0.0], [3.1, 3.0]], dtype=float)
    labels = [("left", 1), ("left", 1), ("right", 2), ("right", 2)]

    result = fit_source_mahalanobis_decoder(source_features=source, source_labels=labels, test_features=test)

    assert result.classes.tolist() == [("left", 1), ("right", 2)]
    assert result.predictions.tolist() == [("left", 1), ("right", 2)]
    assert "('left', 1):2" in result.metadata["source_mahalanobis_class_counts"]


def test_source_mahalanobis_revalidates_direct_dataclass_config() -> None:
    source = np.asarray([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]], dtype=float)
    test = np.asarray([[0.1, 0.0], [3.1, 3.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)

    result = fit_source_mahalanobis_decoder(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config=SourceMahalanobisConfig(regularization="0.01", prior="flat", temperature="2.0"),  # type: ignore[arg-type]
    )

    assert result.metadata["source_mahalanobis_regularization"] == pytest.approx(0.01)
    assert result.metadata["source_mahalanobis_prior"] == "uniform"
    assert result.metadata["source_mahalanobis_temperature"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="temperature"):
        fit_source_mahalanobis_decoder(
            source_features=source,
            source_labels=labels,
            test_features=test,
            config=SourceMahalanobisConfig(temperature=float("nan")),
        )


@pytest.mark.parametrize("value", [True, np.bool_(True), [], {"regularization": 1}, np.asarray(0.1), np.asarray([0.1])])
def test_source_mahalanobis_rejects_invalid_regularization_values(value: object) -> None:
    with pytest.raises(ValueError, match="regularization"):
        source_mahalanobis_config(regularization=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, np.bool_(True), [], {"temperature": 1}, np.asarray(1.0), np.asarray([1.0])])
def test_source_mahalanobis_rejects_invalid_temperature_values(value: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        source_mahalanobis_config(temperature=value)  # type: ignore[arg-type]
