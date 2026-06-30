from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_centroid import SOURCE_CENTROID_CATEGORY, fit_source_centroid_decoder, normalize_centroid_prototype, source_centroid_config


def test_source_centroid_predicts_nearest_classes() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [2.0], [1.5]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"use_diagonal_scale": False},
    )

    assert result.predictions.tolist() == ["left", "right"]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["source_centroid_protocol_category"] == SOURCE_CENTROID_CATEGORY
    assert result.metadata["source_centroid_valid_for_strict_source_only"] is True


def test_source_centroid_median_prototype_is_robust_to_outlier() -> None:
    source_features = np.asarray([[0.0], [1.0], [100.0], [10.0], [11.0], [12.0]], dtype=float)
    source_labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    test_features = np.asarray([[1.0], [11.0]], dtype=float)

    mean_result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"prototype": "mean", "use_diagonal_scale": False},
    )
    median_result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"prototype": "median", "use_diagonal_scale": False},
    )

    assert np.allclose(mean_result.centroids.ravel(), np.asarray([101.0 / 3.0, 11.0]))
    assert np.allclose(median_result.centroids.ravel(), np.asarray([1.0, 11.0]))
    assert median_result.predictions.tolist() == ["a", "b"]
    assert median_result.metadata["source_centroid_prototype"] == "median"


def test_source_centroid_shrinkage_moves_centroids_toward_global_mean() -> None:
    source_features = np.asarray([[0.0], [2.0], [8.0], [10.0]], dtype=float)
    source_labels = np.asarray([0, 0, 1, 1], dtype=object)
    test_features = np.asarray([[1.0], [9.0]], dtype=float)

    no_shrink = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"shrinkage": 0.0, "use_diagonal_scale": False},
    )
    shrink = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"shrinkage": 0.5, "use_diagonal_scale": False},
    )

    assert np.allclose(no_shrink.centroids.ravel(), np.asarray([1.0, 9.0]))
    assert np.allclose(shrink.centroids.ravel(), np.asarray([3.0, 7.0]))


def test_source_centroid_config_validation() -> None:
    cfg = source_centroid_config(temperature="2.5", shrinkage="0.25", prototype="robust")
    assert cfg.temperature == 2.5
    assert cfg.shrinkage == 0.25
    assert cfg.prototype == "median"
    assert normalize_centroid_prototype("average") == "mean"

    with pytest.raises(ValueError, match="shrinkage"):
        source_centroid_config(shrinkage=1.5)

    with pytest.raises(ValueError, match="centroid prototype"):
        normalize_centroid_prototype("bad")


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"temperature": True}, "temperature"),
        ({"temperature": np.bool_(True)}, "temperature"),
        ({"shrinkage": False}, "shrinkage"),
        ({"shrinkage": np.bool_(True)}, "shrinkage"),
        ({"epsilon": True}, "epsilon"),
        ({"epsilon": np.asarray(True)}, "epsilon"),
    ],
)
def test_source_centroid_config_rejects_boolean_numeric_values(kwargs, field) -> None:
    with pytest.raises(ValueError, match=field):
        source_centroid_config(**kwargs)


def test_source_centroid_string_false_disables_diagonal_scale() -> None:
    source_features = np.asarray([[0.0, 10.0], [1.0, 12.0], [5.0, 20.0], [6.0, 22.0]], dtype=float)
    source_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    test_features = np.asarray([[0.5, 11.0], [5.5, 21.0]], dtype=float)

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"use_diagonal_scale": "false"},
    )

    assert result.metadata["source_centroid_use_diagonal_scale"] is False
    assert np.allclose(result.feature_scale, np.ones(source_features.shape[1]))


def test_source_centroid_config_normalizes_common_boolean_values() -> None:
    assert source_centroid_config(use_diagonal_scale="false").use_diagonal_scale is False
    assert source_centroid_config(use_diagonal_scale="OFF").use_diagonal_scale is False
    assert source_centroid_config(use_diagonal_scale="YES").use_diagonal_scale is True
    assert source_centroid_config(use_diagonal_scale=np.bool_(False)).use_diagonal_scale is False
    assert source_centroid_config(use_diagonal_scale=1).use_diagonal_scale is True
    assert source_centroid_config(use_diagonal_scale=np.asarray(False)).use_diagonal_scale is False


@pytest.mark.parametrize("value", ["maybe", 2, -1, 0.5, np.asarray([False, True])])
def test_source_centroid_config_rejects_ambiguous_boolean_values(value) -> None:
    with pytest.raises(ValueError, match="use_diagonal_scale"):
        source_centroid_config(use_diagonal_scale=value)


def test_source_centroid_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_centroid_decoder(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            source_labels=[0, 1],
            test_features=[[0.0]],
        )
