from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_knn import (
    SOURCE_KNN_CATEGORY,
    fit_source_knn_decoder,
    fit_source_knn_reference,
    normalize_weight_mode,
    predict_source_knn_probabilities,
    source_knn_config,
)


def test_source_knn_predicts_nearest_source_labels() -> None:
    source = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_knn_decoder(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"k": 1, "standardize": False},
    )

    assert result.predictions.tolist() == ["left", "right"]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.neighbor_indices.shape == (2, 1)
    assert result.metadata["source_knn_protocol_category"] == SOURCE_KNN_CATEGORY
    assert result.metadata["source_knn_uses_test_features_for_fitting"] is False
    assert result.metadata["source_knn_uses_test_labels"] is False
    assert result.metadata["source_knn_valid_for_strict_source_only"] is True


def test_distance_weighting_prefers_closer_neighbor() -> None:
    source = np.asarray([[0.0], [2.0], [10.0]], dtype=float)
    labels = np.asarray(["a", "b", "b"], dtype=object)
    test = np.asarray([[0.1]], dtype=float)

    result = fit_source_knn_decoder(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"k": 3, "weights": "distance", "standardize": False},
    )

    assert result.predictions.tolist() == ["a"]
    assert result.probabilities[0, 0] > result.probabilities[0, 1]


def test_knn_reference_can_be_reused() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)
    test = np.asarray([[0.2, 0.2], [5.5, 5.5]], dtype=float)
    reference = fit_source_knn_reference(source_features=source, source_labels=labels, config={"k": 1})

    probabilities, indices, distances = predict_source_knn_probabilities(test, reference)

    assert probabilities.shape == (2, 2)
    assert indices.shape == (2, 1)
    assert distances.shape == (2, 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_k_all_uses_all_source_rows() -> None:
    result = fit_source_knn_decoder(
        source_features=[[0.0], [1.0], [2.0], [3.0]],
        source_labels=[0, 0, 1, 1],
        test_features=[[1.5]],
        config={"k": "all", "weights": "uniform"},
    )

    assert result.neighbor_indices.shape == (1, 4)


def test_aliases_and_validation() -> None:
    assert normalize_weight_mode("equal") == "uniform"
    assert normalize_weight_mode("inverse-distance") == "distance"
    cfg = source_knn_config(k="full", standardize="false")
    assert cfg.k == "full"
    assert cfg.standardize is False

    with pytest.raises(ValueError, match="weight mode"):
        normalize_weight_mode("bad")

    with pytest.raises(ValueError, match="k"):
        fit_source_knn_decoder(source_features=[[0.0], [1.0]], source_labels=[0, 1], test_features=[[0.5]], config={"k": 0})


def test_source_knn_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_knn_decoder(source_features=[[0.0, 1.0]], source_labels=[0], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_knn_decoder(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
