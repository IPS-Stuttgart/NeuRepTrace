from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.random_subspace import (
    RANDOM_SUBSPACE_CATEGORY,
    fit_random_subspace_ensemble,
    random_subspace_ensemble_config,
    sample_feature_subspaces,
)


def _toy_data():
    train_features = np.asarray(
        [
            [-2.0, 0.0, 0.0, 0.2],
            [-1.6, 0.1, 0.0, 0.1],
            [-1.8, -0.2, 0.1, 0.0],
            [1.7, 0.0, 0.2, -0.1],
            [2.1, -0.1, 0.1, 0.0],
            [1.8, 0.2, -0.1, 0.1],
        ],
        dtype=float,
    )
    train_labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.7, 0.0, 0.0, 0.1], [1.9, 0.0, 0.1, 0.0]], dtype=float)
    return train_features, train_labels, test_features


def test_random_subspace_ensemble_outputs_probabilities_and_metadata() -> None:
    train_features, train_labels, test_features = _toy_data()

    result = fit_random_subspace_ensemble(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        config={"n_estimators": 6, "feature_fraction": 0.75, "random_state": 4},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.predictions.tolist() == ["left", "right"]
    assert len(result.members) == 6
    assert len(result.member_probabilities) == 6
    assert result.metadata["random_subspace_protocol_category"] == RANDOM_SUBSPACE_CATEGORY
    assert result.metadata["random_subspace_uses_test_features_for_fit"] is False
    assert result.metadata["random_subspace_uses_test_labels"] is False
    assert result.metadata["random_subspace_valid_for_strict_source_only"] is True


def test_sample_feature_subspaces_are_reproducible() -> None:
    first = sample_feature_subspaces(n_features=10, n_estimators=4, feature_fraction=0.3, min_features=2, random_state=13)
    second = sample_feature_subspaces(n_features=10, n_estimators=4, feature_fraction=0.3, min_features=2, random_state=13)

    assert len(first) == 4
    assert [subset.tolist() for subset in first] == [subset.tolist() for subset in second]
    assert all(subset.shape[0] == 3 for subset in first)
    assert all(np.all(np.diff(subset) >= 0) for subset in first)


def test_min_features_caps_to_available_features() -> None:
    subsets = sample_feature_subspaces(n_features=3, n_estimators=2, feature_fraction=0.1, min_features=10, random_state=0)

    assert all(subset.tolist() == [0, 1, 2] for subset in subsets)


def test_bootstrap_rows_records_smaller_member_row_counts() -> None:
    train_features, train_labels, test_features = _toy_data()

    result = fit_random_subspace_ensemble(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        config={"n_estimators": 3, "feature_fraction": 1.0, "bootstrap_rows": True, "row_fraction": 0.5, "random_state": 1},
    )

    assert len(result.members) == 3
    assert all(member.row_indices.shape[0] <= train_features.shape[0] for member in result.members)
    assert result.metadata["random_subspace_bootstrap_rows"] is True
    assert result.metadata["random_subspace_row_fraction"] == 0.5


def test_single_class_bootstrap_falls_back_to_all_rows() -> None:
    train_features = np.asarray([[-2.0], [-1.8], [2.0], [2.2]], dtype=float)
    train_labels = np.asarray([0, 0, 1, 1], dtype=object)
    test_features = np.asarray([[-1.9], [2.1]], dtype=float)

    result = fit_random_subspace_ensemble(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        config={"n_estimators": 2, "feature_fraction": 1.0, "bootstrap_rows": True, "row_fraction": 0.25, "random_state": 2},
    )

    assert all(np.unique(train_labels[member.row_indices]).shape[0] == 2 for member in result.members)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="feature_fraction"):
        random_subspace_ensemble_config(feature_fraction=0.0)
    with pytest.raises(ValueError, match="row_fraction"):
        random_subspace_ensemble_config(row_fraction=2.0)


def test_test_labels_are_not_part_of_public_api() -> None:
    train_features, train_labels, test_features = _toy_data()

    with pytest.raises(TypeError):
        fit_random_subspace_ensemble(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=["left", "right"],  # type: ignore[call-arg]
        )
