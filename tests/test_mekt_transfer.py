from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mekt import (
    centroid_aligned_tangent_features,
    domain_transferability_scores,
    fit_predict_mekt_transfer,
    mekt_transfer_features,
)


def _spd_from_diagonal(diagonal, *, mixing):
    diagonal = np.asarray(diagonal, dtype=float)
    return mixing @ np.diag(diagonal) @ mixing.T


def _toy_covariances(*, mixing, n_per_class=4):
    covariances = []
    labels = []
    for label, base in enumerate(([1.8, 0.7, 0.5], [0.6, 1.7, 0.5])):
        for trial in range(n_per_class):
            jitter = 1.0 + 0.02 * trial
            covariances.append(_spd_from_diagonal(np.asarray(base) * jitter, mixing=mixing))
            labels.append(label)
    return np.stack(covariances, axis=0), np.asarray(labels)


def test_centroid_aligned_tangent_features_are_category_two():
    source, _ = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.diag([0.8, 1.3, 1.1]))

    transfer = centroid_aligned_tangent_features(source, target)

    assert transfer.protocol_category == 2
    assert transfer.uses_target_features is True
    assert transfer.uses_target_labels is False
    assert transfer.source_features.shape == (8, 6)
    assert transfer.target_features.shape == (8, 6)
    assert np.all(np.isfinite(transfer.source_features))
    assert np.all(np.isfinite(transfer.target_features))


def test_full_mekt_iterates_joint_alignment_without_target_labels():
    source_a, labels_a = _toy_covariances(mixing=np.eye(3), n_per_class=5)
    source_b, labels_b = _toy_covariances(mixing=np.array([[1.0, 0.2, 0.0], [0.0, 1.1, 0.1], [0.0, 0.0, 0.9]]), n_per_class=5)
    target, _ = _toy_covariances(mixing=np.diag([0.7, 1.5, 1.2]), n_per_class=5)
    source = np.concatenate([source_a, source_b], axis=0)
    labels = np.concatenate([labels_a, labels_b], axis=0)
    source_domains = np.array(["s1"] * len(labels_a) + ["s2"] * len(labels_b))

    classifier, transfer, predictions = fit_predict_mekt_transfer(
        source,
        labels,
        target,
        source_domains=source_domains,
        n_components=2,
        n_iterations=3,
        n_neighbors=2,
    )

    assert transfer.protocol_category == 2
    assert transfer.uses_target_features is True
    assert transfer.uses_target_labels is False
    assert transfer.source_embedding.shape == (20, 2)
    assert transfer.target_embedding.shape == (10, 2)
    assert transfer.source_projection.shape == (6, 2)
    assert transfer.target_projection.shape == (6, 2)
    assert 1 <= transfer.n_iterations <= 3
    assert len(transfer.pseudo_label_history) == transfer.n_iterations + 1
    assert predictions.shape == (10,)
    assert set(predictions).issubset(set(labels.tolist()))
    assert hasattr(classifier, "predict")


def test_mekt_accepts_single_column_source_domain_vectors():
    source, _ = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.diag([0.8, 1.3, 1.1]))
    domains = np.asarray(["s1"] * 4 + ["s2"] * 4, dtype=object).reshape(-1, 1)

    transfer = centroid_aligned_tangent_features(source, target, source_domains=domains)

    assert transfer.source_domains.shape == (8,)
    assert transfer.source_domains.tolist() == ["s1"] * 4 + ["s2"] * 4
    assert transfer.source_features.shape == (8, 6)


def test_mekt_rejects_matrix_shaped_labels_but_accepts_rowwise_composite_source_domains():
    source, labels = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.eye(3))
    label_matrix = np.asarray([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=int)
    domain_matrix = np.column_stack(
        [
            np.asarray(["s1"] * 4 + ["s2"] * 4),
            np.asarray(["run-a"] * source.shape[0]),
        ]
    )

    with pytest.raises(ValueError, match="source_labels must be a one-dimensional vector"):
        mekt_transfer_features(source, label_matrix, target, n_iterations=1, n_neighbors=2)

    transfer = centroid_aligned_tangent_features(source, target, source_domains=domain_matrix)

    assert labels.shape[0] == source.shape[0]
    assert transfer.source_domains.shape == (source.shape[0],)
    assert transfer.source_domains[0] == ("s1", "run-a")
    assert transfer.source_domains[-1] == ("s2", "run-a")


def test_mekt_domain_transferability_scores_and_selection():
    source_features = np.array(
        [
            [-1.2, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [8.8, 0.0],
            [9.0, 0.0],
            [11.0, 0.0],
            [11.2, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    domains = np.array(["near", "near", "near", "near", "far", "far", "far", "far"])
    target_features = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]], dtype=float)

    scores = domain_transferability_scores(source_features, labels, target_features, domains)

    assert scores["near"] > scores["far"]

    source_a, labels_a = _toy_covariances(mixing=np.eye(3), n_per_class=3)
    source_b, labels_b = _toy_covariances(mixing=np.array([[0.8, 0.4, 0.0], [0.0, 1.4, 0.2], [0.0, 0.0, 0.7]]), n_per_class=3)
    target, _ = _toy_covariances(mixing=np.eye(3), n_per_class=3)
    transfer = mekt_transfer_features(
        np.concatenate([source_a, source_b], axis=0),
        np.concatenate([labels_a, labels_b], axis=0),
        target,
        source_domains=np.array(["near"] * len(labels_a) + ["far"] * len(labels_b)),
        dte_top_k=1,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )

    expected_domain = max(transfer.transferability_scores, key=transfer.transferability_scores.get)
    assert transfer.selected_source_domains.tolist() == [expected_domain]
    assert np.unique(transfer.source_domains).tolist() == [expected_domain]
    assert set(transfer.transferability_scores) == {"near", "far"}


def test_mekt_rejects_target_label_like_or_invalid_arguments():
    source, labels = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.eye(3))

    with pytest.raises(ValueError, match="initial_pseudo_labels length"):
        mekt_transfer_features(source, labels, target, initial_pseudo_labels=[0, 1])

    with pytest.raises(ValueError, match="n_iterations"):
        mekt_transfer_features(source, labels, target, n_iterations=0)

    with pytest.raises(ValueError, match="source_labels length"):
        mekt_transfer_features(source, labels[:-1], target)
