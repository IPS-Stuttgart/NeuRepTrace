from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.riemannian import (
    align_covariances_to_identity,
    fit_predict_riemannian_transfer,
    riemannian_tangent_transfer_features,
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


def test_subject_alignment_recenters_unlabeled_target_covariances():
    target_covariances, _ = _toy_covariances(mixing=np.diag([1.8, 0.6, 1.2]))

    aligned, reference = align_covariances_to_identity(target_covariances)
    aligned_again, _ = align_covariances_to_identity(target_covariances, reference=reference)

    assert aligned.shape == target_covariances.shape
    np.testing.assert_allclose(aligned, aligned_again)
    assert np.all(np.isfinite(aligned))


def test_riemannian_tangent_transfer_uses_target_features_without_labels():
    source_a, labels_a = _toy_covariances(mixing=np.eye(3))
    source_b, labels_b = _toy_covariances(mixing=np.array([[1.0, 0.2, 0.0], [0.0, 1.1, 0.1], [0.0, 0.0, 0.9]]))
    target, target_labels = _toy_covariances(mixing=np.diag([0.7, 1.5, 1.2]))
    source = np.concatenate([source_a, source_b], axis=0)
    labels = np.concatenate([labels_a, labels_b], axis=0)
    source_domains = np.array(["s1"] * len(labels_a) + ["s2"] * len(labels_b))

    classifier, transfer, predictions = fit_predict_riemannian_transfer(
        source,
        labels,
        target,
        source_domains=source_domains,
        tangent_reference_scope="source_target",
    )

    assert transfer.protocol_category == 2
    assert transfer.uses_target_features is True
    assert transfer.uses_target_labels is False
    assert transfer.source_features.shape == (16, 6)
    assert transfer.target_features.shape == (8, 6)
    assert sorted(np.unique(transfer.source_domains).tolist()) == ["s1", "s2"]
    assert hasattr(classifier, "predict")
    assert np.mean(predictions == target_labels) >= 0.75


def test_riemannian_transfer_rejects_target_label_like_arguments_by_api_shape():
    source, labels = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.eye(3))

    transfer = riemannian_tangent_transfer_features(source, target, tangent_reference_scope="source")

    assert transfer.source_features.shape[0] == labels.shape[0]
    with pytest.raises(ValueError, match="source_labels length"):
        fit_predict_riemannian_transfer(source, labels[:-1], target)


def test_riemannian_transfer_rejects_invalid_domains_and_scope():
    source, _ = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.eye(3))

    with pytest.raises(ValueError, match="source_domains length"):
        riemannian_tangent_transfer_features(source, target, source_domains=["s1"])

    with pytest.raises(ValueError, match="tangent_reference_scope"):
        riemannian_tangent_transfer_features(source, target, tangent_reference_scope="oracle")
