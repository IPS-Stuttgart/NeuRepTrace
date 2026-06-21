from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.riemannian import (
    align_covariances_to_identity,
    fit_predict_riemannian_transfer,
    log_euclidean_mean,
    riemannian_dispersion,
    riemannian_procrustes_transfer_features,
    riemannian_tangent_transfer_features,
    tangent_space_features,
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


def _symmetric_from_vector(vector):
    matrix = np.zeros((2, 2), dtype=float)
    rows, cols = np.triu_indices(2)
    values = np.asarray(vector, dtype=float).copy()
    values[rows != cols] /= np.sqrt(2.0)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def _matrix_exp_symmetric(matrix):
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.exp(values)[None, :]) @ vectors.T


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


def test_full_riemannian_procrustes_recenters_and_stretches_domains_without_labels():
    source_a, labels_a = _toy_covariances(mixing=np.eye(3), n_per_class=5)
    source_b, labels_b = _toy_covariances(
        mixing=np.array([[1.2, 0.25, 0.0], [0.0, 0.8, 0.15], [0.0, 0.0, 1.4]]),
        n_per_class=5,
    )
    target, _target_labels = _toy_covariances(mixing=np.diag([0.75, 1.4, 1.1]), n_per_class=5)
    source = np.concatenate([source_a, source_b], axis=0)
    source_domains = np.array(["s1"] * len(labels_a) + ["s2"] * len(labels_b))

    transfer = riemannian_procrustes_transfer_features(
        source,
        target,
        source_domains=source_domains,
        tangent_reference_scope="source_target",
    )

    assert transfer.protocol_category == 2
    assert transfer.uses_target_features is True
    assert transfer.uses_target_labels is False
    assert transfer.source_features.shape == (20, 6)
    assert transfer.target_features.shape == (10, 6)
    assert set(transfer.source_alignments) == {"s1", "s2"}

    identity = np.eye(3)
    target_dispersion = riemannian_dispersion(transfer.target_covariances, reference=identity)
    np.testing.assert_allclose(target_dispersion, transfer.target_dispersion, rtol=1e-8, atol=1e-8)
    for domain, alignment in transfer.source_alignments.items():
        aligned_domain = transfer.source_covariances[transfer.source_domains == domain]
        domain_dispersion = riemannian_dispersion(aligned_domain, reference=identity)
        np.testing.assert_allclose(domain_dispersion, target_dispersion, rtol=1e-5, atol=1e-5)
        assert alignment.rotation_mode == "none"
        assert alignment.n_rotation_pairs == 0
        assert alignment.stretch > 0.0


def test_full_riemannian_procrustes_paired_rotation_is_fit_in_tangent_space():
    rng = np.random.default_rng(23)
    source_vectors = rng.normal(scale=0.15, size=(8, 3))
    source_vectors -= np.mean(source_vectors, axis=0, keepdims=True)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    rotation = q * signs
    target_vectors = source_vectors @ rotation

    source_covariances = np.stack([_matrix_exp_symmetric(_symmetric_from_vector(vector)) for vector in source_vectors], axis=0)
    target_covariances = np.stack([_matrix_exp_symmetric(_symmetric_from_vector(vector)) for vector in target_vectors], axis=0)

    no_rotation = riemannian_procrustes_transfer_features(
        source_covariances,
        target_covariances,
        rotation_mode="none",
    )
    paired = riemannian_procrustes_transfer_features(
        source_covariances,
        target_covariances,
        source_anchor_covariances_by_domain={0: source_covariances},
        target_anchor_covariances=target_covariances,
        rotation_mode="paired",
    )

    no_rotation_source, _ = tangent_space_features(no_rotation.source_covariances, reference=np.eye(2))
    paired_source, _ = tangent_space_features(paired.source_covariances, reference=np.eye(2))
    paired_target, _ = tangent_space_features(paired.target_covariances, reference=np.eye(2))

    assert paired.source_alignments[0].rotation_mode == "paired"
    assert paired.source_alignments[0].n_rotation_pairs == source_covariances.shape[0]
    assert np.linalg.norm(paired_source - paired_target) < np.linalg.norm(no_rotation_source - paired_target)
    np.testing.assert_allclose(paired_source, paired_target, atol=1e-6)


def test_riemannian_procrustes_requires_paired_anchor_geometry_for_rotation():
    source, _labels = _toy_covariances(mixing=np.eye(3))
    target, _target_labels = _toy_covariances(mixing=np.eye(3))

    with pytest.raises(ValueError, match="requires target_anchor_covariances"):
        riemannian_procrustes_transfer_features(source, target, rotation_mode="paired")

    with pytest.raises(ValueError, match="requires source anchor covariances"):
        riemannian_procrustes_transfer_features(source, target, target_anchor_covariances=target, rotation_mode="paired")

    with pytest.raises(ValueError, match="paired anchor covariances"):
        riemannian_procrustes_transfer_features(source, target, target_anchor_covariances=target, rotation_mode="none")


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
