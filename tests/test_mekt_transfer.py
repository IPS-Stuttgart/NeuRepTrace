from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.riemannian import (
    estimate_domain_transferability,
    fit_predict_mekt,
    mekt_tangent_transfer_features,
    mekt_transfer_features,
)


def _spd_from_diagonal(diagonal, *, mixing):
    diagonal = np.asarray(diagonal, dtype=float)
    return mixing @ np.diag(diagonal) @ mixing.T


def _toy_covariances(*, mixing, n_per_class=5):
    covariances = []
    labels = []
    for label, base in enumerate(([2.0, 0.6, 0.4], [0.6, 2.0, 0.4])):
        for trial in range(n_per_class):
            jitter = 1.0 + 0.03 * trial
            covariances.append(_spd_from_diagonal(np.asarray(base) * jitter, mixing=mixing))
            labels.append(label)
    return np.stack(covariances, axis=0), np.asarray(labels)


def test_mekt_tangent_features_match_category2_centroid_alignment():
    source_a, labels_a = _toy_covariances(mixing=np.eye(3))
    source_b, labels_b = _toy_covariances(mixing=np.diag([1.4, 0.8, 1.1]))
    target, _ = _toy_covariances(mixing=np.diag([0.7, 1.6, 1.0]))
    source = np.concatenate([source_a, source_b], axis=0)
    source_domains = np.array(["s1"] * len(labels_a) + ["s2"] * len(labels_b), dtype=object)

    transfer = mekt_tangent_transfer_features(source, target, source_domains=source_domains)

    assert transfer.protocol_category == 2
    assert transfer.uses_target_features is True
    assert transfer.uses_target_labels is False
    assert transfer.source_features.shape == (20, 6)
    assert transfer.target_features.shape == (10, 6)
    np.testing.assert_allclose(transfer.tangent_reference, np.eye(3))


def test_full_mekt_solves_projection_and_refines_pseudo_labels_without_target_labels():
    source_a, labels_a = _toy_covariances(mixing=np.eye(3), n_per_class=8)
    source_b, labels_b = _toy_covariances(mixing=np.array([[1.0, 0.1, 0.0], [0.0, 1.2, 0.1], [0.0, 0.0, 0.9]]), n_per_class=8)
    target, target_labels = _toy_covariances(mixing=np.eye(3), n_per_class=8)
    source = np.concatenate([source_a, source_b], axis=0)
    labels = np.concatenate([labels_a, labels_b], axis=0)
    source_domains = np.array(["s1"] * len(labels_a) + ["s2"] * len(labels_b), dtype=object)

    classifier, transfer, predictions = fit_predict_mekt(
        source,
        labels,
        target,
        source_domains=source_domains,
        n_components=1,
        n_iterations=3,
        n_neighbors=3,
        source_domain_selection=2,
    )

    assert transfer.protocol_category == 2
    assert transfer.uses_target_features is True
    assert transfer.uses_target_labels is False
    assert transfer.source_features.shape == (32, 1)
    assert transfer.target_features.shape == (16, 1)
    assert transfer.source_projection.shape == (6, 1)
    assert transfer.target_projection.shape == (6, 1)
    assert len(transfer.pseudo_label_history) == 3
    assert transfer.generalized_eigenvalues.shape == (1,)
    assert set(transfer.domain_transferability) == {"s1", "s2"}
    assert sorted(transfer.selected_source_domains.tolist()) == ["s1", "s2"]
    assert hasattr(classifier, "predict")
    assert np.mean(predictions == target_labels) >= 0.75


def test_mekt_dte_can_select_the_most_transferable_source_domain():
    rng = np.random.default_rng(13)
    target = np.vstack(
        [
            rng.normal(loc=[1.0, 0.0, 0.0], scale=0.05, size=(8, 3)),
            rng.normal(loc=[0.0, 1.0, 0.0], scale=0.05, size=(8, 3)),
        ]
    )
    good = np.vstack(
        [
            rng.normal(loc=[1.1, 0.0, 0.0], scale=0.05, size=(8, 3)),
            rng.normal(loc=[0.0, 1.1, 0.0], scale=0.05, size=(8, 3)),
        ]
    )
    poor = np.vstack(
        [
            rng.normal(loc=[5.0, 5.0, 0.0], scale=0.05, size=(8, 3)),
            rng.normal(loc=[5.2, 5.2, 0.0], scale=0.05, size=(8, 3)),
        ]
    )
    source = np.vstack([good, poor])
    labels = np.tile(np.repeat([0, 1], 8), 2)
    domains = np.array(["good"] * 16 + ["poor"] * 16, dtype=object)

    scores = estimate_domain_transferability(source, labels, target, source_domains=domains)

    assert scores["good"] > scores["poor"]


def test_mekt_rejects_label_length_and_invalid_hyperparameters():
    source, labels = _toy_covariances(mixing=np.eye(3))
    target, _ = _toy_covariances(mixing=np.eye(3))

    with pytest.raises(ValueError, match="source_labels length"):
        mekt_transfer_features(source, labels[:-1], target)

    with pytest.raises(ValueError, match="n_iterations"):
        mekt_transfer_features(source, labels, target, n_iterations=0)

    with pytest.raises(ValueError, match="source_domain_selection"):
        mekt_transfer_features(source, labels, target, source_domain_selection=0)
