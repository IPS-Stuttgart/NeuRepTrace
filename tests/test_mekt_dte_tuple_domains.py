from __future__ import annotations

import numpy as np

from neureptrace.decoding.mekt import mekt_transfer_features


def _spd_from_diagonal(diagonal, *, mixing):
    diagonal = np.asarray(diagonal, dtype=float)
    return mixing @ np.diag(diagonal) @ mixing.T


def _toy_covariances(*, mixing, n_per_class=3):
    covariances = []
    labels = []
    for label, base in enumerate(([1.8, 0.7, 0.5], [0.6, 1.7, 0.5])):
        for trial in range(n_per_class):
            jitter = 1.0 + 0.02 * trial
            covariances.append(_spd_from_diagonal(np.asarray(base) * jitter, mixing=mixing))
            labels.append(label)
    return np.stack(covariances, axis=0), np.asarray(labels)


def test_mekt_dte_selection_preserves_tuple_source_domains():
    source_a, labels_a = _toy_covariances(mixing=np.eye(3))
    source_b, labels_b = _toy_covariances(mixing=np.array([[0.8, 0.4, 0.0], [0.0, 1.4, 0.2], [0.0, 0.0, 0.7]]))
    target, _ = _toy_covariances(mixing=np.eye(3))
    near_domain = ("subject-a", "near")
    far_domain = ("subject-b", "far")

    transfer = mekt_transfer_features(
        np.concatenate([source_a, source_b], axis=0),
        np.concatenate([labels_a, labels_b], axis=0),
        target,
        source_domains=[near_domain] * len(labels_a) + [far_domain] * len(labels_b),
        dte_top_k=1,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )

    expected_domain = max(transfer.transferability_scores, key=transfer.transferability_scores.get)
    assert isinstance(expected_domain, tuple)
    assert transfer.selected_source_domains.shape == (1,)
    assert transfer.selected_source_domains[0] == expected_domain
    assert transfer.source_domains.shape == (len(labels_a),)
    assert all(domain == expected_domain for domain in transfer.source_domains.tolist())
