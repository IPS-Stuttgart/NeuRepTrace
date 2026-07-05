from __future__ import annotations

import numpy as np

from neureptrace.decoding.mekt import fit_predict_mekt_transfer, mekt_transfer_features


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


def _composite_labels(labels: np.ndarray) -> list[tuple[str, int]]:
    return [("stimulus", int(label)) for label in labels.tolist()]


def test_mekt_preserves_tuple_source_labels_through_estimator_fits():
    source, scalar_labels = _toy_covariances(mixing=np.eye(3), n_per_class=5)
    target, target_scalar_labels = _toy_covariances(mixing=np.diag([0.7, 1.5, 1.2]), n_per_class=5)
    labels = _composite_labels(scalar_labels)
    initial_pseudo_labels = _composite_labels(target_scalar_labels)

    transfer = mekt_transfer_features(
        source,
        labels,
        target,
        initial_pseudo_labels=initial_pseudo_labels,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )
    classifier, fit_transfer, predictions = fit_predict_mekt_transfer(
        source,
        labels,
        target,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )

    expected_classes = {("stimulus", 0), ("stimulus", 1)}
    assert set(transfer.classes.tolist()) == expected_classes
    assert set(fit_transfer.classes.tolist()) == expected_classes
    assert set(predictions.tolist()).issubset(expected_classes)
    assert all(isinstance(label, tuple) for label in transfer.pseudo_labels.tolist())
    assert all(isinstance(label, tuple) for label in predictions.tolist())
    assert hasattr(classifier, "predict")
