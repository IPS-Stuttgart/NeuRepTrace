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


def _heterogeneous_labels(labels: np.ndarray) -> list[object]:
    return [1 if int(label) == 0 else "1" for label in labels.tolist()]


def test_mekt_preserves_heterogeneous_scalar_label_types() -> None:
    source, source_scalar_labels = _toy_covariances(mixing=np.eye(3), n_per_class=5)
    target, target_scalar_labels = _toy_covariances(mixing=np.diag([0.7, 1.5, 1.2]), n_per_class=5)
    source_labels = _heterogeneous_labels(source_scalar_labels)
    initial_pseudo_labels = _heterogeneous_labels(target_scalar_labels)

    transfer = mekt_transfer_features(
        source,
        source_labels,
        target,
        initial_pseudo_labels=initial_pseudo_labels,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )
    _, fit_transfer, predictions = fit_predict_mekt_transfer(
        source,
        source_labels,
        target,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )

    expected_classes = {1, "1"}
    assert transfer.classes.dtype == object
    assert fit_transfer.classes.dtype == object
    assert set(transfer.classes.tolist()) == expected_classes
    assert set(fit_transfer.classes.tolist()) == expected_classes
    assert set(transfer.pseudo_labels.tolist()).issubset(expected_classes)
    assert set(predictions.tolist()).issubset(expected_classes)
