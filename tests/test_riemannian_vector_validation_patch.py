from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.riemannian import (
    fit_predict_riemannian_procrustes,
    fit_predict_riemannian_transfer,
    riemannian_procrustes_transfer_features,
    riemannian_tangent_transfer_features,
)


def _covariances(n_rows: int) -> np.ndarray:
    covariances = []
    for index in range(n_rows):
        scale = 1.0 + 0.05 * index
        covariances.append(
            np.array(
                [
                    [1.0 * scale + 0.02, 0.03],
                    [0.03, 1.5 / scale + 0.02],
                ],
                dtype=float,
            )
        )
    return np.stack(covariances, axis=0)


def test_riemannian_transfer_accepts_column_source_domains():
    source = _covariances(4)
    target = _covariances(3)
    source_domains = np.array(["s1", "s1", "s2", "s2"], dtype=object)[:, None]

    transfer = riemannian_tangent_transfer_features(source, target, source_domains=source_domains)

    assert transfer.source_domains.shape == (4,)
    assert sorted(np.unique(transfer.source_domains).tolist()) == ["s1", "s2"]
    assert transfer.source_features.shape[0] == source.shape[0]


def test_riemannian_procrustes_accepts_column_source_domains():
    source = _covariances(4)
    target = _covariances(3)
    source_domains = np.array(["s1", "s1", "s2", "s2"], dtype=object)[:, None]

    transfer = riemannian_procrustes_transfer_features(source, target, source_domains=source_domains)

    assert transfer.source_domains.shape == (4,)
    assert set(transfer.source_alignments) == {"s1", "s2"}


def test_riemannian_transfer_rejects_matrix_source_domains():
    source = _covariances(4)
    target = _covariances(3)
    source_domains = np.array(
        [["s1", "run1"], ["s1", "run2"], ["s2", "run1"], ["s2", "run2"]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="source_domains must be one-dimensional"):
        riemannian_tangent_transfer_features(source, target, source_domains=source_domains)


def test_riemannian_transfer_rejects_matrix_source_labels_even_when_flat_length_matches():
    source = _covariances(8)
    target = _covariances(3)
    source_labels = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=int)

    with pytest.raises(ValueError, match="source_labels must be one-dimensional"):
        fit_predict_riemannian_transfer(source, source_labels, target)

    with pytest.raises(ValueError, match="source_labels must be one-dimensional"):
        fit_predict_riemannian_procrustes(source, source_labels, target)
