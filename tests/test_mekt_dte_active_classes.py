from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import mekt


def _spd_from_diagonal(diagonal):
    diagonal = np.asarray(diagonal, dtype=float)
    return np.diag(diagonal)


def _three_class_domain_covariances():
    source = np.stack(
        [
            _spd_from_diagonal([1.8, 0.7, 0.5]),
            _spd_from_diagonal([1.7, 0.8, 0.5]),
            _spd_from_diagonal([0.6, 1.7, 0.5]),
            _spd_from_diagonal([0.7, 1.6, 0.5]),
            _spd_from_diagonal([0.5, 0.8, 1.7]),
            _spd_from_diagonal([0.6, 0.7, 1.8]),
            _spd_from_diagonal([1.1, 0.9, 1.2]),
            _spd_from_diagonal([1.0, 0.8, 1.3]),
        ],
        axis=0,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2, 0, 2])
    domains = np.asarray(["near", "near", "near", "near", "far", "far", "far", "far"])
    target = np.stack(
        [
            _spd_from_diagonal([1.6, 0.9, 0.5]),
            _spd_from_diagonal([0.7, 1.5, 0.6]),
            _spd_from_diagonal([1.5, 1.0, 0.6]),
            _spd_from_diagonal([0.8, 1.4, 0.5]),
        ],
        axis=0,
    )
    return source, labels, domains, target


def test_mekt_dte_selection_recomputes_active_source_classes(monkeypatch) -> None:
    source, labels, domains, target = _three_class_domain_covariances()
    monkeypatch.setattr(mekt, "domain_transferability_scores", lambda *args, **kwargs: {"near": 2.0, "far": 1.0})

    transfer = mekt.mekt_transfer_features(
        source,
        labels,
        target,
        source_domains=domains,
        dte_top_k=1,
        n_components=2,
        n_iterations=1,
        n_neighbors=2,
    )

    assert transfer.selected_source_domains.tolist() == ["near"]
    assert transfer.classes.tolist() == [0, 1]
    assert set(transfer.pseudo_labels.tolist()).issubset({0, 1})

    with pytest.raises(ValueError, match="initial_pseudo_labels must contain only source classes"):
        mekt.mekt_transfer_features(
            source,
            labels,
            target,
            source_domains=domains,
            dte_top_k=1,
            n_components=2,
            n_iterations=1,
            n_neighbors=2,
            initial_pseudo_labels=np.full(target.shape[0], 2),
        )
