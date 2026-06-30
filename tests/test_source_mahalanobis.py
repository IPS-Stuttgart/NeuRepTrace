from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_mahalanobis import fit_source_mahalanobis_decoder, source_mahalanobis_config, tied_covariance


def test_source_mahalanobis_preserves_composite_tuple_labels():
    source_features = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [3.0, 3.0],
            [3.2, 2.9],
        ]
    )
    source_labels = [
        ("run-01", "face"),
        ("run-01", "face"),
        ("run-01", "scene"),
        ("run-01", "scene"),
    ]
    test_features = np.array([[0.1, 0.0], [3.1, 3.0]])

    result = fit_source_mahalanobis_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config=source_mahalanobis_config(prior="uniform"),
    )

    assert result.classes.tolist() == [("run-01", "face"), ("run-01", "scene")]
    assert result.predictions.tolist() == [("run-01", "face"), ("run-01", "scene")]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_tied_covariance_preserves_explicit_composite_classes():
    source_features = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [3.0, 3.0],
            [3.2, 2.9],
        ]
    )
    source_labels = np.array(
        [
            ["run-01", "face"],
            ["run-01", "face"],
            ["run-01", "scene"],
            ["run-01", "scene"],
        ],
        dtype=object,
    )

    covariance = tied_covariance(
        source_features,
        source_labels,
        classes=[("run-01", "face"), ("run-01", "scene")],
        regularization=0.1,
    )

    assert covariance.shape == (2, 2)
    assert np.all(np.isfinite(covariance))
