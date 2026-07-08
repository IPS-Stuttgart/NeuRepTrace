from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_bagging import fit_source_bagging_decoder


def test_source_bagging_collapses_repeated_nan_labels() -> None:
    source_features = np.asarray(
        [
            [-1.2, 0.0],
            [-1.0, 0.1],
            [-0.8, -0.1],
            [0.8, 0.0],
            [1.0, 0.1],
            [1.2, -0.1],
        ],
        dtype=float,
    )
    source_labels = np.asarray([np.nan, np.float64("nan"), np.nan, "seen", "seen", "seen"], dtype=object)
    test_features = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 3, "sample_fraction": 1.0, "random_state": 7},
    )

    assert result.metadata["source_bagging_n_classes"] == 2
    assert result.classes.shape == (2,)
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "seen"
    assert result.probabilities.shape == (2, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))


def test_source_bagging_collapses_composite_nan_labels() -> None:
    source_features = np.asarray(
        [
            [-1.2, 0.0],
            [-0.9, 0.1],
            [0.9, -0.1],
            [1.2, 0.0],
        ],
        dtype=float,
    )
    source_labels = [("missing", np.nan), ("missing", np.float64("nan")), ("known", 1.0), ("known", 1.0)]
    test_features = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 1, "sample_fraction": 1.0, "random_state": 11},
    )

    assert result.metadata["source_bagging_n_classes"] == 2
    first_class, second_class = result.classes.tolist()
    assert first_class[0] == "missing"
    assert np.isnan(first_class[1])
    assert second_class == ("known", 1.0)
    assert result.probabilities.shape == (2, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))
