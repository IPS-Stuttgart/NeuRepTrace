from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_whiten import fit_source_whiten, fit_source_whiten_transform


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[True, 1.0], [False, 0.0]], dtype=object),
        [[True, 0.0], [False, 1.0]],
        ((value for value in row) for row in [[True, 0.0], [False, 1.0]]),
    ],
)
def test_source_whiten_rejects_boolean_source_features(features) -> None:
    with pytest.raises(ValueError, match="source_features.*boolean flags"):
        fit_source_whiten_transform(features)


def test_source_whiten_rejects_boolean_test_features() -> None:
    with pytest.raises(ValueError, match="test_features.*boolean flags"):
        fit_source_whiten(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            test_features=np.asarray([[True, False]], dtype=bool),
            config={"method": "pca", "n_components": 1},
        )


def test_source_whiten_still_accepts_one_pass_numeric_iterables() -> None:
    source_features = ((value for value in row) for row in [[0.0, 1.0], [1.0, 0.0]])
    test_features = ((value for value in row) for row in [[0.5, 0.5]])

    result = fit_source_whiten(
        source_features=source_features,
        test_features=test_features,
        config={"method": "pca", "n_components": 1},
    )

    assert result.train_features.shape == (2, 1)
    assert result.test_features.shape == (1, 1)
