from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scale import (
    apply_source_feature_scale,
    fit_source_feature_scale,
    fit_source_feature_scale_stats,
)


@pytest.mark.parametrize(
    ("source_features", "test_features", "message"),
    [
        (np.asarray([[True, False], [False, True]], dtype=bool), [[0.0, 1.0]], "source_features"),
        ([[0.0, 1.0], [1.0, 0.0]], np.asarray([[True, False]], dtype=object), "test_features"),
        (np.asarray([iter([True, 0.0]), iter([False, 1.0])], dtype=object), [[0.0, 1.0]], "source_features"),
    ],
)
def test_source_scale_rejects_boolean_feature_values(source_features: object, test_features: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_source_feature_scale(source_features=source_features, test_features=test_features)


def test_source_scale_rejects_boolean_values_when_applying_stats() -> None:
    stats = fit_source_feature_scale_stats([[0.0], [1.0]])

    with pytest.raises(ValueError, match="features"):
        apply_source_feature_scale([[True]], stats)
