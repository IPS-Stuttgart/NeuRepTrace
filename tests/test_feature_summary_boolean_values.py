from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.feature_summary import summarize_features


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[1.0, np.bool_(False)], [2.0, 3.0]], dtype=object),
        [[True, 0.0], [1.0, 2.0]],
    ],
)
def test_summarize_features_rejects_boolean_values(features: object) -> None:
    with pytest.raises(ValueError, match="boolean flags"):
        summarize_features(features)  # type: ignore[arg-type]


def test_summarize_features_still_accepts_zero_one_numeric_values() -> None:
    result = summarize_features([[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(result.mean, [0.5, 0.5])
