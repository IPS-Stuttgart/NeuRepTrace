from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.feature_summary import summarize_features


def test_summarize_features_returns_column_statistics() -> None:
    x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])

    result = summarize_features(x)

    assert np.allclose(result.mean, [3.0, 14.0 / 3.0])
    assert np.allclose(result.minimum, [1.0, 2.0])
    assert np.allclose(result.maximum, [5.0, 8.0])
    assert result.metadata["feature_summary_n_rows"] == 3
    assert result.metadata["feature_summary_n_features"] == 2


def test_summarize_features_ddof_guardrail() -> None:
    result = summarize_features([[1.0, 2.0]], ddof=5)
    assert result.metadata["feature_summary_ddof"] == 0
    assert np.allclose(result.scale, [0.0, 0.0])


def test_summarize_features_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        summarize_features([1.0, 2.0])
    with pytest.raises(ValueError):
        summarize_features([[1.0, np.nan]])
    with pytest.raises(ValueError):
        summarize_features([[1.0]], ddof=-1)


def test_summarize_features_rejects_non_scalar_ddof() -> None:
    for bad in ([1], (1,), {"value": 1}, {1}, np.asarray([1]), np.asarray([[1]])):
        with pytest.raises(ValueError):
            summarize_features([[1.0], [2.0]], ddof=bad)  # type: ignore[arg-type]


def test_summarize_features_accepts_numpy_scalar_ddof() -> None:
    result = summarize_features([[1.0], [2.0]], ddof=np.asarray(0))
    assert result.metadata["feature_summary_ddof"] == 0
