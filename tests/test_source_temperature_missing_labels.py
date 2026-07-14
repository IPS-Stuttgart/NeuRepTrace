from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding import source_temperature


def test_fit_source_temperature_scaling_reports_missing_list_label() -> None:
    with pytest.raises(ValueError, match="absent from classes"):
        source_temperature.fit_source_temperature_scaling(
            source_probabilities=[[0.9, 0.1], [0.1, 0.9]],
            source_labels=[[1, 1], [9, 9]],
            test_probabilities=[[0.5, 0.5]],
            classes=[[1, 1], [2, 2]],
        )


def test_fit_source_temperature_scaling_matches_distinct_nan_label_objects() -> None:
    result = source_temperature.fit_source_temperature_scaling(
        source_probabilities=[[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]],
        source_labels=[float("nan"), "seen", float("nan")],
        test_probabilities=[[0.55, 0.45]],
        classes=[float("nan"), "seen"],
        config={"temperatures": (1.0,)},
    )

    assert result.probabilities.shape == (1, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), [1.0])
    assert result.metadata["source_temperature_n_classes"] == 2


def test_fit_source_temperature_scaling_matches_pandas_missing_labels() -> None:
    result = source_temperature.fit_source_temperature_scaling(
        source_probabilities=[[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]],
        source_labels=pd.Series([pd.NA, "seen", pd.NA], dtype="object"),
        test_probabilities=[[0.55, 0.45]],
        classes=pd.Index([pd.NA, "seen"], dtype="object"),
        config={"temperatures": (1.0,)},
    )

    assert result.probabilities.shape == (1, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), [1.0])
    assert result.metadata["source_temperature_n_classes"] == 2
