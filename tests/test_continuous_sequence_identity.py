from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from neureptrace.continuous_observations import standardize_continuous_observations
from neureptrace.continuous_stimulus_scan import _standardize_stream_observations


_STANDARDIZERS: tuple[Callable[..., pd.DataFrame], ...] = (
    standardize_continuous_observations,
    _standardize_stream_observations,
)


def _standardize(standardizer: Callable[..., pd.DataFrame], observations: pd.DataFrame) -> pd.DataFrame:
    return standardizer(
        observations,
        subject=None,
        split_id="split",
        slice_seed=13,
        decoder="logistic",
        emission_mode="calibrated",
        train_time=0.15,
        preprocessing_hash="preprocessing",
        model_hash="model",
    )


@pytest.mark.parametrize("standardizer", _STANDARDIZERS)
def test_continuous_standardizers_group_time_rows_by_stream(standardizer: Callable[..., pd.DataFrame]) -> None:
    observations = pd.DataFrame(
        {
            "stream_id": ["run-a", "run-a", "run-b", "run-b"],
            "sample_index": [0, 1, 0, 1],
            "time": [0.0, 0.1, 0.0, 0.1],
        }
    )

    standardized = _standardize(standardizer, observations)

    assert standardized["sequence_id"].tolist() == ["run-a", "run-a", "run-b", "run-b"]
    assert standardized.groupby("stream_id")["sequence_id"].nunique().eq(1).all()


@pytest.mark.parametrize("standardizer", _STANDARDIZERS)
def test_continuous_standardizers_preserve_explicit_sequence_ids(standardizer: Callable[..., pd.DataFrame]) -> None:
    observations = pd.DataFrame(
        {
            "stream_id": ["run-a", "run-a"],
            "sample_index": [0, 1],
            "sequence_id": ["trial-7", "trial-7"],
            "time": [0.0, 0.1],
        }
    )

    standardized = _standardize(standardizer, observations)

    assert standardized["sequence_id"].tolist() == ["trial-7", "trial-7"]
