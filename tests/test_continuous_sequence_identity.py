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


def _stream_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stream_id": ["run-a", "run-a", "run-b", "run-b"],
            "sample_index": [0, 1, 0, 1],
            "time": [0.0, 0.1, 0.0, 0.1],
        }
    )


def test_continuous_scan_groups_time_rows_by_stream() -> None:
    standardized = _standardize(_standardize_stream_observations, _stream_observations())

    assert standardized["sequence_id"].tolist() == ["run-a", "run-a", "run-b", "run-b"]
    assert standardized.groupby("stream_id")["sequence_id"].nunique().eq(1).all()


def test_generic_continuous_standardizer_preserves_sample_identity() -> None:
    standardized = _standardize(standardize_continuous_observations, _stream_observations())

    assert standardized["sequence_id"].tolist() == [
        ("run-a", 0),
        ("run-a", 1),
        ("run-b", 0),
        ("run-b", 1),
    ]


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
