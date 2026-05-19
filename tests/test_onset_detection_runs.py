from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from reptrace.onset_detection import (
    _detection_runs,
    _first_detection_run,
    _has_matching_threshold_annotation,
    _prepare_thresholded_observations,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6],
            "window_start": [-0.01, 0.09, 0.19, 0.39, 0.49, 0.59],
            "window_stop": [0.01, 0.11, 0.21, 0.41, 0.51, 0.61],
            "_onset_score": [0.90, 0.92, 0.20, 0.91, 0.93, 0.94],
            "predicted_label": [0, 0, 0, 1, 1, 1],
        }
    )


def _thresholded_observations_with_stale_threshold() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-0.2, -0.1, 0.1],
            "sequence_id": [0, 0, 0],
            "prob_class_0": [0.1, 0.2, 0.9],
            "prob_class_1": [0.9, 0.8, 0.1],
            "onset_score": [0.9, 0.8, 0.9],
            "score_threshold": [0.1, 0.1, 0.1],
            "above_threshold": [True, True, True],
            "score_column": ["confidence", "confidence", "confidence"],
            "threshold_method": ["point", "point", "point"],
            "threshold_quantile": [0.5, 0.5, 0.5],
            "threshold_window_start": [-0.2, -0.2, -0.2],
            "threshold_window_stop": [-0.1, -0.1, -0.1],
            "min_consecutive": [1, 1, 1],
            "min_duration": [float("nan")] * 3,
            "require_stable_prediction": pd.Series([False, False, False], dtype=object),
        }
    )


@pytest.mark.parametrize(
    "metadata_column",
    ["score_column", "threshold_method", "require_stable_prediction"],
)
def test_threshold_annotation_reuse_rejects_partially_missing_metadata(
    metadata_column,
):
    observations = _thresholded_observations_with_stale_threshold()
    observations.loc[1, metadata_column] = None

    assert not _has_matching_threshold_annotation(
        observations,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=0.5,
        score_column="confidence",
        threshold_method="point",
        min_consecutive=1,
        min_duration=None,
        require_stable_prediction=False,
    )


@pytest.mark.parametrize(
    ("metadata_column", "stale_value"),
    [
        ("score_column", "probability_true_class"),
        ("threshold_method", "max_run"),
        ("threshold_quantile", 0.9),
        ("threshold_window_start", -0.3),
        ("threshold_window_stop", -0.05),
        ("min_consecutive", 2),
        ("min_duration", 0.1),
        ("require_stable_prediction", True),
    ],
)
def test_threshold_annotation_reuse_rejects_parameter_mismatches(
    metadata_column,
    stale_value,
):
    observations = _thresholded_observations_with_stale_threshold()
    observations[metadata_column] = stale_value

    assert not _has_matching_threshold_annotation(
        observations,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=0.5,
        score_column="confidence",
        threshold_method="point",
        min_consecutive=1,
        min_duration=None,
        require_stable_prediction=False,
    )


def test_prepare_thresholded_observations_recomputes_partial_metadata_match():
    observations = _thresholded_observations_with_stale_threshold()
    observations.loc[1, "require_stable_prediction"] = None

    thresholded = _prepare_thresholded_observations(
        observations,
        threshold_window=(-0.2, -0.1),
        threshold_quantile=0.5,
        score_column="confidence",
        threshold_method="point",
        min_consecutive=1,
        min_duration=None,
        require_stable_prediction=False,
    )

    assert thresholded["score_threshold"].tolist() == pytest.approx([0.85, 0.85, 0.85])
    assert thresholded["above_threshold"].tolist() == [True, False, True]


def test_detection_runs_returns_all_valid_segments_and_preserves_first_run_behavior():
    candidates = _candidates()

    runs = _detection_runs(
        candidates,
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
    )
    first = _first_detection_run(
        candidates,
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
    )

    assert len(runs) == 2
    assert [float(run.iloc[0]["time"]) for run in runs] == [0.0, 0.4]
    pdt.assert_frame_equal(first, runs[0])


def test_detection_runs_can_merge_close_segments_with_same_prediction():
    candidates = _candidates()
    candidates["predicted_label"] = 0

    runs = _detection_runs(
        candidates,
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
        merge_gap=0.35,
    )

    assert len(runs) == 1
    assert runs[0]["time"].tolist() == [0.0, 0.1, 0.4, 0.5, 0.6]


def test_detection_runs_refractory_suppresses_close_duplicate_runs():
    runs = _detection_runs(
        _candidates(),
        threshold=0.80,
        min_consecutive=2,
        min_duration=None,
        require_stable_prediction=True,
        refractory=0.5,
    )

    assert len(runs) == 1
    assert runs[0]["time"].tolist() == [0.0, 0.1]


@pytest.mark.parametrize("kwargs", [{"merge_gap": -0.1}, {"refractory": -0.1}])
def test_detection_runs_rejects_negative_gap_controls(kwargs):
    with pytest.raises(ValueError):
        _detection_runs(
            _candidates(),
            threshold=0.80,
            min_consecutive=2,
            min_duration=None,
            require_stable_prediction=True,
            **kwargs,
        )
