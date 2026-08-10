from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.katja_press_timing import (
    behavioral_press_times_seconds,
    detect_trigger_onsets,
    finalize_press_timing_rows,
    match_trigger_onsets,
)


def test_behavioral_press_times_add_cue_duration():
    behavior = {
        "timing_ms": np.array([[0, 100, 200, 300, 400]], dtype=float),
        "cue_duration_ms": np.array([500.0]),
    }
    np.testing.assert_allclose(
        behavioral_press_times_seconds(behavior),
        [[0.5, 0.6, 0.7, 0.8, 0.9]],
    )


def test_detect_trigger_onsets_and_collapse_bounce():
    signal = np.zeros(1000)
    signal[100:110] = 1
    signal[102] = 0
    signal[300:310] = 2
    result = detect_trigger_onsets(
        signal,
        sampling_frequency=1000.0,
        time_onset=-0.2,
        min_separation_seconds=0.01,
    )
    np.testing.assert_array_equal(result.sample_indices, [100, 300])
    np.testing.assert_allclose(result.times_seconds, [-0.1, 0.1])


def test_match_trigger_onsets_skips_extra_and_marks_missing():
    behavioral = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    triggers = np.array([0.4, 1.03, 2.03, 3.03, 5.03, 5.8])
    result = match_trigger_onsets(
        behavioral,
        triggers,
        expected_lag_seconds=0.03,
        tolerance_seconds=0.08,
    )
    np.testing.assert_allclose(
        result.trigger_times_seconds[[0, 1, 2, 4]],
        [1.03, 2.03, 3.03, 5.03],
    )
    assert np.isnan(result.trigger_times_seconds[3])


def test_finalize_uses_subject_median_lag_for_missing_trigger():
    rows = pd.DataFrame(
        {
            "subject": ["s1"] * 5,
            "trial_id": [1] * 5,
            "press_position": [1, 2, 3, 4, 5],
            "behavior_time_seconds": [1, 2, 3, 4, 5],
            "trigger_time_seconds": [1.02, 2.04, np.nan, 4.03, 5.03],
            "trigger_matched": [True, True, False, True, True],
            "trigger_minus_behavior_ms": [20, 40, np.nan, 30, 30],
            "correct_order": [True] * 5,
        }
    )
    finalized, summary = finalize_press_timing_rows(rows, default_lag_seconds=0.03)
    assert (
        finalized.loc[2, "recommended_time_source"]
        == "behavior_plus_subject_median_trigger_lag"
    )
    assert finalized.loc[2, "recommended_time_seconds"] == 3.03
    assert summary.loc[0, "median_trigger_lag_ms"] == 30.0
