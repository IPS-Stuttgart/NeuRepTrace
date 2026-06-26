from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.bushmeg_cue_source_weights import (
    CueSourceWeights,
    CueSubjectData,
    _channel_logvar,
    _crop_data,
    _evoked_bin_means,
    _evoked_gfp_bins,
    _truthy,
    _unit_feature_vector,
    _window_tuple,
    cue_subject_feature_vector,
    normalize_cue_feature_kinds,
    normalize_cue_source_weighting_mode,
)


def test_cue_source_weights_softmax_topk_returns_mean_one_weights():
    features = {
        "target": np.array([1.0, 0.0, 0.0]),
        "near": np.array([0.9, 0.1, 0.0]),
        "far": np.array([-1.0, 0.0, 0.0]),
        "mid": np.array([0.5, 0.5, 0.0]),
    }
    cue_weights = CueSourceWeights(
        features,
        mode="softmax_top_k",
        top_k=2,
        temperature=0.25,
        blend=1.0,
    )

    weights = cue_weights.for_fold("target", ["near", "far", "mid"])

    assert set(weights) == {"near", "far", "mid"}
    assert np.isclose(np.mean(list(weights.values())), 1.0)
    assert weights["near"] > weights["mid"] > weights["far"]
    assert weights["far"] == 0.0


def test_cue_source_weights_blend_keeps_nonnearest_sources_alive():
    cue_weights = CueSourceWeights(
        {
            "target": np.array([1.0, 0.0]),
            "near": np.array([1.0, 0.1]),
            "far": np.array([-1.0, 0.0]),
        },
        mode="top_k",
        top_k=1,
        blend=0.5,
    )

    weights = cue_weights.for_fold("target", ["near", "far"])

    assert np.isclose(np.mean(list(weights.values())), 1.0)
    assert weights["near"] > weights["far"] > 0.0


def test_cue_feature_aliases_are_normalized():
    assert normalize_cue_source_weighting_mode("softmax-topk") == "softmax_top_k"
    assert normalize_cue_feature_kinds(["gfp", "baseline-var", "evoked"]) == (
        "evoked_gfp",
        "baseline_logvar",
        "evoked_mean",
    )


def test_cue_subject_feature_vector_concatenates_requested_parts():
    data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]],
        ],
        dtype=np.float32,
    )
    subject = CueSubjectData(
        subject="s1",
        data=data,
        times=np.array([-0.10, -0.05, 0.10, 0.20]),
        metadata=pd.DataFrame(),
    )

    vector = cue_subject_feature_vector(
        subject,
        feature_kinds=["baseline_logvar", "evoked_gfp", "evoked_mean"],
        baseline_window=(-0.10, -0.05),
        response_window=(0.10, 0.20),
        temporal_bins=2,
    )

    assert vector.shape == (2 + 2 + 4,)
    assert np.all(np.isfinite(vector))


def test_cue_evoked_bins_reject_more_bins_than_response_samples():
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([-0.1, 0.1, 0.2])

    with pytest.raises(ValueError, match="temporal_bins .* must not exceed the 2 cue response-window sample"):
        _evoked_bin_means(data, times, (0.1, 0.2), temporal_bins=3)

    with pytest.raises(ValueError, match="temporal_bins .* must not exceed the 2 cue response-window sample"):
        _evoked_gfp_bins(data, times, (0.1, 0.2), temporal_bins=3)


def test_cue_source_weights_rejects_boolean_and_fractional_controls():
    features = {
        "target": np.array([1.0, 0.0]),
        "source": np.array([0.9, 0.1]),
    }

    with pytest.raises(ValueError, match="temperature"):
        CueSourceWeights(features, temperature=True)

    with pytest.raises(ValueError, match="blend"):
        CueSourceWeights(features, blend=True)

    with pytest.raises(ValueError, match="top_k must be an integer"):
        CueSourceWeights(features, top_k=1.5)

    with pytest.raises(ValueError, match="top_k must be an integer"):
        CueSourceWeights(features, top_k=True)

    with pytest.raises(ValueError, match="top_k must be an integer"):
        CueSourceWeights(features, top_k=[1])


def test_cue_feature_helpers_reject_silent_numeric_coercions():
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([-0.1, 0.1, 0.2])
    subject = CueSubjectData(subject="s1", data=data, times=times, metadata=pd.DataFrame())

    with pytest.raises(ValueError, match="boolean flag"):
        _truthy(0.5)

    with pytest.raises(ValueError, match="Cue calibration windows"):
        _window_tuple("0,1", (0.0, 1.0))

    with pytest.raises(ValueError, match="Cue calibration windows"):
        _window_tuple(0.1, (0.0, 1.0))

    with pytest.raises(ValueError, match="cue_window_start"):
        _window_tuple([False, 0.2], (0.0, 1.0))

    with pytest.raises(ValueError, match="cue_tmin"):
        _crop_data(data, times, tmin=True, tmax=None)

    with pytest.raises(ValueError, match="cue_feature_epsilon"):
        _channel_logvar(data, times, (-0.1, 0.1), name="cue baseline", epsilon=True)

    with pytest.raises(ValueError, match="temporal_bins must be an integer"):
        _evoked_bin_means(data, times, (0.1, 0.2), temporal_bins=1.5)

    with pytest.raises(ValueError, match="temporal_bins must be an integer"):
        _evoked_gfp_bins(data, times, (0.1, 0.2), temporal_bins=True)

    with pytest.raises(ValueError, match="temporal_bins must be an integer"):
        cue_subject_feature_vector(subject, feature_kinds=["evoked_mean"], response_window=(0.1, 0.2), temporal_bins=1.5)

    with pytest.raises(ValueError, match="cue_feature_epsilon"):
        _unit_feature_vector(np.array([1.0, 2.0]), epsilon=True)
