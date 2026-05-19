import numpy as np
import pytest

from neureptrace.decoding.epoch_features import (
    EpochArray,
    extract_matching_baseline_features,
    extract_window_features,
    normalize_window_features,
    resolve_time_window,
)


def _epochs():
    data = np.array(
        [
            [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]],
            [[100, 101, 102, 103, 104], [110, 111, 112, 113, 114]],
        ],
        dtype=float,
    )
    times = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    return EpochArray(
        data=data,
        times=times,
        labels=np.array(["a", "b"]),
        groups=np.array([1, 2]),
        channel_names=("C1", "C2"),
    )


def test_extract_window_features_uses_channel_time_order_by_default():
    result = extract_window_features(_epochs(), window=(0.01, 0.03))

    assert result.features.tolist() == [
        [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
        [101.0, 102.0, 103.0, 111.0, 112.0, 113.0],
    ]
    assert result.labels.tolist() == ["a", "b"]
    assert result.groups.tolist() == [1, 2]
    assert result.n_channels == 2
    assert result.n_window_samples == 3
    assert result.feature_order == "channel_time"
    assert result.channel_names == ("C1", "C2")
    assert result.window.sample_slice == slice(1, 4)
    assert result.window.center == pytest.approx(0.02)


def test_extract_window_features_can_match_matlab_time_channel_order():
    result = extract_window_features(_epochs(), center=0.02, size=0.02, feature_order="time-channel")

    assert result.features.tolist() == [
        [1.0, 11.0, 2.0, 12.0, 3.0, 13.0],
        [101.0, 111.0, 102.0, 112.0, 103.0, 113.0],
    ]
    assert result.feature_order == "time_channel"


def test_extract_matching_baseline_features_uses_target_sample_count_and_disjointness():
    target = extract_window_features(_epochs(), window=(0.03, 0.04), feature_order="time_channel")
    baseline = extract_matching_baseline_features(_epochs(), target, baseline_start=0.00)

    assert baseline.features.tolist() == [
        [0.0, 10.0, 1.0, 11.0],
        [100.0, 110.0, 101.0, 111.0],
    ]
    assert baseline.n_window_samples == target.n_window_samples
    assert baseline.feature_order == "time_channel"
    assert baseline.window.sample_slice == slice(0, 2)


def test_extract_matching_baseline_features_rejects_overlapping_window():
    target = extract_window_features(_epochs(), window=(0.01, 0.03))

    with pytest.raises(ValueError, match="overlap"):
        extract_matching_baseline_features(_epochs(), target, baseline_start=0.02)


def test_normalize_window_features_uses_reference_statistics_with_safe_constant_columns():
    features = np.array([[5.0, 2.0, 13.0]])
    reference = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 9.0]])

    normalized, normalizer = normalize_window_features(features, reference_features=reference, mode="z-score")

    assert normalized.tolist() == [[3.0, 0.0, 3.0]]
    assert normalizer.reference_rows == 2
    assert normalizer.scale.tolist() == [1.0, 1.0, 2.0]


def test_resolve_time_window_rejects_unsupported_bounds():
    with pytest.raises(ValueError, match="outside"):
        resolve_time_window(np.array([0.0, 0.01, 0.02]), window=(0.0, 0.04))
