import numpy as np
import pytest

from neureptrace.decoding.covariance_features import (
    COVARIANCE_FEATURE_MODES,
    CovarianceWindow,
    channel_subset_indices,
    covariance_feature_vector,
    normalize_covariance_feature_mode,
    trial_covariance,
    window_covariance_features,
)


def test_covariance_feature_modes_are_finite_and_pymegdec_compatible():
    signal = np.array(
        [
            [1.0, 2.0, 4.0, 5.0],
            [2.0, 3.0, 5.0, 7.0],
            [7.0, 6.0, 4.0, 3.0],
        ],
        dtype=float,
    )

    features_by_mode = {
        mode: covariance_feature_vector(signal, mode, shrinkage=0.1, epsilon=1e-6)
        for mode in COVARIANCE_FEATURE_MODES
    }

    assert normalize_covariance_feature_mode("logeig-covariance") == "logeuclidean_covariance"
    assert normalize_covariance_feature_mode("covariance") == "covariance_upper"
    assert normalize_covariance_feature_mode("correlation") == "correlation_upper"
    assert normalize_covariance_feature_mode("logvariance") == "variance"
    assert features_by_mode["variance"].shape == (3,)
    assert features_by_mode["covariance_upper"].shape == (6,)
    assert features_by_mode["correlation_upper"].shape == (6,)
    assert features_by_mode["logeuclidean_covariance"].shape == (6,)
    for features in features_by_mode.values():
        assert np.all(np.isfinite(features))
    np.testing.assert_allclose(features_by_mode["correlation_upper"][[0, 3, 5]], np.ones(3))


def test_window_covariance_features_selects_window_and_channels():
    times = np.linspace(-0.1, 0.3, 9)
    data = np.arange(2 * 5 * times.size, dtype=float).reshape(2, 5, times.size)
    window = CovarianceWindow(name="post", start=0.0, stop=0.2)

    features = window_covariance_features(
        data,
        times,
        window,
        mode="variance",
        shrinkage=0.2,
        epsilon=1e-6,
        max_channels=3,
    )

    assert features.shape == (2, 3)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(channel_subset_indices(5, 3), np.array([0, 2, 4]))


def test_covariance_feature_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="finite"):
        trial_covariance([[1.0, np.nan]])
    with pytest.raises(ValueError, match="positive"):
        channel_subset_indices(0, 3)
    with pytest.raises(ValueError, match="does not overlap"):
        window_covariance_features(
            np.ones((1, 2, 3)),
            np.array([0.0, 0.1, 0.2]),
            CovarianceWindow("late", 1.0, 1.1),
        )
