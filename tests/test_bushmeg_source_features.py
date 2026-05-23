import numpy as np

from neureptrace.bushmeg_source_loso import WindowSpec, _window_features


def test_bandpower_window_features_return_finite_band_bin_matrix():
    rng = np.random.default_rng(41)
    sfreq = 500.0
    times = np.arange(500, dtype=float) / sfreq - 0.2
    data = rng.normal(size=(5, 4, times.size)).astype(np.float32)

    features = _window_features(
        data,
        times,
        WindowSpec(center=0.1, width=0.2),
        temporal_bins=2,
        feature_kind="bandpower",
    )

    assert features.shape == (5, 4 * 2 * 4)
    assert np.isfinite(features).all()


def test_evoked_bandpower_concatenates_evoked_and_bandpower_features():
    rng = np.random.default_rng(43)
    sfreq = 500.0
    times = np.arange(500, dtype=float) / sfreq - 0.2
    data = rng.normal(size=(5, 4, times.size)).astype(np.float32)

    features = _window_features(data, times, WindowSpec(center=0.1, width=0.2), temporal_bins=2, feature_kind="evoked_bandpower")

    assert features.shape == (5, 4 * 2 + 4 * 2 * 4)
    assert np.isfinite(features).all()
