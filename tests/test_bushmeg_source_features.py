import numpy as np

from neureptrace.bushmeg_source_loso import (
    WindowSpec,
    _source_class_pseudotrials,
    _window_features,
    normalize_source_feature_kind,
)


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


def test_evoked_slope_window_features_append_one_slope_contrast_per_bin():
    rng = np.random.default_rng(47)
    sfreq = 500.0
    times = np.arange(500, dtype=float) / sfreq - 0.2
    data = rng.normal(size=(5, 4, times.size)).astype(np.float32)

    features = _window_features(
        data,
        times,
        WindowSpec(center=0.1, width=0.2),
        temporal_bins=2,
        feature_kind="evoked_slope",
    )

    assert features.shape == (5, 4 * 2 * 2)
    assert np.isfinite(features).all()


def test_evoked_dct_window_features_return_compact_temporal_coefficients():
    rng = np.random.default_rng(49)
    sfreq = 500.0
    times = np.arange(500, dtype=float) / sfreq - 0.2
    data = rng.normal(size=(5, 4, times.size)).astype(np.float32)

    features = _window_features(
        data,
        times,
        WindowSpec(center=0.1, width=0.2),
        temporal_bins=4,
        feature_kind="evoked_dct",
    )

    assert features.shape == (5, 4 * 4)
    assert np.isfinite(features).all()


def test_temporal_dct_alias_normalizes_to_evoked_dct():
    assert normalize_source_feature_kind("temporal-dct") == "evoked_dct"


def test_source_class_pseudotrials_replace_averages_within_subject_class():
    features = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
            [100.0, 0.0],
            [102.0, 0.0],
            [110.0, 0.0],
            [112.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    subjects = np.asarray(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"], dtype=object)

    pseudo_features, pseudo_labels, pseudo_subjects = _source_class_pseudotrials(
        features,
        labels,
        subjects,
        n_classes=2,
        pseudotrials_per_class=1,
        pseudotrial_mode="replace",
        pseudotrial_seed=7,
    )

    lookup = {
        (str(subject), int(label)): row
        for row, label, subject in zip(pseudo_features, pseudo_labels, pseudo_subjects, strict=True)
    }
    assert pseudo_features.shape == (4, 2)
    np.testing.assert_allclose(lookup[("s1", 0)], [1.0, 0.0])
    np.testing.assert_allclose(lookup[("s1", 1)], [11.0, 0.0])
    np.testing.assert_allclose(lookup[("s2", 0)], [101.0, 0.0])
    np.testing.assert_allclose(lookup[("s2", 1)], [111.0, 0.0])


def test_source_class_pseudotrials_augment_preserves_single_trial_prefix():
    features = np.arange(16, dtype=np.float32).reshape(8, 2)
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    subjects = np.asarray(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"], dtype=object)

    augmented_features, augmented_labels, augmented_subjects = _source_class_pseudotrials(
        features,
        labels,
        subjects,
        n_classes=2,
        pseudotrials_per_class=1,
        pseudotrial_mode="augment",
        pseudotrial_seed=11,
    )

    assert augmented_features.shape == (12, 2)
    np.testing.assert_array_equal(augmented_features[:8], features)
    np.testing.assert_array_equal(augmented_labels[:8], labels)
    np.testing.assert_array_equal(augmented_subjects[:8], subjects)
