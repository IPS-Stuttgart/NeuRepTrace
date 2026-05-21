import numpy as np

from neureptrace.mne_time_decode_foldlocal import _normalize_epoch_data_for_fold


def test_subject_z_normalization_uses_train_fold_statistics_only():
    data = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0]],
            [[100.0, 100.0, 100.0]],
        ]
    )
    times = np.array([-0.1, 0.0, 0.1])

    normalized = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_z",
        baseline_window=(-0.1, 0.0),
        train_idx=np.array([0, 1]),
    )

    np.testing.assert_allclose(normalized[:2].mean(axis=(0, 2), keepdims=True), 0.0)
    np.testing.assert_allclose(normalized[:2].std(axis=(0, 2), keepdims=True), 1.0)
    np.testing.assert_allclose(normalized[2], 99.0)


def test_subject_baseline_z_normalization_uses_train_baseline_only():
    data = np.array(
        [
            [[0.0, 0.0, 10.0]],
            [[2.0, 2.0, 20.0]],
            [[100.0, 100.0, 30.0]],
        ]
    )
    times = np.array([-0.2, -0.1, 0.1])

    normalized = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_baseline_z",
        baseline_window=(-0.2, -0.1),
        train_idx=np.array([0, 1]),
    )

    np.testing.assert_allclose(normalized[:2, :, :2].mean(axis=(0, 2), keepdims=True), 0.0)
    np.testing.assert_allclose(normalized[:2, :, :2].std(axis=(0, 2), keepdims=True), 1.0)
    np.testing.assert_allclose(normalized[2, :, :2], [[99.0, 99.0]])


def test_subject_baseline_whiten_fit_excludes_test_fold_outlier():
    data = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            [[2.0, 2.0, 3.0], [0.0, 0.0, 4.0]],
            [[100.0, 100.0, 5.0], [100.0, 100.0, 6.0]],
        ]
    )
    times = np.array([-0.2, -0.1, 0.1])

    fold_local = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_baseline_whiten",
        baseline_window=(-0.2, -0.1),
        train_idx=np.array([0, 1]),
    )

    contaminated = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_baseline_whiten",
        baseline_window=(-0.2, -0.1),
        train_idx=np.array([0, 1, 2]),
    )

    assert abs(float(fold_local[2, 0, 0])) > abs(float(contaminated[2, 0, 0]))
