from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.bushmeg_loso_decode import (
    CachedSubject,
    _average_probabilities,
    _features_for_window,
    make_source_pseudotrials,
    normalize_window_feature_mode,
    run_bushmeg_loso_decode,
)


def test_make_source_pseudotrials_balances_classes():
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.array([0] * 6 + [1] * 6)
    rng = np.random.default_rng(13)

    pseudo_features, pseudo_labels = make_source_pseudotrials(
        features,
        labels,
        classes=np.array([0, 1]),
        pseudotrials_per_class=3,
        rng=rng,
    )

    assert pseudo_features.shape == (6, 2)
    assert pd.Series(pseudo_labels).value_counts().sort_index().to_dict() == {0: 3, 1: 3}


def test_normalize_window_feature_mode_accepts_aliases():
    assert normalize_window_feature_mode("flat") == "sensor_flat"
    assert normalize_window_feature_mode("evoked-bin-means") == "bin_means"
    assert normalize_window_feature_mode("evoked-slope") == "mean_slope"
    assert normalize_window_feature_mode("temporal-dct") == "dct"
    assert normalize_window_feature_mode("evoked-stats") == "stats"


def test_normalize_window_feature_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown window feature mode"):
        normalize_window_feature_mode("definitely_not_a_feature")


def test_average_probabilities_rejects_invalid_inputs():
    first = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=float)
    second = np.array([[0.2, 0.2], [0.4, 0.6]], dtype=float)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _average_probabilities([first, second], mode="log")


def test_average_probabilities_rejects_values_above_one():
    first = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=float)
    second = np.array([[1.2, 0.0], [0.4, 0.6]], dtype=float)

    with pytest.raises(ValueError, match="must not exceed 1.0"):
        _average_probabilities([first, second], mode="mean")


def test_compact_window_feature_modes_have_expected_shapes():
    data = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
    times = np.linspace(0.0, 0.05, 6)
    window = (1, 5, 0.025)

    flat = _features_for_window(data, window)
    np.testing.assert_array_equal(flat, data[:, :, 1:5].reshape(2, -1))
    assert flat.shape == (2, 12)

    bin_means = _features_for_window(data, window, window_feature_mode="bin-means", temporal_bins=2)
    expected_bin_means = np.concatenate(
        [
            data[:, :, [1, 2]].mean(axis=2),
            data[:, :, [3, 4]].mean(axis=2),
        ],
        axis=1,
    )
    np.testing.assert_allclose(bin_means, expected_bin_means)
    assert bin_means.shape == (2, 6)

    mean_slope = _features_for_window(data, window, times=times, window_feature_mode="mean-slope", temporal_bins=2)
    assert mean_slope.shape == (2, 12)

    dct_features = _features_for_window(data, window, window_feature_mode="dct", temporal_bins=2)
    assert dct_features.shape == (2, 6)

    stats = _features_for_window(data, window, times=times, window_feature_mode="stats", temporal_bins=2)
    assert stats.shape == (2, 30)

    flat_plus_stats = _features_for_window(data, window, times=times, window_feature_mode="sensor-flat-plus-stats", temporal_bins=2)
    assert flat_plus_stats.shape == (2, 42)


def test_compact_window_feature_modes_reject_too_many_temporal_bins():
    data = np.zeros((2, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="not enough for 3 temporal bins"):
        _features_for_window(data, (0, 2, 0.005), window_feature_mode="bin-means", temporal_bins=3)

    with pytest.raises(ValueError, match="not enough for 3 DCT coefficients"):
        _features_for_window(data, (0, 2, 0.005), window_feature_mode="dct", temporal_bins=3)


def test_run_bushmeg_loso_decode_uses_source_only_cached_subjects(tmp_path: Path, monkeypatch):
    times = np.array([-0.02, -0.01, 0.00, 0.01, 0.02], dtype=float)
    labels = np.array([0, 1, 0, 1, 0, 1])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached: dict[str, CachedSubject] = {}
    for participant, offset in [("1", -0.1), ("2", 0.0), ("3", 0.1)]:
        data = np.zeros((6, 2, len(times)), dtype=np.float32)
        data[:, :, :] = offset
        data[labels == 1, 0, 2:] += 2.0
        data[labels == 0, 1, 2:] += 2.0
        data_path = cache_dir / f"sub-{participant}.data.npy"
        labels_path = cache_dir / f"sub-{participant}.labels.npy"
        times_path = cache_dir / f"sub-{participant}.times.npy"
        np.save(data_path, data)
        np.save(labels_path, labels)
        np.save(times_path, times)
        cached[participant] = CachedSubject(
            participant=participant,
            source_path=tmp_path / f"Part{participant}Data.mat",
            data_path=data_path,
            labels_path=labels_path,
            times_path=times_path,
        )

    def fake_prepare_subject_cache(*, participant: str, **_kwargs):
        return cached[participant]

    monkeypatch.setattr("neureptrace.bushmeg_loso_decode._prepare_subject_cache", fake_prepare_subject_cache)

    out = tmp_path / "bushmeg_loso.csv"
    results = run_bushmeg_loso_decode(
        data_dir=tmp_path,
        out_path=out,
        participants="1-3",
        cache_dir=cache_dir,
        tmin=None,
        tmax=None,
        window_ms=20.0,
        step_ms=20.0,
        decode_window=(0.0, 0.02),
        decoders=("correlation-prototype",),
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        normalization="none",
        pseudotrials_per_class=2,
        ensemble_mode="mean",
        max_iter=200,
        resume=False,
    )

    ensemble = results[results["analysis"] == "temporal_ensemble"]
    assert out.exists()
    assert ensemble["heldout_subject"].astype(str).tolist() == ["1", "2", "3"]
    assert ensemble["n_source_subjects"].tolist() == [2, 2, 2]
    assert ensemble["balanced_accuracy"].min() > 0.9
