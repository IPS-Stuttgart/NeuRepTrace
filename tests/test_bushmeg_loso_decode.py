from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.bushmeg_loso_decode import CachedSubject, make_source_pseudotrials, run_bushmeg_loso_decode


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
