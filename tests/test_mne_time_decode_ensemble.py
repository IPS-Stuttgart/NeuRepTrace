from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.mne_time_decode_ensemble import (
    ENSEMBLE_DECODER,
    ENSEMBLE_DECODER_CLI_CHOICES,
    normalize_time_decode_decoder_name,
    run_time_resolved_decode,
)


def _source_observations(decoder: str) -> pd.DataFrame:
    if decoder == "logistic":
        probabilities = [(0.85, 0.15), (0.20, 0.80), (0.70, 0.30), (0.35, 0.65)]
    elif decoder == "linear_svm":
        probabilities = [(0.75, 0.25), (0.30, 0.70), (0.60, 0.40), (0.25, 0.75)]
    else:
        raise AssertionError(f"unexpected decoder {decoder!r}")

    rows = []
    for sample_index, (prob_0, prob_1) in enumerate(probabilities):
        true_label = sample_index % 2
        rows.append(
            {
                "subject": "sub-01",
                "fold": 0,
                "split_id": "stratified-kfold-2",
                "seed": 13,
                "decoder": decoder,
                "backend": "sklearn",
                "emission_mode": "calibrated",
                "feature_preprocessor": "pca_whiten",
                "pca_components": 0.95,
                "temporal_mode": "same_time",
                "train_time": 0.1,
                "test_time": 0.1,
                "time": 0.1,
                "window_start": 0.08,
                "window_stop": 0.12,
                "sample_index": sample_index,
                "sequence_id": sample_index,
                "true_label": true_label,
                "true_class": "A" if true_label == 0 else "B",
                "predicted_label": int(prob_1 > prob_0),
                "predicted_class": "A" if prob_0 >= prob_1 else "B",
                "probability_true_class": prob_0 if true_label == 0 else prob_1,
                "confidence": max(prob_0, prob_1),
                "is_correct": True,
                "class_0": "A",
                "class_1": "B",
                "prob_class_0": prob_0,
                "prob_class_1": prob_1,
                "preprocessing_hash": "pre",
                "model_hash": f"model-{decoder}",
            }
        )
    return pd.DataFrame(rows)


def test_logistic_svm_ensemble_aliases_are_exposed():
    assert "logistic-svm-ensemble" in ENSEMBLE_DECODER_CLI_CHOICES
    assert normalize_time_decode_decoder_name("calibrated-logistic-linear-svm-ensemble") == ENSEMBLE_DECODER


def test_logistic_svm_ensemble_runs_as_first_class_decoder(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_time_resolved_decode(*, decoder: str, out_path: Path, observation_out_path: Path, **_kwargs):
        calls.append(decoder)
        _source_observations(decoder).to_csv(observation_out_path, index=False)
        metrics = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "decoder": decoder,
                    "emission_mode": "calibrated",
                    "time": 0.1,
                    "window_start": 0.08,
                    "window_stop": 0.12,
                    "accuracy": 1.0,
                    "log_loss": 0.1,
                    "brier": 0.05,
                    "ece": 0.01,
                    "n_train": 4,
                    "n_test": 4,
                    "n_classes": 2,
                }
            ]
        )
        metrics.to_csv(out_path, index=False)
        return metrics

    monkeypatch.setattr("neureptrace.mne_time_decode_ensemble._run_time_resolved_decode", fake_run_time_resolved_decode)

    out_path = tmp_path / "ensemble_metrics.csv"
    observations_out = tmp_path / "ensemble_observations.csv"
    results = run_time_resolved_decode(
        epochs_path=tmp_path / "dummy-epo.fif",
        metadata_csv=tmp_path / "dummy-events.csv",
        label_column="condition",
        group_column="session",
        out_path=out_path,
        decoder="logistic-svm-ensemble",
        emission_mode="calibrated",
        feature_preprocessor="pca-whiten",
        pca_components="0.95",
        observation_out_path=observations_out,
        ensemble_baseline_window=None,
    )

    assert calls == ["logistic", "linear_svm"]
    assert out_path.exists()
    assert observations_out.exists()
    assert results["decoder"].unique().tolist() == [ENSEMBLE_DECODER]
    assert results["emission_mode"].unique().tolist() == ["baseline_debiased_calibrated_ensemble"]

    observations = pd.read_csv(observations_out)
    assert observations["decoder"].unique().tolist() == [ENSEMBLE_DECODER]
    probability_sums = observations[["prob_class_0", "prob_class_1"]].sum(axis=1).to_numpy()
    assert np.allclose(probability_sums, 1.0)


def test_logistic_svm_ensemble_requires_calibrated_emissions(tmp_path):
    with pytest.raises(ValueError, match="calibrated only"):
        run_time_resolved_decode(
            epochs_path=tmp_path / "dummy-epo.fif",
            label_column="condition",
            out_path=tmp_path / "out.csv",
            decoder="logistic-svm-ensemble",
            emission_mode="uncalibrated",
        )
