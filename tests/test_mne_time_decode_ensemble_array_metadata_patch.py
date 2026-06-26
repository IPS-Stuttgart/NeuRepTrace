from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.mne_time_decode_ensemble import run_time_resolved_decode


def _write_calibrated_source_decode_outputs(kwargs):
    decoder = kwargs["decoder"]
    rows = [
        {
            "subject": "sub-01",
            "fold": 0,
            "decoder": decoder,
            "emission_mode": "calibrated",
            "time": 0.184,
            "window_start": 0.134,
            "window_stop": 0.234,
            "sample_index": index,
            "sequence_id": index,
            "true_label": label,
            "true_class": f"class-{label}",
            "predicted_label": label,
            "predicted_class": f"class-{label}",
            "probability_true_class": 0.8,
            "confidence": 0.8,
            "class_0": "class-0",
            "class_1": "class-1",
            "prob_class_0": 0.8 if label == 0 else 0.2,
            "prob_class_1": 0.2 if label == 0 else 0.8,
        }
        for index, label in enumerate((0, 1))
    ]
    kwargs["observation_out_path"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(kwargs["observation_out_path"], index=False)

    frame = pd.DataFrame(
        [
            {
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.184,
                "window_start": 0.134,
                "window_stop": 0.234,
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": 1.0,
                "log_loss": 0.2,
                "brier": 0.1,
                "ece": 0.0,
                "n_test": 2,
            }
        ]
    )
    kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(kwargs["out_path"], index=False)
    return frame


def test_logistic_svm_ensemble_accepts_numpy_time_metadata_arrays(tmp_path, monkeypatch):
    calls = []

    def fake_source_decode(**kwargs):
        calls.append(kwargs)
        return _write_calibrated_source_decode_outputs(kwargs)

    monkeypatch.setattr("neureptrace.mne_time_decode_ensemble._run_time_resolved_decode", fake_source_decode)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "dummy-epo.fif",
        label_column="condition",
        out_path=tmp_path / "ensemble.csv",
        decoder="logistic-svm-ensemble",
        emission_mode="calibrated",
        source_time_selection_times=np.array([0.1, 0.2]),
        alignment_times=np.array([0.3, 0.4]),
        ensemble_baseline_window=None,
    )

    assert len(calls) == 2
    assert all(call["source_time_selection_times"] == (0.1, 0.2) for call in calls)
    assert all(call["alignment_times"] == (0.3, 0.4) for call in calls)
    assert results["source_time_selection_times"].unique().tolist() == ["0.1,0.2"]
    assert results["alignment_times"].unique().tolist() == ["0.3,0.4"]
