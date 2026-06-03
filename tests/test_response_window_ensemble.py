from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.response_window_ensemble import run_response_window_ensemble


def _toy_observations() -> pd.DataFrame:
    rows = []
    times = (0.088, 0.136, 0.184, 0.232)
    for subject in ("sub-01", "sub-02", "sub-03"):
        for sample_index, true_label in enumerate((0, 1, 2, 0, 1, 2)):
            for time_index, time in enumerate(times):
                probabilities = np.full(3, 0.1)
                if time_index == 2:
                    probabilities[true_label] = 0.8
                elif subject == "sub-03" and time_index == 0:
                    probabilities[true_label] = 0.7
                else:
                    probabilities[(true_label + 1) % 3] = 0.8
                probabilities = probabilities / probabilities.sum()
                predicted_label = int(probabilities.argmax())
                rows.append(
                    {
                        "subject": subject,
                        "fold": subject,
                        "decoder": "base",
                        "emission_mode": "calibrated",
                        "time": time,
                        "test_time": time,
                        "sample_index": sample_index,
                        "sequence_id": sample_index,
                        "true_label": true_label,
                        "true_class": f"class-{true_label}",
                        "predicted_label": predicted_label,
                        "predicted_class": f"class-{predicted_label}",
                        "probability_true_class": float(probabilities[true_label]),
                        "confidence": float(probabilities.max()),
                        "class_0": "class-0",
                        "class_1": "class-1",
                        "class_2": "class-2",
                        "prob_class_0": float(probabilities[0]),
                        "prob_class_1": float(probabilities[1]),
                        "prob_class_2": float(probabilities[2]),
                    }
                )
    return pd.DataFrame(rows)


def test_response_window_uniform_logit_ensemble_writes_metrics(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    _toy_observations().to_csv(csv_path, index=False)

    out_observations = tmp_path / "response_observations.csv"
    out_metrics = tmp_path / "response_metrics.csv"
    ensembled, metrics = run_response_window_ensemble(
        [csv_path],
        mode="uniform",
        out_observations=out_observations,
        out_metrics=out_metrics,
    )

    assert out_observations.exists()
    assert out_metrics.exists()
    assert ensembled["time"].unique().tolist() == [0.184]
    assert ensembled["response_window_mode"].unique().tolist() == ["uniform"]
    assert ensembled["response_window_actual_times"].unique().tolist() == ["0.088|0.136|0.184|0.232"]
    assert metrics["decoder"].unique().tolist() == ["poststimulus_response_window_logit_ensemble"]
    assert metrics["balanced_accuracy"].between(0.0, 1.0).all()


def test_response_window_learned_weights_are_source_subject_only(tmp_path: Path):
    csv_path = tmp_path / "observations.csv"
    _toy_observations().to_csv(csv_path, index=False)

    ensembled, _ = run_response_window_ensemble(
        [csv_path],
        mode="source_oof_nonnegative",
        weight_grid_step=0.5,
    )

    weights = ensembled.groupby("subject")["response_window_weights"].first().to_dict()
    assert set(weights) == {"sub-01", "sub-02", "sub-03"}
    assert all(weight for weight in weights.values())
    assert ensembled["response_window_source_score"].replace("", np.nan).notna().all()
