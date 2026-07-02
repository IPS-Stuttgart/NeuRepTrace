import numpy as np
import pandas as pd

from neureptrace.probability_stacking import stack_probability_observations, summarize_stacked_metrics


def _rows(*, subject: str, labels: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decoder in ("weak", "strong"):
        for sample_index, true_label in enumerate(labels):
            if decoder == "strong":
                probabilities = (0.90, 0.10) if true_label == 0 else (0.10, 0.90)
            else:
                probabilities = (0.55, 0.45) if true_label == 0 else (0.45, 0.55)
            rows.append(
                {
                    "subject": subject,
                    "fold": sample_index % 2,
                    "decoder": decoder,
                    "emission_mode": "calibrated",
                    "time": 0.10,
                    "window_start": 0.05,
                    "window_stop": 0.15,
                    "sample_index": sample_index,
                    "true_label": int(true_label),
                    "true_class": "zero" if true_label == 0 else "one",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": float(probabilities[0]),
                    "prob_class_1": float(probabilities[1]),
                }
            )
    return pd.DataFrame(rows)


def test_summary_preserves_source_oof_temperature_and_alignment_metadata() -> None:
    source = _rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _rows(subject="target", labels=[0, 1, 0])

    stacked = stack_probability_observations(
        source,
        target,
        alignment_columns=["subject", "sample_index"],
        weighting="softmax",
        temperature=0.5,
    )
    metrics = summarize_stacked_metrics(stacked)

    assert {"source_oof_temperature", "source_oof_alignment_columns"}.issubset(metrics.columns)
    assert metrics["source_oof_temperature"].tolist() == [0.5] * len(metrics)
    assert metrics["source_oof_alignment_columns"].tolist() == ["subject|sample_index"] * len(metrics)
    assert np.all(np.isfinite(metrics["balanced_accuracy"]))
