import numpy as np
import pandas as pd

from neureptrace.observation_ensemble import ensemble_probability_observations


def _subject_biased_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decoder in ("logistic", "linear_svm"):
        for subject, true_label, baseline, poststimulus in (
            ("sub-01", 0, (0.90, 0.10), (0.80, 0.20)),
            ("sub-02", 1, (0.10, 0.90), (0.20, 0.80)),
        ):
            for sample_index, (time, probabilities) in enumerate(((-0.10, baseline), (0.10, poststimulus))):
                rows.append(
                    {
                        "subject": subject,
                        "fold": subject,
                        "decoder": decoder,
                        "emission_mode": "calibrated",
                        "time": time,
                        "sample_index": sample_index,
                        "sequence_id": f"{subject}-{sample_index}",
                        "true_label": true_label,
                        "true_class": "zero" if true_label == 0 else "one",
                        "class_0": "zero",
                        "class_1": "one",
                        "prob_class_0": float(probabilities[0]),
                        "prob_class_1": float(probabilities[1]),
                    }
                )
    return pd.DataFrame(rows)


def test_ensemble_baseline_group_columns_accepts_single_string() -> None:
    observations = _subject_biased_observations()

    string_grouped = ensemble_probability_observations(
        observations,
        baseline_window=(-0.20, 0.00),
        baseline_group_columns="subject",
    )
    tuple_grouped = ensemble_probability_observations(
        observations,
        baseline_window=(-0.20, 0.00),
        baseline_group_columns=("subject",),
    )

    assert string_grouped["baseline_group_columns"].unique().tolist() == ["subject"]
    np.testing.assert_allclose(
        string_grouped[["prob_class_0", "prob_class_1"]].to_numpy(dtype=float),
        tuple_grouped[["prob_class_0", "prob_class_1"]].to_numpy(dtype=float),
    )
