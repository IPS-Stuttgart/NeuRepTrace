from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.bushmeg_diagnostics import (
    build_bushmeg_diagnostics,
    infer_balanced_accuracy_chance,
    prediction_diagnostics,
    run_bushmeg_diagnostics,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "outer_test_subject": "s1",
                "candidate": "late",
                "accuracy": 0.50,
                "balanced_accuracy": 0.50,
                "top2_accuracy": 0.75,
                "log_loss": 1.2,
                "n_test_trials": 4,
                "n_classes": 4,
                "class_names": "a|b|c|d",
                "inner_mean_score": 0.44,
            },
            {
                "outer_test_subject": "s2",
                "candidate": "late",
                "accuracy": 0.25,
                "balanced_accuracy": 0.25,
                "top2_accuracy": 0.50,
                "log_loss": 1.5,
                "n_test_trials": 4,
                "n_classes": 4,
                "class_names": "a|b|c|d",
                "inner_mean_score": 0.41,
            },
            {
                "outer_test_subject": "s3",
                "candidate": "wide",
                "accuracy": 0.75,
                "balanced_accuracy": 0.75,
                "top2_accuracy": 1.00,
                "log_loss": 0.9,
                "n_test_trials": 4,
                "n_classes": 4,
                "class_names": "a|b|c|d",
                "inner_mean_score": 0.52,
            },
        ]
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_class": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "predicted_class": ["a", "b", "b", "c", "c", "d", "a", "d"],
        }
    )


def test_infer_balanced_accuracy_chance_from_class_count() -> None:
    assert infer_balanced_accuracy_chance(_summary()) == 0.25


def test_build_bushmeg_diagnostics_summarizes_subjects_candidates_and_classes() -> None:
    tables = build_bushmeg_diagnostics(_summary(), _predictions(), n_bootstrap=50, random_state=0)

    assert set(tables) == {"overall", "subjects", "candidates", "classes", "confusion"}
    overall = tables["overall"].iloc[0]
    assert overall["n_subjects"] == 3
    assert np.isclose(overall["mean_balanced_accuracy_subject"], 0.5)
    assert np.isclose(overall["mean_balanced_accuracy_excess_chance"], 0.25)
    assert overall["most_selected_candidate"] == "late"

    subjects = tables["subjects"].set_index("subject")
    assert np.isclose(subjects.loc["s1", "balanced_accuracy_normalized"], (0.50 - 0.25) / 0.75)

    candidates = tables["candidates"].set_index("candidate")
    assert candidates.loc["late", "selected_n_subjects"] == 2

    classes = tables["classes"].set_index("class_name")
    assert classes.loc["a", "support"] == 2
    assert np.isclose(classes.loc["a", "recall"], 0.5)

    confusion = tables["confusion"]
    d_as_a = confusion.loc[(confusion["true_class"] == "d") & (confusion["predicted_class"] == "a"), "count"].iloc[0]
    assert int(d_as_a) == 1


def test_prediction_diagnostics_can_use_numeric_label_columns() -> None:
    predictions = pd.DataFrame({"true_label": [0, 0, 1, 1], "predicted_label": [0, 1, 1, 1]})

    classes, confusion = prediction_diagnostics(predictions)

    assert list(classes["class_name"]) == ["0", "1"]
    assert int(confusion["count"].sum()) == 4


def test_run_bushmeg_diagnostics_writes_expected_csvs(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    predictions_path = tmp_path / "predictions.csv"
    out_dir = tmp_path / "diag"
    _summary().to_csv(summary_path, index=False)
    _predictions().to_csv(predictions_path, index=False)

    written = run_bushmeg_diagnostics(
        summary_path,
        predictions_path=predictions_path,
        out_dir=out_dir,
        n_bootstrap=10,
        random_state=0,
    )

    assert {"overall", "subjects", "candidates", "classes", "confusion"} == set(written)
    assert (out_dir / "bushmeg_diagnostics_overall.csv").exists()
    assert pd.read_csv(written["overall"]).loc[0, "n_subjects"] == 3
