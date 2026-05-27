from __future__ import annotations

import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def _valid_canonical_observations() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-001", "trial-001"],
            "time": [-0.1, -0.08],
            "decoder": ["logistic", "logistic"],
            "backend": ["sklearn", "sklearn"],
            "emission_mode": ["calibrated", "calibrated"],
            "session": ["run-01", "run-01"],
            "fold": [0, 0],
            "split_id": ["split-001", "split-001"],
            "seed": [13, 13],
            "train_time": [-0.1, -0.08],
            "test_time": [-0.1, -0.08],
            "true_label": [1, 1],
            "predicted_label": [0, 1],
            "true_class": ["face", "face"],
            "predicted_class": ["object", "face"],
            "probability_true_class": [0.4, 0.55],
            "confidence": [0.6, 0.55],
            "prob_class_0": [0.6, 0.45],
            "prob_class_1": [0.4, 0.55],
            "class_0": ["object", "object"],
            "class_1": ["face", "face"],
            "calibration_fold": ["", ""],
            "preprocessing_hash": ["pre-123", "pre-123"],
            "model_hash": ["model-456", "model-456"],
        }
    )
    return frame.astype({"true_label": object, "predicted_label": object})


def test_canonical_profile_rejects_fractional_predicted_labels() -> None:
    frame = _valid_canonical_observations()
    frame.loc[0, "predicted_label"] = 0.5

    report = validate_probability_observations(frame, profile="canonical")

    assert not report.is_valid
    assert any(issue.code == "non_integer_label" and issue.column == "predicted_label" for issue in report.errors)
    assert not any(issue.code == "predicted_class_mismatch" for issue in report.errors)


def test_canonical_profile_rejects_non_finite_true_labels_without_crashing() -> None:
    frame = _valid_canonical_observations()
    frame.loc[0, "true_label"] = float("inf")

    report = validate_probability_observations(frame, profile="canonical")

    assert not report.is_valid
    assert any(issue.code == "non_finite_label" and issue.column == "true_label" for issue in report.errors)


def test_canonical_profile_reports_missing_true_label_probability_column() -> None:
    frame = _valid_canonical_observations()
    frame.loc[0, "true_label"] = 7
    frame.loc[0, "probability_true_class"] = 0.1

    report = validate_probability_observations(frame, profile="canonical")

    assert not report.is_valid
    assert any(issue.code == "missing_true_label_probability_column" and issue.column == "true_label" for issue in report.errors)
