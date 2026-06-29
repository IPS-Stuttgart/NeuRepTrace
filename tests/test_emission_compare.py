from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.emission_compare import compare_emission_modes, compare_temporal_summary


def _temporal_summary() -> pd.DataFrame:
    rows = []
    for emission_mode, observed, baseline, shuffled_time, shuffled_label in [
        ("calibrated", 0.12, 0.02, 0.04, 0.03),
        ("uncalibrated", 0.10, 0.05, 0.07, 0.06),
    ]:
        for condition, gain, p_value in [
            ("observed_effect", observed, None),
            ("baseline_window", baseline, None),
            ("shuffled_time", shuffled_time, 0.02 if emission_mode == "calibrated" else 0.12),
            ("shuffled_label", shuffled_label, 0.04 if emission_mode == "calibrated" else 0.16),
        ]:
            rows.append(
                {
                    "decoder": "linear_svm",
                    "emission_mode": emission_mode,
                    "condition": condition,
                    "n_sequences": 10,
                    "n_observations": 100,
                    "n_states": 2,
                    "best_stay_probability": 0.9,
                    "persistence_gain_per_observation": gain,
                    "empirical_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def test_compare_emission_modes_reports_control_margin_delta():
    comparison = compare_emission_modes(_temporal_summary())

    row = comparison.iloc[0]
    assert row["decoder"] == "linear_svm"
    assert round(row["calibrated_control_margin"], 3) == 0.08
    assert round(row["uncalibrated_control_margin"], 3) == 0.03
    assert round(row["delta_control_margin"], 3) == 0.05
    assert row["preferred_emission_mode"] == "calibrated"


def test_compare_temporal_summary_writes_csv_and_report(tmp_path: Path):
    summary_csv = tmp_path / "temporal_model.csv"
    out_csv = tmp_path / "emission_compare.csv"
    out_report = tmp_path / "emission_compare.md"
    _temporal_summary().to_csv(summary_csv, index=False)

    comparison, report = compare_temporal_summary(summary_csv, out_csv=out_csv, out_report=out_report)

    assert out_csv.exists()
    assert out_report.exists()
    assert report is not None and "calibrated probabilities produce cleaner" in report
    assert comparison["preferred_emission_mode"].tolist() == ["calibrated"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns=["empirical_p_value"]), "missing required columns"),
        (lambda frame: frame.iloc[0:0], "at least one row"),
        (lambda frame: frame.assign(decoder=[None, *frame["decoder"].iloc[1:].tolist()]), "cannot be missing"),
        (lambda frame: frame.assign(emission_mode=["", *frame["emission_mode"].iloc[1:].tolist()]), "emission_mode values cannot be blank"),
        (lambda frame: frame.assign(condition=["", *frame["condition"].iloc[1:].tolist()]), "condition values cannot be blank"),
    ],
)
def test_compare_emission_modes_rejects_malformed_structure(mutate, message: str):
    with pytest.raises(ValueError, match=message):
        compare_emission_modes(mutate(_temporal_summary()))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("persistence_gain_per_observation", "high", "persistence_gain_per_observation values must be numeric"),
        ("persistence_gain_per_observation", True, "persistence_gain_per_observation values must be numeric"),
        ("persistence_gain_per_observation", np.inf, "persistence_gain_per_observation values must be finite"),
        ("empirical_p_value", "small", "empirical_p_value values must be numeric"),
        ("empirical_p_value", np.bool_(True), "empirical_p_value values must be numeric"),
        ("empirical_p_value", np.nan, "empirical_p_value values must be finite"),
        ("empirical_p_value", 1.2, "empirical_p_value values must be between 0 and 1"),
        ("best_stay_probability", "sticky", "best_stay_probability values must be numeric"),
        ("best_stay_probability", False, "best_stay_probability values must be numeric"),
        ("best_stay_probability", np.inf, "best_stay_probability values must be finite"),
        ("best_stay_probability", -0.1, "best_stay_probability values must be between 0 and 1"),
    ],
)
def test_compare_emission_modes_rejects_malformed_numeric_values(column: str, value, message: str):
    summary = _temporal_summary()
    if isinstance(value, (str, bool, np.bool_)):
        summary[column] = summary[column].astype(object)
    row = summary.index[summary["condition"] == "shuffled_time"][0] if column == "empirical_p_value" else 0
    summary.loc[row, column] = value

    with pytest.raises(ValueError, match=message):
        compare_emission_modes(summary)


def test_compare_emission_modes_rejects_duplicate_condition_rows():
    summary = pd.concat([_temporal_summary(), _temporal_summary().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate decoder/emission/condition rows"):
        compare_emission_modes(summary)


@pytest.mark.parametrize(
    ("drop_condition", "message"),
    [
        ("observed_effect", "missing required condition"),
        ("baseline_window", "has no control condition rows"),
    ],
)
def test_compare_emission_modes_rejects_missing_required_evidence(drop_condition: str, message: str):
    summary = _temporal_summary()
    mask = ~((summary["emission_mode"] == "calibrated") & (summary["condition"] == drop_condition))
    if drop_condition == "baseline_window":
        mask &= ~((summary["emission_mode"] == "calibrated") & summary["condition"].isin(["shuffled_time", "shuffled_label"]))
    summary = summary.loc[mask].reset_index(drop=True)

    with pytest.raises(ValueError, match=message):
        compare_emission_modes(summary)
