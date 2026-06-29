from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_CONDITIONS = ("baseline_window", "shuffled_time", "shuffled_label")
REQUIRED_COLUMNS = {
    "decoder",
    "emission_mode",
    "condition",
    "persistence_gain_per_observation",
    "empirical_p_value",
    "best_stay_probability",
}
REQUIRED_CONDITIONS = ("observed_effect",)
PAIRED_EMISSION_MODES = ("calibrated", "uncalibrated")


def _contains_boolean_values(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return True
    return bool(values.map(lambda value: isinstance(value, (bool, np.bool_))).any())


def _coerce_finite_numeric_column(frame: pd.DataFrame, column: str) -> None:
    if _contains_boolean_values(frame[column]):
        raise ValueError(f"{column} values must be numeric, not boolean.")
    try:
        frame[column] = pd.to_numeric(frame[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} values must be numeric.") from exc
    if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
        raise ValueError(f"{column} values must be finite.")


def _validate_optional_unit_interval_column(frame: pd.DataFrame, column: str) -> None:
    present = frame[column].notna()
    if not present.any():
        return
    if _contains_boolean_values(frame.loc[present, column]):
        raise ValueError(f"{column} values must be numeric, not boolean.")
    try:
        frame.loc[present, column] = pd.to_numeric(frame.loc[present, column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} values must be numeric.") from exc
    values = frame.loc[present, column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} values must be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{column} values must be between 0 and 1.")


def _validate_temporal_summary(summary: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(summary.columns))
    if missing:
        raise ValueError(f"Temporal-model summary is missing required columns: {missing}")
    if summary.empty:
        raise ValueError("Temporal-model summary must contain at least one row.")

    validated = summary.copy()
    key_columns = ["decoder", "emission_mode", "condition"]
    if validated[key_columns].isna().any().any():
        raise ValueError("Temporal-model summary decoder, emission_mode, and condition values cannot be missing.")
    for column in key_columns:
        validated[column] = validated[column].astype(str)
        blank = validated[column].str.strip() == ""
        if blank.any():
            raise ValueError(f"Temporal-model summary {column} values cannot be blank.")

    _coerce_finite_numeric_column(validated, "persistence_gain_per_observation")
    _validate_optional_unit_interval_column(validated, "empirical_p_value")
    _validate_optional_unit_interval_column(validated, "best_stay_probability")
    shuffled_mask = validated["condition"].isin(("shuffled_time", "shuffled_label"))
    if validated.loc[shuffled_mask, "empirical_p_value"].isna().any():
        raise ValueError("empirical_p_value values must be finite for shuffled control rows.")
    observed_mask = validated["condition"] == "observed_effect"
    if validated.loc[observed_mask, "best_stay_probability"].isna().any():
        raise ValueError("best_stay_probability values must be finite for observed_effect rows.")

    duplicate_mask = validated.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        examples = validated.loc[duplicate_mask, key_columns].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Temporal-model summary contains duplicate decoder/emission/condition rows: {examples}")
    return validated


def _validate_emission_mode_frame(frame: pd.DataFrame, *, decoder: str, emission_mode: str) -> None:
    conditions = set(frame["condition"])
    missing_required = sorted(set(REQUIRED_CONDITIONS).difference(conditions))
    if missing_required:
        raise ValueError(f"Decoder '{decoder}' emission_mode '{emission_mode}' is missing required condition(s): {missing_required}")
    if not any(condition in conditions for condition in CONTROL_CONDITIONS):
        raise ValueError(f"Decoder '{decoder}' emission_mode '{emission_mode}' has no control condition rows.")


def _condition_value(frame: pd.DataFrame, condition: str, column: str) -> float:
    row = frame.loc[frame["condition"] == condition]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def _control_margin(frame: pd.DataFrame) -> float:
    observed = _condition_value(frame, "observed_effect", "persistence_gain_per_observation")
    controls = [
        _condition_value(frame, condition, "persistence_gain_per_observation")
        for condition in CONTROL_CONDITIONS
    ]
    controls = [value for value in controls if not np.isnan(value)]
    return observed - max(controls) if controls else float("nan")


def summarize_emission_mode(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize temporal-model evidence for one emission mode."""
    observed_gain = _condition_value(frame, "observed_effect", "persistence_gain_per_observation")
    baseline_gain = _condition_value(frame, "baseline_window", "persistence_gain_per_observation")
    shuffled_time_gain = _condition_value(frame, "shuffled_time", "persistence_gain_per_observation")
    shuffled_label_gain = _condition_value(frame, "shuffled_label", "persistence_gain_per_observation")
    return {
        "observed_gain": observed_gain,
        "baseline_gain": baseline_gain,
        "effect_minus_baseline_gain": observed_gain - baseline_gain,
        "shuffled_time_gain": shuffled_time_gain,
        "shuffled_label_gain": shuffled_label_gain,
        "effect_minus_shuffled_time_gain": observed_gain - shuffled_time_gain,
        "effect_minus_shuffled_label_gain": observed_gain - shuffled_label_gain,
        "control_margin": _control_margin(frame),
        "shuffled_time_p": _condition_value(frame, "shuffled_time", "empirical_p_value"),
        "shuffled_label_p": _condition_value(frame, "shuffled_label", "empirical_p_value"),
        "best_stay_probability": _condition_value(frame, "observed_effect", "best_stay_probability"),
    }


def compare_emission_modes(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare calibrated and uncalibrated temporal-model evidence by decoder."""
    summary = _validate_temporal_summary(summary)

    rows = []
    for decoder, decoder_frame in summary.groupby("decoder", sort=True):
        modes = {mode: frame for mode, frame in decoder_frame.groupby("emission_mode", sort=True)}
        if not all(mode in modes for mode in PAIRED_EMISSION_MODES):
            continue
        for emission_mode in PAIRED_EMISSION_MODES:
            _validate_emission_mode_frame(modes[emission_mode], decoder=str(decoder), emission_mode=emission_mode)
        calibrated = summarize_emission_mode(modes["calibrated"])
        uncalibrated = summarize_emission_mode(modes["uncalibrated"])
        delta_control_margin = calibrated["control_margin"] - uncalibrated["control_margin"]
        delta_effect_minus_baseline = calibrated["effect_minus_baseline_gain"] - uncalibrated["effect_minus_baseline_gain"]
        delta_observed_gain = calibrated["observed_gain"] - uncalibrated["observed_gain"]
        preferred = "calibrated" if delta_control_margin >= 0 else "uncalibrated"
        rows.append(
            {
                "decoder": decoder,
                "calibrated_observed_gain": calibrated["observed_gain"],
                "uncalibrated_observed_gain": uncalibrated["observed_gain"],
                "delta_observed_gain": delta_observed_gain,
                "calibrated_control_margin": calibrated["control_margin"],
                "uncalibrated_control_margin": uncalibrated["control_margin"],
                "delta_control_margin": delta_control_margin,
                "calibrated_effect_minus_baseline_gain": calibrated["effect_minus_baseline_gain"],
                "uncalibrated_effect_minus_baseline_gain": uncalibrated["effect_minus_baseline_gain"],
                "delta_effect_minus_baseline_gain": delta_effect_minus_baseline,
                "calibrated_shuffled_time_p": calibrated["shuffled_time_p"],
                "uncalibrated_shuffled_time_p": uncalibrated["shuffled_time_p"],
                "calibrated_shuffled_label_p": calibrated["shuffled_label_p"],
                "uncalibrated_shuffled_label_p": uncalibrated["shuffled_label_p"],
                "calibrated_best_stay_probability": calibrated["best_stay_probability"],
                "uncalibrated_best_stay_probability": uncalibrated["best_stay_probability"],
                "preferred_emission_mode": preferred,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_control_margin", ascending=False).reset_index(drop=True)


def _format_float(value: float, digits: int = 4) -> str:
    return "nan" if pd.isna(value) else f"{value:.{digits}f}"


def build_emission_comparison_report(comparison: pd.DataFrame, *, summary_csv: Path) -> str:
    """Build a compact Markdown report for calibrated-vs-uncalibrated emissions."""
    lines = [
        "# NeuRepTrace Emission Calibration Comparison",
        "",
        f"- Temporal-model summary: `{summary_csv}`",
        "",
        "Question: do calibrated probabilities produce cleaner state inference than",
        "uncalibrated score-derived emissions?",
        "",
        "The main comparison is the control margin: observed effect-window",
        "persistence gain minus the strongest baseline, shuffled-time, or",
        "shuffled-label control gain. Positive deltas favor calibrated emissions.",
        "",
    ]
    if comparison.empty:
        lines.append("No decoder had both calibrated and uncalibrated emission-mode rows.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Decoder | Preferred | Delta control margin | Calibrated margin | Uncalibrated margin | Delta effect-baseline | Calibrated p(time) | Uncalibrated p(time) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.decoder} | {row.preferred_emission_mode} | {_format_float(row.delta_control_margin)} | "
            f"{_format_float(row.calibrated_control_margin)} | {_format_float(row.uncalibrated_control_margin)} | "
            f"{_format_float(row.delta_effect_minus_baseline_gain)} | {_format_float(row.calibrated_shuffled_time_p)} | "
            f"{_format_float(row.uncalibrated_shuffled_time_p)} |"
        )
    lines.append("")
    return "\n".join(lines)


def compare_temporal_summary(
    summary_csv: Path,
    *,
    out_csv: Path | None = None,
    out_report: Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Compare calibrated and uncalibrated emission rows from a temporal-model summary CSV."""
    comparison = compare_emission_modes(pd.read_csv(summary_csv))
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_csv, index=False)
    report = None
    if out_report is not None:
        report = build_emission_comparison_report(comparison, summary_csv=summary_csv)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(report, encoding="utf-8")
    return comparison, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare calibrated versus uncalibrated emissions in NeuRepTrace temporal-model summaries."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path)
    args = parser.parse_args()

    comparison, _ = compare_temporal_summary(
        args.summary_csv,
        out_csv=args.out_csv,
        out_report=args.out_report,
    )
    print(f"Wrote emission comparison: {args.out_csv}")
    if args.out_report is not None:
        print(f"Wrote emission comparison report: {args.out_report}")
    print(comparison.to_string(index=False) if not comparison.empty else "No paired emission modes found.")


if __name__ == "__main__":
    main()
