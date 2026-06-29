from __future__ import annotations

import argparse
import glob
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

CALIBRATION_METRICS = ("log_loss", "brier", "ece")
GROUP_COLUMNS = ("decoder", "emission_mode")
SUMMARY_REQUIRED_COLUMNS = (
    "time",
    "accuracy_mean",
    "log_loss_mean",
    "brier_mean",
    "ece_mean",
    "n_subjects",
)
SUMMARY_NUMERIC_COLUMNS = SUMMARY_REQUIRED_COLUMNS
SUMMARY_UNIT_INTERVAL_COLUMNS = ("accuracy_mean", "ece_mean")
RELIABILITY_BIN_REQUIRED_COLUMNS = (
    "time",
    "bin",
    "bin_left",
    "bin_right",
    "n_samples",
    "accuracy",
    "confidence",
)
RELIABILITY_BIN_NUMERIC_COLUMNS = RELIABILITY_BIN_REQUIRED_COLUMNS
RELIABILITY_BIN_INTEGER_COLUMNS = ("bin", "n_samples")
RELIABILITY_BIN_UNIT_INTERVAL_COLUMNS = ("bin_left", "bin_right", "accuracy", "confidence")
RELIABILITY_BIN_OPTIONAL_EMPTY_COLUMNS = ("accuracy", "confidence")
RELIABILITY_BIN_WEIGHT_COLUMN = "sample_weight"


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _is_boolean_like_numeric(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        try:
            scalar = value.item()
        except ValueError:
            return False
        return isinstance(scalar, (bool, np.bool_))
    return False


def _validate_time_window(window: Sequence[object], *, name: str) -> tuple[float, float]:
    if isinstance(window, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two finite numeric endpoints.")
    try:
        values = tuple(window)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two finite numeric endpoints.") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two finite numeric endpoints.")

    endpoints: list[float] = []
    for value in values:
        if _is_boolean_like_numeric(value):
            raise ValueError(f"{name} endpoints must be finite numeric values, not booleans.")
        try:
            endpoint = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} endpoints must be finite numeric values.") from exc
        if not np.isfinite(endpoint):
            raise ValueError(f"{name} endpoints must be finite numeric values.")
        endpoints.append(endpoint)

    start, stop = endpoints
    if stop < start:
        raise ValueError(f"{name} stop must be greater than or equal to start.")
    return start, stop


def _window_mean(frame: pd.DataFrame, column: str, start: float, stop: float) -> float:
    window = frame[(frame["time"] >= start) & (frame["time"] <= stop)]
    if window.empty:
        raise ValueError(f"No time points found in window [{start}, {stop}].")
    return float(window[column].mean())


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _reject_boolean_numeric_values(values: pd.Series, column: str, *, source: str) -> None:
    boolean_values = values.map(_is_boolean_like_numeric).fillna(False).astype(bool)
    if boolean_values.any():
        bad_rows = boolean_values[boolean_values].index.tolist()[:5]
        raise ValueError(f"{source} contains boolean values in numeric column '{column}' at row(s) {bad_rows}.")


def _present_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_COLUMNS if column in frame.columns]


def _validate_calibration_summary(summary: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SUMMARY_REQUIRED_COLUMNS).difference(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    validated = summary.copy()
    for column in SUMMARY_NUMERIC_COLUMNS:
        _reject_boolean_numeric_values(validated[column], column, source="Summary")
        values = pd.to_numeric(validated[column], errors="coerce")
        if values.isna().any():
            bad_rows = values[values.isna()].index.tolist()[:5]
            raise ValueError(f"Summary contains non-numeric or missing values in column '{column}' at row(s) {bad_rows}.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Summary contains non-finite values in column '{column}'.")
        validated[column] = values

    for column in SUMMARY_UNIT_INTERVAL_COLUMNS:
        outside = (validated[column] < 0.0) | (validated[column] > 1.0)
        if outside.any():
            bad_rows = outside[outside].index.tolist()[:5]
            raise ValueError(f"Summary contains values outside [0, 1] in column '{column}' at row(s) {bad_rows}.")

    negative_log_loss = validated["log_loss_mean"] < 0.0
    if negative_log_loss.any():
        bad_rows = negative_log_loss[negative_log_loss].index.tolist()[:5]
        raise ValueError(f"Summary contains negative log_loss_mean at row(s) {bad_rows}.")

    fractional_subjects = validated["n_subjects"] % 1 != 0
    if fractional_subjects.any():
        bad_rows = validated.index[fractional_subjects].tolist()[:5]
        raise ValueError(f"Summary contains non-integer n_subjects at row(s) {bad_rows}.")

    non_positive_subjects = validated["n_subjects"] <= 0
    if non_positive_subjects.any():
        bad_rows = validated.index[non_positive_subjects].tolist()[:5]
        raise ValueError(f"Summary contains non-positive n_subjects at row(s) {bad_rows}.")
    validated["n_subjects"] = validated["n_subjects"].astype(int)

    return validated


def summarize_calibration_metrics(
    summary: pd.DataFrame,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> pd.DataFrame:
    """Summarize accuracy and calibration metrics over benchmark time windows."""
    baseline_window = _validate_time_window(baseline_window, name="baseline_window")
    effect_window = _validate_time_window(effect_window, name="effect_window")
    summary = _validate_calibration_summary(summary)

    group_columns = _present_group_columns(summary)
    group_items = summary.groupby(group_columns, sort=True) if group_columns else [("overall", summary)]
    rows = []
    for keys, frame in group_items:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_values, strict=True)) if group_columns else {}
        group_values.setdefault("decoder", "overall")
        effect = frame[(frame["time"] >= effect_window[0]) & (frame["time"] <= effect_window[1])]
        if effect.empty:
            raise ValueError(f"No time points found in effect window [{effect_window[0]}, {effect_window[1]}].")
        best_ece = effect.loc[effect["ece_mean"].idxmin()]
        rows.append(
            {
                **group_values,
                "n_subjects": int(frame["n_subjects"].max()),
                "baseline_accuracy_mean": _window_mean(frame, "accuracy_mean", *baseline_window),
                "effect_accuracy_mean": _window_mean(frame, "accuracy_mean", *effect_window),
                "effect_log_loss_mean": _window_mean(frame, "log_loss_mean", *effect_window),
                "effect_brier_mean": _window_mean(frame, "brier_mean", *effect_window),
                "effect_ece_mean": _window_mean(frame, "ece_mean", *effect_window),
                "best_ece_time": float(best_ece["time"]),
                "best_ece": float(best_ece["ece_mean"]),
                "accuracy_at_best_ece": float(best_ece["accuracy_mean"]),
                "brier_at_best_ece": float(best_ece["brier_mean"]),
                "log_loss_at_best_ece": float(best_ece["log_loss_mean"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["effect_ece_mean", "effect_brier_mean", "effect_log_loss_mean"]).reset_index(drop=True)


def _validate_reliability_bins(frame: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    missing = sorted(set(RELIABILITY_BIN_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    validated = frame.copy()
    for column in RELIABILITY_BIN_NUMERIC_COLUMNS:
        _reject_boolean_numeric_values(validated[column], column, source=str(csv_path))
        values = pd.to_numeric(validated[column], errors="coerce")
        missing_values = values.isna()
        allowed_missing = pd.Series(False, index=values.index)
        if column in RELIABILITY_BIN_OPTIONAL_EMPTY_COLUMNS:
            allowed_missing = missing_values & validated["n_samples"].eq(0)
        invalid_missing = missing_values & ~allowed_missing
        if invalid_missing.any():
            bad_rows = invalid_missing[invalid_missing].index.tolist()[:5]
            if column in RELIABILITY_BIN_OPTIONAL_EMPTY_COLUMNS:
                raise ValueError(
                    f"{csv_path} contains missing values in column '{column}' for non-empty reliability bin(s); "
                    f"missing or non-finite values in column '{column}' are invalid for row(s) with positive "
                    f"n_samples: {bad_rows}."
                )
            raise ValueError(f"{csv_path} contains non-numeric values in column '{column}' at row(s) {bad_rows}.")
        finite_values = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)
        invalid_non_finite = ~finite_values & ~allowed_missing
        if invalid_non_finite.any():
            bad_rows = invalid_non_finite[invalid_non_finite].index.tolist()[:5]
            raise ValueError(f"{csv_path} contains non-finite values in column '{column}' at row(s) {bad_rows}.")
        validated[column] = values

    for column in RELIABILITY_BIN_INTEGER_COLUMNS:
        fractional = validated[column] % 1 != 0
        if fractional.any():
            bad_rows = validated.index[fractional].tolist()[:5]
            raise ValueError(f"{csv_path} contains non-integer {column} at row(s) {bad_rows}.")
        negative = validated[column] < 0
        if negative.any():
            bad_rows = validated.index[negative].tolist()[:5]
            raise ValueError(f"{csv_path} contains negative {column} at row(s) {bad_rows}.")
        validated[column] = validated[column].astype(int)

    non_empty_rows = validated["n_samples"] > 0
    for column in RELIABILITY_BIN_OPTIONAL_EMPTY_COLUMNS:
        missing_or_non_finite = ~np.isfinite(validated[column].to_numpy(dtype=float))
        invalid = missing_or_non_finite & non_empty_rows.to_numpy(dtype=bool)
        if invalid.any():
            bad_rows = validated.index[invalid].tolist()[:5]
            raise ValueError(
                f"{csv_path} contains missing values in column '{column}' for non-empty reliability bin(s); "
                f"missing or non-finite values in column '{column}' are invalid for row(s) with positive "
                f"n_samples: {bad_rows}."
            )

    for column in RELIABILITY_BIN_UNIT_INTERVAL_COLUMNS:
        present = validated[column].notna()
        outside = present & ((validated[column] < 0.0) | (validated[column] > 1.0))
        if outside.any():
            bad_rows = outside[outside].index.tolist()[:5]
            raise ValueError(f"{csv_path} contains values outside [0, 1] in column '{column}' at row(s) {bad_rows}.")

    if RELIABILITY_BIN_WEIGHT_COLUMN in validated.columns:
        _reject_boolean_numeric_values(validated[RELIABILITY_BIN_WEIGHT_COLUMN], RELIABILITY_BIN_WEIGHT_COLUMN, source=str(csv_path))
        sample_weight = pd.to_numeric(validated[RELIABILITY_BIN_WEIGHT_COLUMN], errors="coerce")
        missing_weight = sample_weight.isna()
        if missing_weight.any():
            bad_rows = missing_weight[missing_weight].index.tolist()[:5]
            raise ValueError(f"{csv_path} contains non-numeric values in column '{RELIABILITY_BIN_WEIGHT_COLUMN}' at row(s) {bad_rows}.")
        finite_weight = pd.Series(np.isfinite(sample_weight.to_numpy(dtype=float)), index=sample_weight.index)
        if not finite_weight.all():
            bad_rows = finite_weight[~finite_weight].index.tolist()[:5]
            raise ValueError(f"{csv_path} contains non-finite values in column '{RELIABILITY_BIN_WEIGHT_COLUMN}' at row(s) {bad_rows}.")
        negative_weight = sample_weight < 0.0
        if negative_weight.any():
            bad_rows = negative_weight[negative_weight].index.tolist()[:5]
            raise ValueError(f"{csv_path} contains negative {RELIABILITY_BIN_WEIGHT_COLUMN} at row(s) {bad_rows}.")
        validated[RELIABILITY_BIN_WEIGHT_COLUMN] = sample_weight

    if (validated["bin_right"] < validated["bin_left"]).any():
        bad_rows = validated.index[validated["bin_right"] < validated["bin_left"]].tolist()[:5]
        raise ValueError(f"{csv_path} contains reliability bins with bin_right < bin_left at row(s) {bad_rows}.")

    return validated


def aggregate_reliability_bins(csv_paths: list[Path]) -> pd.DataFrame:
    """Aggregate reliability-bin CSVs emitted by ``neureptrace.mne_time_decode``."""
    if not csv_paths:
        raise ValueError("At least one calibration-bin CSV path is required.")

    frames = []
    for csv_path in csv_paths:
        frame = _validate_reliability_bins(pd.read_csv(csv_path), csv_path)
        if "decoder" not in frame.columns:
            frame["decoder"] = "overall"
        if "emission_mode" not in frame.columns:
            frame["emission_mode"] = "calibrated"
        frame["source_file"] = csv_path.name
        frames.append(frame)

    bins = pd.concat(frames, ignore_index=True)
    has_sample_weight = RELIABILITY_BIN_WEIGHT_COLUMN in bins.columns
    if has_sample_weight:
        missing_weight = bins[RELIABILITY_BIN_WEIGHT_COLUMN].isna()
        bins.loc[missing_weight, RELIABILITY_BIN_WEIGHT_COLUMN] = bins.loc[missing_weight, "n_samples"].astype(float)

    group_columns = ["decoder", "emission_mode", "time", "bin", "bin_left", "bin_right"]
    rows = []
    for keys, group in bins.groupby(group_columns, sort=True):
        n_samples = int(group["n_samples"].sum())
        if has_sample_weight:
            aggregation_mass = group[RELIABILITY_BIN_WEIGHT_COLUMN].astype(float)
            mass_sum = float(aggregation_mass.sum())
        else:
            aggregation_mass = group["n_samples"].astype(float)
            mass_sum = float(n_samples)

        if mass_sum > 0.0:
            weights = aggregation_mass / mass_sum
            accuracy = float((group["accuracy"].fillna(0.0) * weights).sum())
            confidence = float((group["confidence"].fillna(0.0) * weights).sum())
        else:
            accuracy = float("nan")
            confidence = float("nan")

        row = {
            **dict(zip(group_columns, keys, strict=True)),
            "n_samples": n_samples,
            "accuracy": accuracy,
            "confidence": confidence,
            "gap": accuracy - confidence if mass_sum > 0.0 else float("nan"),
        }
        if has_sample_weight:
            row[RELIABILITY_BIN_WEIGHT_COLUMN] = mass_sum
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    if has_sample_weight and not aggregated.empty:
        total_weight = float(aggregated[RELIABILITY_BIN_WEIGHT_COLUMN].sum())
        aggregated["sample_weight_fraction"] = (
            aggregated[RELIABILITY_BIN_WEIGHT_COLUMN] / total_weight if total_weight > 0.0 else 0.0
        )
    return aggregated


def build_calibration_report(
    summary_csv: Path,
    *,
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    effect_window: tuple[float, float] = (0.1, 0.8),
) -> str:
    """Build a Markdown report that foregrounds calibration metrics."""
    baseline_window = _validate_time_window(baseline_window, name="baseline_window")
    effect_window = _validate_time_window(effect_window, name="effect_window")
    summary = summarize_calibration_metrics(
        pd.read_csv(summary_csv),
        baseline_window=baseline_window,
        effect_window=effect_window,
    )
    has_emission_mode = "emission_mode" in summary.columns
    lines = [
        "# NeuRepTrace Calibration Report",
        "",
        f"- Summary CSV: `{summary_csv}`",
        f"- Baseline window: {_format_float(baseline_window[0])} to {_format_float(baseline_window[1])} s",
        f"- Effect window: {_format_float(effect_window[0])} to {_format_float(effect_window[1])} s",
        "",
    ]
    if has_emission_mode:
        lines.extend(
            [
                "| Decoder | Emission mode | Subjects | Effect ECE | Effect Brier | Effect log loss | Effect accuracy | Baseline accuracy | Best ECE time (s) | Best ECE | Accuracy at best ECE |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Decoder | Subjects | Effect ECE | Effect Brier | Effect log loss | Effect accuracy | Baseline accuracy | Best ECE time (s) | Best ECE | Accuracy at best ECE |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in summary.itertuples(index=False):
        emission_prefix = f"| {row.decoder} | {row.emission_mode} |" if has_emission_mode else f"| {row.decoder} |"
        lines.append(
            f"{emission_prefix} {row.n_subjects} | {_format_float(row.effect_ece_mean)} | {_format_float(row.effect_brier_mean)} | "
            f"{_format_float(row.effect_log_loss_mean)} | {_format_float(row.effect_accuracy_mean)} | {_format_float(row.baseline_accuracy_mean)} | "
            f"{_format_float(row.best_ece_time)} | {_format_float(row.best_ece)} | {_format_float(row.accuracy_at_best_ece)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create calibration-focused summaries from NeuRepTrace benchmark outputs."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("calibration_csv", nargs="*", help="Optional reliability-bin CSVs or glob patterns.")
    parser.add_argument("--out-report", type=Path)
    parser.add_argument("--out-bins", type=Path)
    parser.add_argument("--baseline-window", type=float, nargs=2, default=(-0.1, 0.0), metavar=("START", "STOP"))
    parser.add_argument("--effect-window", type=float, nargs=2, default=(0.1, 0.8), metavar=("START", "STOP"))
    args = parser.parse_args()

    report = build_calibration_report(
        args.summary_csv,
        baseline_window=tuple(args.baseline_window),
        effect_window=tuple(args.effect_window),
    )
    if args.out_report is None:
        print(report)
    else:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report, encoding="utf-8")
        print(f"Wrote calibration report: {args.out_report}")

    calibration_csvs = _expand_paths(args.calibration_csv)
    if args.out_bins is not None:
        aggregated = aggregate_reliability_bins(calibration_csvs)
        args.out_bins.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(args.out_bins, index=False)
        print(f"Wrote aggregate reliability bins: {args.out_bins}")


if __name__ == "__main__":
    main()
