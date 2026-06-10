"""Diagnostics for LOSO probability-observation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error
from neureptrace.observations import probability_columns

MAXIMIZE_METRICS = {"accuracy", "balanced_accuracy", "top2_accuracy", "top3_accuracy"}
MINIMIZE_METRICS = {"log_loss", "brier", "ece"}
SELECTION_METRICS = tuple(sorted(MAXIMIZE_METRICS | MINIMIZE_METRICS))
DEFAULT_SELECTIVE_COVERAGES = (1.0, 0.9, 0.8, 0.7)
SINGLE_PROTOCOL_COLUMNS = (
    "decoder",
    "backend",
    "emission_mode",
    "feature_preprocessor",
    "pca_components",
    "normalization",
    "temporal_mode",
    "class_prior_correction",
    "source_calibration",
    "source_time_selection",
    "alignment_method",
    "alignment_anchor_mode",
    "alignment_anchor_column",
    "alignment_target_projection",
    "alignment_valid_for_benchmark",
    "response_window_combine",
    "response_window_requested_times",
    "response_window_actual_times",
    "label_shuffle_control",
    "label_shuffle_seed",
)


def _sem(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return float("nan")
    return float(numeric.sem())


def _subject_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Subject column '{requested}' not found in observations.")
        return requested
    for column in ("group", "outer_group", "subject"):
        if column in frame.columns and not frame[column].isna().all():
            return column
    raise ValueError("Observation table must contain group, outer_group, or subject.")


def _protocol_token(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if pd.isna(value):
        return ""
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false", "yes", "no", "on", "off"}:
        return lowered
    return text


def _validate_single_protocol_observations(frame: pd.DataFrame) -> None:
    for column in SINGLE_PROTOCOL_COLUMNS:
        if column not in frame.columns:
            continue
        tokens = [_protocol_token(value) for value in frame[column].drop_duplicates().tolist()]
        values = sorted({token for token in tokens if token})
        if len(values) > 1:
            raise ValueError(f"Observation table mixes {column!r} provenance values: {values}")
        if values and any(not token for token in tokens):
            raise ValueError(f"Observation table has missing {column!r} provenance mixed with {values[0]!r}.")


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = probability_columns(frame)
    if not columns:
        raise ValueError("Observation table must contain prob_class_* columns.")
    probabilities = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Observation table prob_class_* values must be finite.")
    if bool((probabilities < 0).any()):
        raise ValueError("Observation table prob_class_* values must be non-negative.")
    row_sums = probabilities.sum(axis=1)
    if not np.isclose(row_sums, 1.0, rtol=0.0, atol=1e-6).all():
        raise ValueError("Observation table prob_class_* rows must sum to 1.")
    return probabilities


def _integer_array(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Observation table {name} values must be numeric and non-missing.")
    value_array = numeric.to_numpy(dtype=float)
    rounded = np.rint(value_array)
    if not bool(np.isclose(value_array, rounded, rtol=0.0, atol=1.0e-12).all()):
        raise ValueError(f"Observation table {name} values must be integer-valued.")
    return rounded.astype(int)


def _label_array(frame: pd.DataFrame) -> np.ndarray:
    if "true_label" not in frame.columns:
        raise ValueError("Observation table must contain true_label.")
    return _integer_array(frame["true_label"], name="true_label")


def _predicted_array(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_label" in frame.columns:
        return _integer_array(frame["predicted_label"], name="predicted_label")
    return _probability_matrix(frame).argmax(axis=1)


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if len(labels) == 0:
        return float("nan")
    effective_k = min(int(k), probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, ::-1][:, :effective_k]
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _top_k_chance(n_classes: int, *, k: int) -> float:
    return min(int(k), int(n_classes)) / float(n_classes)


def _top_k_interpretation(n_classes: int, *, k: int) -> str:
    return "automatic_ceiling" if int(n_classes) <= int(k) else "informative"


def _metrics_for_rows(frame: pd.DataFrame) -> dict[str, float | int | str]:
    probabilities = _probability_matrix(frame)
    labels = _label_array(frame)
    predictions = _predicted_array(frame)
    n_classes = probabilities.shape[1]
    if bool(((labels < 0) | (labels >= n_classes)).any()):
        raise ValueError("Observation table true_label values must index prob_class_* columns.")
    if bool(((predictions < 0) | (predictions >= n_classes)).any()):
        raise ValueError("Observation table predicted_label values must index prob_class_* columns.")
    if len(frame) == 0:
        return {
            "n_observations": 0,
            "n_classes": n_classes,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "top2_accuracy": float("nan"),
            "top3_accuracy": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    return {
        "n_observations": int(len(frame)),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "top2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
        "top3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(n_classes))),
        "brier": float(brier_score_multiclass(probabilities, labels)),
        "ece": float(expected_calibration_error(probabilities, labels)),
    }


def _confidence_array(frame: pd.DataFrame) -> np.ndarray:
    if "confidence" in frame.columns:
        confidence = pd.to_numeric(frame["confidence"], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isfinite(confidence)):
            return confidence
    probabilities = _probability_matrix(frame)
    return probabilities.max(axis=1)


def _metric_value_for_selection(frame: pd.DataFrame, metric: str) -> float:
    value = _metrics_for_rows(frame)[metric]
    return float(value)


def time_course_summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_classes = len(probability_columns(observations))
    chance = 1.0 / float(n_classes)
    top2_chance = _top_k_chance(n_classes, k=2)
    top3_chance = _top_k_chance(n_classes, k=3)
    top2_interpretation = _top_k_interpretation(n_classes, k=2)
    top3_interpretation = _top_k_interpretation(n_classes, k=3)
    for time, frame in observations.groupby("time", sort=True):
        rows.append(
            {
                "time": float(time),
                **_metrics_for_rows(frame),
                "chance_accuracy": chance,
                "top2_chance": top2_chance,
                "top3_chance": top3_chance,
                "top2_interpretation": top2_interpretation,
                "top3_interpretation": top3_interpretation,
            }
        )
    return pd.DataFrame(rows)


def _best_time(summary: pd.DataFrame, metric: str) -> float:
    if metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric '{metric}'. Available metrics: {', '.join(SELECTION_METRICS)}.")
    if summary.empty:
        raise ValueError("Cannot select a best time from an empty summary.")
    if metric in MINIMIZE_METRICS:
        index = summary[metric].astype(float).idxmin()
    else:
        index = summary[metric].astype(float).idxmax()
    return float(summary.loc[index, "time"])


def _nearest_time(frame: pd.DataFrame, requested_time: float) -> float:
    times = np.asarray(sorted(frame["time"].dropna().unique()), dtype=float)
    if times.size == 0:
        raise ValueError("Observation table contains no time values.")
    return float(times[np.argmin(np.abs(times - float(requested_time)))])


def _rows_at_time(frame: pd.DataFrame, time: float) -> pd.DataFrame:
    actual_time = _nearest_time(frame, time)
    return frame.loc[np.isclose(frame["time"].astype(float), actual_time)].copy()


def _class_count_string(frame: pd.DataFrame) -> str:
    counts = frame["true_class"].astype(str).value_counts().sort_index()
    return json.dumps({str(label): int(count) for label, count in counts.items()}, sort_keys=True, separators=(",", ":"))


def _read_optional_stage_summary(stage_summary_csv: str | Path | None) -> pd.DataFrame | None:
    if stage_summary_csv is None:
        return None
    stage_summary_path = Path(stage_summary_csv)
    if not stage_summary_path.is_file():
        return None
    return pd.read_csv(stage_summary_path)


def per_subject_diagnostics(
    observations: pd.DataFrame,
    *,
    subject_column: str,
    fixed_time: float,
    selection_metric: str = "balanced_accuracy",
    stage_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fixed = _rows_at_time(observations, fixed_time)
    n_classes = len(probability_columns(observations))
    chance = 1.0 / float(n_classes)
    top2_chance = _top_k_chance(n_classes, k=2)
    top3_chance = _top_k_chance(n_classes, k=3)
    staged_trials: dict[str, int] = {}
    if stage_summary is not None and {"subject", "n_trials"}.issubset(stage_summary.columns):
        staged_trials = {
            str(row.subject): int(row.n_trials)
            for row in stage_summary.loc[:, ["subject", "n_trials"]].itertuples(index=False)
        }

    rows = []
    for subject, subject_frame in observations.groupby(subject_column, sort=True):
        subject_summary = time_course_summary(subject_frame)
        best_time = _best_time(subject_summary, selection_metric)
        best_rows = _rows_at_time(subject_frame, best_time)
        fixed_rows = fixed.loc[fixed[subject_column].astype(str) == str(subject)]
        best_metrics = _metrics_for_rows(best_rows)
        fixed_metrics = _metrics_for_rows(fixed_rows)
        n_trials = int(fixed_rows["sample_index"].nunique()) if "sample_index" in fixed_rows.columns else int(len(fixed_rows))
        rows.append(
            {
                "subject": str(subject),
                "best_time": best_time,
                "balanced_accuracy": best_metrics["balanced_accuracy"],
                "balanced_minus_chance": float(best_metrics["balanced_accuracy"]) - chance,
                "top2_accuracy": best_metrics["top2_accuracy"],
                "top2_minus_chance": float(best_metrics["top2_accuracy"]) - top2_chance,
                "top3_accuracy": best_metrics["top3_accuracy"],
                "top3_minus_chance": float(best_metrics["top3_accuracy"]) - top3_chance,
                "fixed_time": _nearest_time(observations, fixed_time),
                "fixed_balanced_accuracy": fixed_metrics["balanced_accuracy"],
                "fixed_balanced_minus_chance": float(fixed_metrics["balanced_accuracy"]) - chance,
                "fixed_top2_accuracy": fixed_metrics["top2_accuracy"],
                "fixed_top2_minus_chance": float(fixed_metrics["top2_accuracy"]) - top2_chance,
                "fixed_top3_accuracy": fixed_metrics["top3_accuracy"],
                "fixed_top3_minus_chance": float(fixed_metrics["top3_accuracy"]) - top3_chance,
                "n_trials": n_trials,
                "staged_n_trials": staged_trials.get(str(subject), ""),
                "class_counts": _class_count_string(fixed_rows),
                "top2_interpretation": _top_k_interpretation(n_classes, k=2),
                "top3_interpretation": _top_k_interpretation(n_classes, k=3),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(observations: pd.DataFrame, *, time: float) -> pd.DataFrame:
    fixed = _rows_at_time(observations, time)
    classes = sorted(set(fixed["true_class"].astype(str)) | set(fixed["predicted_class"].astype(str)))
    table = pd.crosstab(fixed["true_class"].astype(str), fixed["predicted_class"].astype(str)).reindex(index=classes, columns=classes, fill_value=0)
    rows = []
    actual_time = _nearest_time(observations, time)
    for true_class in classes:
        for predicted_class in classes:
            rows.append(
                {
                    "time": actual_time,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": int(table.loc[true_class, predicted_class]),
                }
            )
    return pd.DataFrame(rows)


def class_counts(observations: pd.DataFrame, *, subject_column: str, time: float) -> pd.DataFrame:
    fixed = _rows_at_time(observations, time)
    counts = (
        fixed.groupby([subject_column, "true_class"], dropna=False)
        .size()
        .reset_index(name="n_trials")
        .rename(columns={subject_column: "subject"})
    )
    counts.insert(0, "time", _nearest_time(observations, time))
    return counts


def selective_coverage_summary(
    observations: pd.DataFrame,
    *,
    time: float,
    coverages: tuple[float, ...] = DEFAULT_SELECTIVE_COVERAGES,
) -> pd.DataFrame:
    """Return confidence-thresholded fixed-time metrics.

    These rows are diagnostics only: they describe how accuracy changes when
    low-confidence held-out observations are rejected, while the main LOSO
    quality summary remains the full-coverage result.
    """

    fixed = _rows_at_time(observations, time).reset_index(drop=True)
    if fixed.empty:
        raise ValueError("Cannot compute selective coverage from an empty fixed-time table.")
    probabilities = _probability_matrix(fixed)
    n_classes = probabilities.shape[1]
    confidence = _confidence_array(fixed)
    order = np.lexsort((np.arange(len(fixed)), -confidence))
    sorted_fixed = fixed.iloc[order].reset_index(drop=True)
    sorted_confidence = confidence[order]
    rows = []
    total = int(len(sorted_fixed))
    actual_time = _nearest_time(observations, time)
    for coverage in coverages:
        target = float(coverage)
        if not 0.0 < target <= 1.0:
            raise ValueError("Selective coverage targets must be in (0, 1].")
        n_selected = total if np.isclose(target, 1.0) else int(np.ceil(total * target))
        n_selected = min(max(n_selected, 1), total)
        selected = sorted_fixed.iloc[:n_selected].copy()
        metrics = _metrics_for_rows(selected)
        selected_labels = _label_array(selected)
        class_support = {
            str(class_index): int(np.sum(selected_labels == class_index))
            for class_index in range(n_classes)
        }
        rows.append(
            {
                "time": actual_time,
                "coverage_target": target,
                "coverage": n_selected / float(total),
                "n_selected": int(n_selected),
                "n_total": total,
                "rejection_rate": 1.0 - n_selected / float(total),
                "confidence_threshold": float(sorted_confidence[n_selected - 1]),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "selective_risk": 1.0 - float(metrics["accuracy"]),
                "balanced_selective_risk": 1.0 - float(metrics["balanced_accuracy"]),
                "n_selected_classes": int(sum(count > 0 for count in class_support.values())),
                "all_classes_present": all(count > 0 for count in class_support.values()),
                "selected_class_support": json.dumps(class_support, sort_keys=True, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def quality_summary(
    observations: pd.DataFrame,
    *,
    time_summary: pd.DataFrame,
    per_subject: pd.DataFrame,
    subject_column: str,
    fixed_time: float,
    selection_metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Return one paper-facing summary row for LOSO decode quality."""

    n_classes = len(probability_columns(observations))
    chance = 1.0 / float(n_classes)
    top2_chance = _top_k_chance(n_classes, k=2)
    top3_chance = _top_k_chance(n_classes, k=3)
    fixed_rows = _rows_at_time(observations, fixed_time)
    fixed_metrics = _metrics_for_rows(fixed_rows)
    global_best_time = _best_time(time_summary, selection_metric)
    global_best_row = time_summary.loc[np.isclose(time_summary["time"].astype(float), global_best_time)].iloc[0]
    label_shuffle_values = (
        sorted(observations["label_shuffle_control"].dropna().astype(str).unique().tolist())
        if "label_shuffle_control" in observations.columns
        else []
    )
    label_shuffle_seed_values = (
        sorted(observations["label_shuffle_seed"].dropna().astype(str).unique().tolist())
        if "label_shuffle_seed" in observations.columns
        else []
    )
    subject_best = pd.to_numeric(per_subject["balanced_accuracy"], errors="coerce")
    subject_fixed = pd.to_numeric(per_subject["fixed_balanced_accuracy"], errors="coerce")

    return pd.DataFrame(
        [
            {
                "subject_column": subject_column,
                "n_subjects": int(per_subject["subject"].nunique()),
                "n_observations": int(len(observations)),
                "n_observations_fixed_time": int(len(fixed_rows)),
                "n_classes": int(n_classes),
                "selection_metric": selection_metric,
                "global_best_time": global_best_time,
                "global_best_selection_value": float(global_best_row[selection_metric]),
                "fixed_time": _nearest_time(observations, fixed_time),
                "chance_accuracy": chance,
                "top2_chance": top2_chance,
                "top3_chance": top3_chance,
                "top2_interpretation": _top_k_interpretation(n_classes, k=2),
                "top3_interpretation": _top_k_interpretation(n_classes, k=3),
                "fixed_accuracy": fixed_metrics["accuracy"],
                "fixed_balanced_accuracy": fixed_metrics["balanced_accuracy"],
                "fixed_balanced_minus_chance": float(fixed_metrics["balanced_accuracy"]) - chance,
                "fixed_top2_accuracy": fixed_metrics["top2_accuracy"],
                "fixed_top2_minus_chance": float(fixed_metrics["top2_accuracy"]) - top2_chance,
                "fixed_top3_accuracy": fixed_metrics["top3_accuracy"],
                "fixed_top3_minus_chance": float(fixed_metrics["top3_accuracy"]) - top3_chance,
                "fixed_log_loss": fixed_metrics["log_loss"],
                "fixed_brier": fixed_metrics["brier"],
                "fixed_ece": fixed_metrics["ece"],
                "subject_best_balanced_accuracy_mean": float(subject_best.mean()),
                "subject_best_balanced_accuracy_sem": _sem(subject_best),
                "subject_fixed_balanced_accuracy_mean": float(subject_fixed.mean()),
                "subject_fixed_balanced_accuracy_sem": _sem(subject_fixed),
                "subjects_best_above_chance": int((subject_best > chance).sum()),
                "subjects_fixed_above_chance": int((subject_fixed > chance).sum()),
                "label_shuffle_control_values": ",".join(label_shuffle_values),
                "label_shuffle_seed_values": ",".join(label_shuffle_seed_values),
            }
        ]
    )


def write_loso_observation_diagnostics(
    observations_csv: str | Path,
    *,
    out_dir: str | Path,
    summary_csv: str | Path | None = None,
    stage_summary_csv: str | Path | None = None,
    best_time: float | None = None,
    selection_metric: str = "balanced_accuracy",
    subject_column: str | None = None,
) -> dict[str, Path]:
    observations = pd.read_csv(observations_csv)
    if observations.empty:
        raise ValueError("Observation table is empty.")
    _validate_single_protocol_observations(observations)
    subject_column_name = _subject_column(observations, subject_column)
    time_summary = time_course_summary(observations)
    requested_time = best_time
    if requested_time is None and summary_csv is not None:
        summary = pd.read_csv(summary_csv)
        metric_columns = [column for column in SELECTION_METRICS if column in summary.columns]
        if selection_metric not in metric_columns:
            raise ValueError(f"Selection metric '{selection_metric}' not found in summary CSV.")
        requested_time = _best_time(summary.groupby("time", as_index=False)[metric_columns].mean(), selection_metric)
    if requested_time is None:
        requested_time = _best_time(time_summary, selection_metric)
    actual_time = _nearest_time(observations, float(requested_time))

    stage_summary = _read_optional_stage_summary(stage_summary_csv)
    per_subject = per_subject_diagnostics(
        observations,
        subject_column=subject_column_name,
        fixed_time=actual_time,
        selection_metric=selection_metric,
        stage_summary=stage_summary,
    )
    confusion = confusion_matrix(observations, time=actual_time)
    counts = class_counts(observations, subject_column=subject_column_name, time=actual_time)
    selective = selective_coverage_summary(observations, time=actual_time)
    quality = quality_summary(
        observations,
        time_summary=time_summary,
        per_subject=per_subject,
        subject_column=subject_column_name,
        fixed_time=actual_time,
        selection_metric=selection_metric,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "time_course": out_dir / "time_course_summary.csv",
        "per_subject": out_dir / "per_subject.csv",
        "confusion_matrix": out_dir / "confusion_matrix.csv",
        "class_counts": out_dir / "class_counts.csv",
        "selective_coverage": out_dir / "selective_coverage.csv",
        "quality_summary": out_dir / "quality_summary.csv",
    }
    time_summary.to_csv(paths["time_course"], index=False)
    per_subject.to_csv(paths["per_subject"], index=False)
    confusion.to_csv(paths["confusion_matrix"], index=False)
    counts.to_csv(paths["class_counts"], index=False)
    selective.to_csv(paths["selective_coverage"], index=False)
    quality.to_csv(paths["quality_summary"], index=False)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write per-subject and confusion diagnostics from LOSO probability observations.")
    parser.add_argument("observations_csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--stage-summary-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--best-time", type=float)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="balanced_accuracy")
    parser.add_argument("--subject-column")
    args = parser.parse_args(argv)

    paths = write_loso_observation_diagnostics(
        args.observations_csv,
        out_dir=args.out_dir,
        summary_csv=args.summary_csv,
        stage_summary_csv=args.stage_summary_csv,
        best_time=args.best_time,
        selection_metric=args.selection_metric,
        subject_column=args.subject_column,
    )
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
