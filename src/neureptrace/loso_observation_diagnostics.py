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


def _subject_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Subject column '{requested}' not found in observations.")
        return requested
    for column in ("group", "outer_group", "subject"):
        if column in frame.columns and not frame[column].isna().all():
            return column
    raise ValueError("Observation table must contain group, outer_group, or subject.")


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = probability_columns(frame)
    if not columns:
        raise ValueError("Observation table must contain prob_class_* columns.")
    return frame.loc[:, list(columns)].to_numpy(dtype=float)


def _label_array(frame: pd.DataFrame) -> np.ndarray:
    if "true_label" not in frame.columns:
        raise ValueError("Observation table must contain true_label.")
    return frame["true_label"].astype(int).to_numpy()


def _predicted_array(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_label" in frame.columns:
        return frame["predicted_label"].astype(int).to_numpy()
    return _probability_matrix(frame).argmax(axis=1)


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if len(labels) == 0:
        return float("nan")
    effective_k = min(int(k), probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, ::-1][:, :effective_k]
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _metrics_for_rows(frame: pd.DataFrame) -> dict[str, float | int | str]:
    probabilities = _probability_matrix(frame)
    labels = _label_array(frame)
    predictions = _predicted_array(frame)
    n_classes = probabilities.shape[1]
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


def _metric_value_for_selection(frame: pd.DataFrame, metric: str) -> float:
    value = _metrics_for_rows(frame)[metric]
    return float(value)


def time_course_summary(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_classes = len(probability_columns(observations))
    chance = 1.0 / float(n_classes)
    top2_chance = min(2, n_classes) / float(n_classes)
    top3_chance = min(3, n_classes) / float(n_classes)
    top3_interpretation = "automatic_ceiling" if n_classes <= 3 else "informative"
    for time, frame in observations.groupby("time", sort=True):
        rows.append(
            {
                "time": float(time),
                **_metrics_for_rows(frame),
                "chance_accuracy": chance,
                "top2_chance": top2_chance,
                "top3_chance": top3_chance,
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
                "top3_accuracy": best_metrics["top3_accuracy"],
                "fixed_time": _nearest_time(observations, fixed_time),
                "fixed_balanced_accuracy": fixed_metrics["balanced_accuracy"],
                "fixed_balanced_minus_chance": float(fixed_metrics["balanced_accuracy"]) - chance,
                "fixed_top2_accuracy": fixed_metrics["top2_accuracy"],
                "n_trials": n_trials,
                "staged_n_trials": staged_trials.get(str(subject), ""),
                "class_counts": _class_count_string(fixed_rows),
                "top3_interpretation": "automatic_ceiling" if n_classes <= 3 else "informative",
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

    stage_summary = pd.read_csv(stage_summary_csv) if stage_summary_csv is not None else None
    per_subject = per_subject_diagnostics(
        observations,
        subject_column=subject_column_name,
        fixed_time=actual_time,
        selection_metric=selection_metric,
        stage_summary=stage_summary,
    )
    confusion = confusion_matrix(observations, time=actual_time)
    counts = class_counts(observations, subject_column=subject_column_name, time=actual_time)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "time_course": out_dir / "time_course_summary.csv",
        "per_subject": out_dir / "per_subject.csv",
        "confusion_matrix": out_dir / "confusion_matrix.csv",
        "class_counts": out_dir / "class_counts.csv",
    }
    time_summary.to_csv(paths["time_course"], index=False)
    per_subject.to_csv(paths["per_subject"], index=False)
    confusion.to_csv(paths["confusion_matrix"], index=False)
    counts.to_csv(paths["class_counts"], index=False)
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
