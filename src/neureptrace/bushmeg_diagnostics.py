"""Diagnostics for strict source-only BUSH-MEG LOSO benchmark outputs.

The source-only and covariance LOSO runners write one row per held-out subject
plus optional held-out trial predictions.  This module turns those raw artifacts
into compact benchmark diagnostics that are useful when deciding whether a new
candidate is a real improvement rather than fold noise or class collapse.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_SEED = 13
SUMMARY_METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "log_loss",
)
OUTPUT_FILENAMES = {
    "overall": "bushmeg_diagnostics_overall.csv",
    "subjects": "bushmeg_diagnostics_subjects.csv",
    "candidates": "bushmeg_diagnostics_candidates.csv",
    "classes": "bushmeg_diagnostics_classes.csv",
    "confusion": "bushmeg_diagnostics_confusion.csv",
}


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, table_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _subject_column(summary: pd.DataFrame) -> str:
    if "outer_test_subject" in summary.columns:
        return "outer_test_subject"
    if "subject" in summary.columns:
        return "subject"
    raise ValueError("Summary table must contain either 'outer_test_subject' or 'subject'.")


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(np.isfinite(values))


def _first_nonempty_string(values: pd.Series) -> str | None:
    for value in values.dropna().astype(str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _validate_chance(chance: float) -> float:
    chance = float(chance)
    if not np.isfinite(chance) or not 0.0 < chance < 1.0:
        raise ValueError("chance must be a finite probability strictly between 0 and 1.")
    return chance


def infer_balanced_accuracy_chance(
    summary: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    chance: float | None = None,
) -> float:
    """Infer the balanced-accuracy chance level from BUSH-MEG outputs.

    Source-only LOSO summaries produced by NeuRepTrace include ``n_classes`` and
    ``class_names``.  The explicit ``chance`` argument takes precedence and is
    useful for external summary tables that do not preserve those metadata.
    """

    if chance is not None:
        return _validate_chance(chance)

    if "n_classes" in summary.columns:
        values = _finite_numeric(summary, "n_classes").dropna().astype(int).unique()
        if len(values) == 1 and int(values[0]) > 1:
            return 1.0 / float(values[0])

    if "class_names" in summary.columns:
        class_names = _first_nonempty_string(summary["class_names"])
        if class_names is not None:
            classes = [token for token in class_names.split("|") if token]
            if len(classes) > 1:
                return 1.0 / float(len(classes))

    if predictions is not None:
        true_column, _ = _prediction_label_columns(predictions)
        n_classes = int(predictions[true_column].dropna().nunique())
        if n_classes > 1:
            return 1.0 / float(n_classes)

    raise ValueError("Could not infer chance level; pass --chance explicitly.")


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        raise ValueError("Cannot bootstrap an empty metric vector.")
    if n_resamples <= 0 or values_array.size == 1:
        mean_value = float(values_array.mean())
        return mean_value, mean_value
    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    rng = np.random.default_rng(int(random_state))
    indices = rng.integers(0, values_array.size, size=(int(n_resamples), values_array.size))
    bootstrap_means = values_array[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def subject_diagnostics(summary: pd.DataFrame, *, chance: float) -> pd.DataFrame:
    """Return per-held-out-subject LOSO metrics with chance-normalized scores."""

    _require_columns(summary, ["balanced_accuracy"], table_name="summary")
    subject_column = _subject_column(summary)
    metrics = [column for column in SUMMARY_METRIC_COLUMNS if column in summary.columns]
    rows: list[dict[str, Any]] = []
    for subject, group in summary.groupby(subject_column, sort=True, dropna=False):
        row: dict[str, Any] = {
            "subject": str(subject),
            "n_summary_rows": int(len(group)),
        }
        if "n_test_trials" in group.columns:
            row["n_test_trials"] = int(_finite_numeric(group, "n_test_trials").fillna(0).sum())
        elif "n_test" in group.columns:
            row["n_test_trials"] = int(_finite_numeric(group, "n_test").fillna(0).sum())
        if "candidate" in group.columns:
            candidates = sorted(set(group["candidate"].dropna().astype(str)))
            row["candidate"] = candidates[0] if len(candidates) == 1 else "|".join(candidates)
        for metric in metrics:
            row[metric] = float(_finite_numeric(group, metric).mean())
        balanced = float(row["balanced_accuracy"])
        row["balanced_accuracy_excess_chance"] = balanced - chance
        row["balanced_accuracy_normalized"] = (balanced - chance) / (1.0 - chance)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def candidate_selection_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize which nested-LOSO candidate was selected for held-out subjects."""

    if "candidate" not in summary.columns:
        return pd.DataFrame()
    subject_column = _subject_column(summary)
    n_subjects = max(1, int(summary[subject_column].nunique()))
    rows: list[dict[str, Any]] = []
    for candidate, group in summary.groupby("candidate", sort=False, dropna=False):
        row: dict[str, Any] = {
            "candidate": str(candidate),
            "selected_n_subjects": int(group[subject_column].nunique()),
            "selected_fraction_subjects": float(group[subject_column].nunique() / n_subjects),
            "n_summary_rows": int(len(group)),
        }
        for column in ("accuracy", "balanced_accuracy", "top2_accuracy", "top3_accuracy", "log_loss", "inner_mean_score", "inner_std_score"):
            if column in group.columns:
                row[f"mean_{column}"] = float(_finite_numeric(group, column).mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    sort_columns = ["selected_n_subjects"]
    ascending = [False]
    if any("mean_balanced_accuracy" in row for row in rows):
        sort_columns.append("mean_balanced_accuracy")
        ascending.append(False)
    return pd.DataFrame(rows).sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def _prediction_label_columns(predictions: pd.DataFrame) -> tuple[str, str]:
    true_column = "true_class" if "true_class" in predictions.columns else "true_label" if "true_label" in predictions.columns else None
    predicted_column = "predicted_class" if "predicted_class" in predictions.columns else "predicted_label" if "predicted_label" in predictions.columns else None
    if true_column is None or predicted_column is None:
        raise ValueError("Predictions must contain true_class/predicted_class or true_label/predicted_label columns.")
    return true_column, predicted_column


def _stable_unique(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        token = str(value)
        if token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def _class_order(summary: pd.DataFrame | None, predictions: pd.DataFrame, true_column: str, predicted_column: str) -> list[str]:
    if summary is not None and true_column == "true_class" and predicted_column == "predicted_class" and "class_names" in summary.columns:
        class_names = _first_nonempty_string(summary["class_names"])
        if class_names is not None:
            classes = [token for token in class_names.split("|") if token]
            if classes:
                return classes
    return _stable_unique([*predictions[true_column].tolist(), *predictions[predicted_column].tolist()])


def prediction_diagnostics(
    predictions: pd.DataFrame,
    *,
    summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-class metrics and a long-form confusion table."""

    true_column, predicted_column = _prediction_label_columns(predictions)
    frame = predictions[[true_column, predicted_column]].dropna().copy()
    if frame.empty:
        raise ValueError("Predictions table contains no rows with both true and predicted labels.")
    frame[true_column] = frame[true_column].astype(str)
    frame[predicted_column] = frame[predicted_column].astype(str)
    classes = _class_order(summary, frame, true_column, predicted_column)
    if not classes:
        raise ValueError("Could not determine class order from predictions.")

    confusion = pd.crosstab(frame[true_column], frame[predicted_column]).reindex(index=classes, columns=classes, fill_value=0)
    confusion_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for class_name in classes:
        true_support = int(confusion.loc[class_name].sum()) if class_name in confusion.index else 0
        predicted_support = int(confusion[class_name].sum()) if class_name in confusion.columns else 0
        true_positive = int(confusion.loc[class_name, class_name]) if class_name in confusion.index and class_name in confusion.columns else 0
        recall = true_positive / true_support if true_support else np.nan
        precision = true_positive / predicted_support if predicted_support else np.nan
        f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0.0 else np.nan
        off_diagonal = confusion.loc[class_name].copy() if class_name in confusion.index else pd.Series(0, index=classes)
        if class_name in off_diagonal.index:
            off_diagonal.loc[class_name] = 0
        top_confusion_count = int(off_diagonal.max()) if len(off_diagonal) else 0
        top_confused_as = str(off_diagonal.idxmax()) if top_confusion_count > 0 else ""
        class_rows.append(
            {
                "class_name": class_name,
                "support": true_support,
                "predicted_support": predicted_support,
                "true_positive": true_positive,
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
                "top_confused_as": top_confused_as,
                "top_confusion_count": top_confusion_count,
                "top_confusion_fraction": float(top_confusion_count / true_support) if true_support else np.nan,
            }
        )
        for predicted_name in classes:
            count = int(confusion.loc[class_name, predicted_name]) if class_name in confusion.index and predicted_name in confusion.columns else 0
            confusion_rows.append(
                {
                    "true_class": class_name,
                    "predicted_class": predicted_name,
                    "count": count,
                    "true_support": true_support,
                    "fraction_of_true": float(count / true_support) if true_support else np.nan,
                    "is_correct": bool(class_name == predicted_name),
                }
            )
    return pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def overall_diagnostics(
    summary: pd.DataFrame,
    subjects: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    chance: float,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Return a one-row table with benchmark-level uncertainty and stability."""

    balanced = _finite_numeric(subjects, "balanced_accuracy").dropna().to_numpy(dtype=float)
    lower, upper = _bootstrap_mean_ci(
        balanced,
        n_resamples=int(n_bootstrap),
        confidence_level=float(confidence_level),
        random_state=int(random_state),
    )
    row: dict[str, Any] = {
        "n_subjects": int(len(subjects)),
        "n_summary_rows": int(len(summary)),
        "chance_balanced_accuracy": chance,
        "mean_balanced_accuracy_subject": float(np.mean(balanced)),
        "std_balanced_accuracy_subject": float(np.std(balanced, ddof=1)) if balanced.size > 1 else 0.0,
        "sem_balanced_accuracy_subject": float(np.std(balanced, ddof=1) / np.sqrt(balanced.size)) if balanced.size > 1 else 0.0,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_resamples": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "mean_balanced_accuracy_excess_chance": float(np.mean(balanced) - chance),
        "mean_balanced_accuracy_normalized": float((np.mean(balanced) - chance) / (1.0 - chance)),
    }
    if "n_test_trials" in subjects.columns:
        row["n_test_trials"] = int(_finite_numeric(subjects, "n_test_trials").fillna(0).sum())
    if "accuracy" in subjects.columns:
        accuracies = _finite_numeric(subjects, "accuracy").dropna().to_numpy(dtype=float)
        row["mean_accuracy_subject"] = float(np.mean(accuracies)) if accuracies.size else np.nan
        if "n_test_trials" in subjects.columns and accuracies.size == len(subjects):
            weights = _finite_numeric(subjects, "n_test_trials").fillna(0).to_numpy(dtype=float)
            if np.all(weights >= 0.0) and float(weights.sum()) > 0.0:
                row["trial_weighted_accuracy"] = float(np.average(_finite_numeric(subjects, "accuracy").to_numpy(dtype=float), weights=weights))
    if not candidates.empty:
        top = candidates.iloc[0]
        row["n_unique_selected_candidates"] = int(len(candidates))
        row["most_selected_candidate"] = str(top["candidate"])
        row["most_selected_fraction_subjects"] = float(top["selected_fraction_subjects"])
    return pd.DataFrame([row])


def build_bushmeg_diagnostics(
    summary: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    chance: float | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Build all available BUSH-MEG benchmark diagnostics from loaded tables."""

    summary = summary.copy()
    predictions = None if predictions is None else predictions.copy()
    inferred_chance = infer_balanced_accuracy_chance(summary, predictions, chance=chance)
    subjects = subject_diagnostics(summary, chance=inferred_chance)
    candidates = candidate_selection_diagnostics(summary)
    tables: dict[str, pd.DataFrame] = {
        "overall": overall_diagnostics(
            summary,
            subjects,
            candidates,
            chance=inferred_chance,
            n_bootstrap=int(n_bootstrap),
            confidence_level=float(confidence_level),
            random_state=int(random_state),
        ),
        "subjects": subjects,
    }
    if not candidates.empty:
        tables["candidates"] = candidates
    if predictions is not None:
        classes, confusion = prediction_diagnostics(predictions, summary=summary)
        tables["classes"] = classes
        tables["confusion"] = confusion
    return tables


def write_diagnostics_tables(tables: Mapping[str, pd.DataFrame], out_dir: str | Path) -> dict[str, Path]:
    """Write diagnostics tables to ``out_dir`` and return their paths."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        if table.empty:
            continue
        filename = OUTPUT_FILENAMES.get(name, f"bushmeg_diagnostics_{name}.csv")
        path = out_path / filename
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_bushmeg_diagnostics(
    summary_path: str | Path,
    *,
    predictions_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    chance: float | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Path]:
    """Read LOSO CSV artifacts, build diagnostics, and write CSV reports."""

    summary_path = Path(summary_path)
    predictions = None if predictions_path is None else pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)
    output_dir = Path(out_dir) if out_dir is not None else summary_path.with_name("bushmeg_diagnostics")
    tables = build_bushmeg_diagnostics(
        summary,
        predictions,
        chance=chance,
        n_bootstrap=int(n_bootstrap),
        confidence_level=float(confidence_level),
        random_state=int(random_state),
    )
    return write_diagnostics_tables(tables, output_dir)


def _jsonable_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize strict source-only BUSH-MEG LOSO outputs with subject/bootstrap and class-confusion diagnostics.")
    parser.add_argument("summary_csv", type=Path, help="BUSH-MEG source/covariance LOSO summary CSV.")
    parser.add_argument("--predictions", type=Path, help="Optional held-out trial predictions CSV.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for diagnostic CSV files. Defaults to ./bushmeg_diagnostics beside the summary CSV.")
    parser.add_argument("--chance", type=float, help="Balanced-accuracy chance level. Defaults to 1/n_classes inferred from the summary or predictions.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES, help="Number of subject-bootstrap resamples for the mean balanced-accuracy CI.")
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL, help="Bootstrap confidence level, e.g. 0.95.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for subject bootstrap resampling.")
    args = parser.parse_args(argv)

    written = run_bushmeg_diagnostics(
        args.summary_csv,
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        chance=args.chance,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_state=args.random_state,
    )
    print(json.dumps(_jsonable_paths(written), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
