"""Compare two BUSH-MEG LOSO decoding artifacts.

The utility is CSV-based so it can compare historical GitHub Action artifacts,
local smoke runs, and new source-only sweeps.  It reports fold-level metric
deltas, prediction-row alignment diagnostics, and per-class recall deltas.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SUMMARY_METRICS = ("balanced_accuracy", "accuracy", "top2_accuracy", "top3_accuracy", "log_loss")
DEFAULT_GROUP_COLUMN = "outer_test_subject"


def _existing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in {"log_loss", "brier", "ece"} else "higher_is_better"


def compare_summary_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    group_column: str = DEFAULT_GROUP_COLUMN,
    metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
) -> pd.DataFrame:
    """Return fold-level and mean metric deltas between two summary CSVs."""

    rows: list[dict[str, object]] = []
    available_metrics = [metric for metric in metrics if metric in reference.columns and metric in candidate.columns]
    if not available_metrics:
        raise ValueError("No requested metric columns are present in both summary frames.")
    if group_column in reference.columns and group_column in candidate.columns:
        ref_grouped = reference.groupby(group_column, dropna=False)
        cand_grouped = candidate.groupby(group_column, dropna=False)
        groups = sorted(set(ref_grouped.groups) | set(cand_grouped.groups), key=str)
    else:
        reference = reference.reset_index().rename(columns={"index": "row_index"})
        candidate = candidate.reset_index().rename(columns={"index": "row_index"})
        group_column = "row_index"
        ref_grouped = reference.groupby(group_column, dropna=False)
        cand_grouped = candidate.groupby(group_column, dropna=False)
        groups = sorted(set(ref_grouped.groups) | set(cand_grouped.groups), key=str)

    for metric in available_metrics:
        for group in groups:
            ref_value = float(ref_grouped[metric].mean().get(group, np.nan))
            cand_value = float(cand_grouped[metric].mean().get(group, np.nan))
            rows.append({group_column: group, "metric": metric, "direction": _metric_direction(metric), "reference": ref_value, "candidate": cand_value, "delta_candidate_minus_reference": cand_value - ref_value})
        rows.append({group_column: "__mean__", "metric": metric, "direction": _metric_direction(metric), "reference": float(reference[metric].mean()), "candidate": float(candidate[metric].mean()), "delta_candidate_minus_reference": float(candidate[metric].mean() - reference[metric].mean())})
    return pd.DataFrame(rows)


def _prediction_key_columns(reference: pd.DataFrame, candidate: pd.DataFrame) -> list[str]:
    for columns in (("outer_test_subject", "trial_index"), ("participant", "trial_index"), ("participant", "trial"), ("validation_trial_index",), ("trial_index",)):
        if all(column in reference.columns and column in candidate.columns for column in columns):
            return list(columns)
    return []


def _label_columns(frame: pd.DataFrame) -> tuple[str, str]:
    true_column = next((column for column in ("true_label", "true_class", "true_stimulus", "stimulus_class") if column in frame.columns), None)
    pred_column = next((column for column in ("predicted_label", "predicted_class", "predicted_stimulus") if column in frame.columns), None)
    if true_column is None or pred_column is None:
        raise ValueError("Prediction frames must contain true and predicted label/class columns.")
    return true_column, pred_column


def per_class_recall_frame(predictions: pd.DataFrame, *, group_columns: Sequence[str] = (DEFAULT_GROUP_COLUMN,)) -> pd.DataFrame:
    """Compute per-group/per-class recall from a prediction CSV."""

    true_column, predicted_column = _label_columns(predictions)
    group_columns = _existing_columns(predictions, group_columns)
    frame = predictions.copy()
    frame["__correct"] = frame[true_column].astype(str) == frame[predicted_column].astype(str)
    rows: list[dict[str, object]] = []
    grouped = frame.groupby([*group_columns, true_column], dropna=False) if group_columns else frame.groupby(true_column, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip([*group_columns, "true_class"], key, strict=True))
        row.update({"n_trials": int(len(group)), "n_correct": int(group["__correct"].sum()), "recall": float(group["__correct"].mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_prediction_frames(reference: pd.DataFrame, candidate: pd.DataFrame, *, group_columns: Sequence[str] = (DEFAULT_GROUP_COLUMN,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return label-mismatch diagnostics and per-class recall deltas."""

    key_columns = _prediction_key_columns(reference, candidate)
    ref_true, ref_pred = _label_columns(reference)
    cand_true, cand_pred = _label_columns(candidate)
    diagnostics: list[dict[str, object]] = []
    if key_columns:
        merged = reference[key_columns + [ref_true, ref_pred]].merge(
            candidate[key_columns + [cand_true, cand_pred]],
            on=key_columns,
            how="outer",
            suffixes=("_reference", "_candidate"),
            indicator=True,
        )
        ref_true_name = f"{ref_true}_reference" if ref_true == cand_true else ref_true
        cand_true_name = f"{cand_true}_candidate" if ref_true == cand_true else cand_true
        matched = merged["_merge"] == "both"
        mismatched = matched & (merged[ref_true_name].astype(str) != merged[cand_true_name].astype(str))
        diagnostics.extend([
            {"diagnostic": "matched_prediction_rows", "value": int(matched.sum())},
            {"diagnostic": "reference_only_rows", "value": int((merged["_merge"] == "left_only").sum())},
            {"diagnostic": "candidate_only_rows", "value": int((merged["_merge"] == "right_only").sum())},
            {"diagnostic": "true_label_mismatch_rows", "value": int(mismatched.sum())},
        ])

    group_columns = [column for column in group_columns if column in reference.columns and column in candidate.columns]
    ref_recall = per_class_recall_frame(reference, group_columns=group_columns).rename(columns={"recall": "reference_recall", "n_trials": "reference_n_trials", "n_correct": "reference_n_correct"})
    cand_recall = per_class_recall_frame(candidate, group_columns=group_columns).rename(columns={"recall": "candidate_recall", "n_trials": "candidate_n_trials", "n_correct": "candidate_n_correct"})
    join_columns = [*group_columns, "true_class"]
    recall_delta = ref_recall.merge(cand_recall, on=join_columns, how="outer")
    recall_delta["delta_candidate_minus_reference"] = recall_delta["candidate_recall"] - recall_delta["reference_recall"]
    return pd.DataFrame(diagnostics), recall_delta


def _parse_metrics(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_SUMMARY_METRICS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two BUSH-MEG LOSO decoding artifacts.")
    parser.add_argument("reference_summary", type=Path)
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--candidate-predictions", type=Path)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--metrics")
    parser.add_argument("--summary-out", type=Path, default=Path("artifact_summary_diff.csv"))
    parser.add_argument("--prediction-diagnostics-out", type=Path, default=Path("artifact_prediction_diagnostics.csv"))
    parser.add_argument("--per-class-out", type=Path, default=Path("artifact_per_class_recall_diff.csv"))
    args = parser.parse_args(argv)

    summary = compare_summary_frames(pd.read_csv(args.reference_summary), pd.read_csv(args.candidate_summary), group_column=args.group_column, metrics=_parse_metrics(args.metrics))
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    print(f"Wrote {args.summary_out}")
    for row in summary[summary[args.group_column] == "__mean__"].to_dict("records"):
        print(f"{row['metric']}: reference={row['reference']:.6g} candidate={row['candidate']:.6g} delta={row['delta_candidate_minus_reference']:.6g}")

    if args.reference_predictions and args.candidate_predictions:
        diagnostics, per_class = compare_prediction_frames(pd.read_csv(args.reference_predictions), pd.read_csv(args.candidate_predictions), group_columns=(args.group_column,))
        args.prediction_diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        args.per_class_out.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.prediction_diagnostics_out, index=False)
        per_class.to_csv(args.per_class_out, index=False)
        print(f"Wrote {args.prediction_diagnostics_out} and {args.per_class_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
