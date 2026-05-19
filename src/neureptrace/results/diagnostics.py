from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from neureptrace.metrics.confusion import (
    DEFAULT_METADATA_LABEL_COLUMNS,
    confusion_counts,
    confusion_pair_summary,
    per_class_accuracy,
)
from neureptrace.metrics.ranking import rank_class_scores

DEFAULT_TOP_K = (2, 3)
DEFAULT_RANK_ROW_TOP_K = 3
DEFAULT_SCORE_COLUMN_PREFIX = "score_class_"
DEFAULT_DIAGNOSTIC_EXPORT_FILENAMES = {
    "predictions": "predictions.csv",
    "confusion": "confusion.csv",
    "per_class": "per_class.csv",
    "rank_summary": "rank_summary.csv",
    "confusion_pairs": "confusion_pairs.csv",
}


@dataclass(frozen=True)
class DiagnosticTables:
    """Reusable diagnostic result tables derived from trial-level predictions."""

    predictions: pd.DataFrame
    confusion: pd.DataFrame
    per_class: pd.DataFrame
    rank_summary: pd.DataFrame
    confusion_pairs: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return tables keyed by their stable export names."""

        return {
            "predictions": self.predictions,
            "confusion": self.confusion,
            "per_class": self.per_class,
            "rank_summary": self.rank_summary,
            "confusion_pairs": self.confusion_pairs,
        }


def prediction_diagnostic_table(
    true_labels: Sequence | np.ndarray,
    predicted_labels: Sequence | np.ndarray,
    *,
    scores: Sequence[Sequence[float]] | np.ndarray | None = None,
    classes: Sequence | np.ndarray | None = None,
    sample_ids: Sequence | np.ndarray | None = None,
    group_values: Mapping[str, object] | None = None,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    row_top_k: int = DEFAULT_RANK_ROW_TOP_K,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    sample_column: str = "sample_index",
    class_column: str = "class",
    score_column_prefix: str = DEFAULT_SCORE_COLUMN_PREFIX,
) -> pd.DataFrame:
    """Build a trial-level prediction table with optional class-score diagnostics.

    The resulting table is intentionally dataset-neutral: callers provide labels,
    predictions, optional class scores, and optional group columns such as subject,
    fold, decoder, window, or transfer condition.  When scores and classes are
    supplied, the table also includes true-label rank fields, top-k hit flags,
    per-row top-ranked classes, and one score column per class.
    """

    _require_distinct_columns(true_column, predicted_column, sample_column)
    y_true = _as_1d_array("true_labels", true_labels)
    y_pred = _as_1d_array("predicted_labels", predicted_labels)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("predicted_labels and true_labels must contain the same samples.")

    n_samples = int(y_true.shape[0])
    if sample_ids is None:
        sample_ids_array = np.arange(n_samples)
    else:
        sample_ids_array = _as_1d_array("sample_ids", sample_ids)
        if sample_ids_array.shape[0] != n_samples:
            raise ValueError("sample_ids and true_labels must contain the same samples.")

    frame = pd.DataFrame(
        {
            sample_column: sample_ids_array,
            true_column: y_true,
            predicted_column: y_pred,
            "correct": y_true == y_pred,
        }
    )
    frame = _add_group_values(frame, group_values, n_samples)

    if scores is None and classes is None:
        return frame
    if scores is None or classes is None:
        raise ValueError("scores and classes must be supplied together.")

    score_matrix = _as_score_matrix(scores, classes, n_samples=n_samples)
    class_order = np.asarray(classes).ravel()
    for class_index, column_name in enumerate(_score_column_names(class_order, score_column_prefix)):
        frame[column_name] = score_matrix[:, class_index]

    rank_result = rank_class_scores(
        score_matrix,
        class_order,
        y_true,
        top_k=top_k,
        row_top_k=row_top_k,
        class_column=class_column,
    )
    rank_rows = pd.DataFrame(rank_result["rows"])
    if not rank_rows.empty:
        frame = pd.concat([frame.reset_index(drop=True), rank_rows.reset_index(drop=True)], axis=1)
        ranks = pd.to_numeric(frame["true_label_rank"], errors="coerce")
        for k in _normalize_top_k(top_k):
            frame[f"top_{k}_hit"] = np.isfinite(ranks) & (ranks <= k)
    return frame


def diagnostic_summary_tables(
    predictions: pd.DataFrame,
    *,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] | str | None = None,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    metadata_frame: pd.DataFrame | None = None,
    metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
    label_prefix: str = "label",
    include_confusion_pairs: bool = True,
) -> DiagnosticTables:
    """Build reusable prediction, confusion, per-class, and rank summaries."""

    group_columns = _normalize_columns(group_columns)
    required_columns = [true_column, predicted_column, *group_columns]
    if participant_column is not None:
        required_columns.append(participant_column)
    _require_columns(predictions, required_columns)

    confusion = confusion_counts(
        predictions,
        true_column=true_column,
        predicted_column=predicted_column,
        group_columns=group_columns,
    )
    per_class = per_class_accuracy(
        predictions,
        true_column=true_column,
        predicted_column=predicted_column,
        participant_column=participant_column,
        group_columns=group_columns,
    )
    rank_summary = rank_summary_table(predictions, group_columns=group_columns, top_k=top_k)
    confusion_pairs = (
        confusion_pair_summary(
            predictions,
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_frame=metadata_frame,
            metadata_label_columns=metadata_label_columns,
            label_prefix=label_prefix,
        )
        if include_confusion_pairs
        else pd.DataFrame()
    )

    return DiagnosticTables(
        predictions=predictions.reset_index(drop=True),
        confusion=confusion,
        per_class=per_class,
        rank_summary=rank_summary,
        confusion_pairs=confusion_pairs,
    )


def rank_summary_table(
    predictions: pd.DataFrame,
    *,
    rank_column: str = "true_label_rank",
    group_columns: Sequence[str] | str | None = None,
    top_k: Sequence[int] = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Summarize true-label rank and top-k hit rates from a prediction table."""

    group_columns = _normalize_columns(group_columns)
    _require_columns(predictions, group_columns)
    top_k = _normalize_top_k(top_k)
    result_columns = [
        *group_columns,
        "n_rows",
        "n_ranked",
        "mean_true_label_rank",
        "median_true_label_rank",
        *(f"top_{k}_accuracy" for k in top_k),
    ]
    if rank_column not in predictions.columns:
        return pd.DataFrame(columns=result_columns)

    rows: list[dict[str, object]] = []
    for group_key, group in _iter_groups(predictions, group_columns):
        row = _group_row(group_columns, group_key)
        ranks = pd.to_numeric(group[rank_column], errors="coerce")
        finite_ranks = ranks[np.isfinite(ranks)]
        row["n_rows"] = int(len(group))
        row["n_ranked"] = int(finite_ranks.size)
        row["mean_true_label_rank"] = float(finite_ranks.mean()) if finite_ranks.size else np.nan
        row["median_true_label_rank"] = float(finite_ranks.median()) if finite_ranks.size else np.nan
        for k in top_k:
            hit_column = f"top_{k}_hit"
            if hit_column in group.columns:
                hits = group[hit_column].fillna(False).astype(bool)
            else:
                hits = np.isfinite(ranks) & (ranks <= k)
            row[f"top_{k}_accuracy"] = float(np.mean(hits)) if len(group) else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    if group_columns and not result.empty:
        result = result.sort_values(group_columns, kind="mergesort")
    return result[result_columns].reset_index(drop=True)


def write_diagnostic_exports(
    predictions: pd.DataFrame,
    out_dir: str | Path,
    *,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    participant_column: str | None = None,
    group_columns: Sequence[str] | str | None = None,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    metadata_frame: pd.DataFrame | None = None,
    metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
    label_prefix: str = "label",
    include_confusion_pairs: bool = True,
    filenames: Mapping[str, str] | None = None,
) -> DiagnosticTables:
    """Write diagnostic prediction/summary tables to CSV files."""

    tables = diagnostic_summary_tables(
        predictions,
        true_column=true_column,
        predicted_column=predicted_column,
        participant_column=participant_column,
        group_columns=group_columns,
        top_k=top_k,
        metadata_frame=metadata_frame,
        metadata_label_columns=metadata_label_columns,
        label_prefix=label_prefix,
        include_confusion_pairs=include_confusion_pairs,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename_map = {**DEFAULT_DIAGNOSTIC_EXPORT_FILENAMES, **dict(filenames or {})}
    for table_name, frame in tables.as_dict().items():
        if table_name == "confusion_pairs" and not include_confusion_pairs:
            continue
        frame.to_csv(out_path / filename_map[table_name], index=False)
    return tables


def _normalize_columns(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")


def _require_distinct_columns(*columns: str) -> None:
    cleaned = [column for column in columns if column]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Output column names must be distinct.")


def _as_1d_array(name: str, values: Sequence | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array.ravel()


def _as_score_matrix(scores: Sequence[Sequence[float]] | np.ndarray, classes: Sequence | np.ndarray, *, n_samples: int) -> np.ndarray:
    score_matrix = np.asarray(scores, dtype=float)
    class_order = np.asarray(classes).ravel()
    if score_matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix.")
    if score_matrix.shape[0] != n_samples:
        raise ValueError("scores and true_labels must contain the same samples.")
    if score_matrix.shape[1] != class_order.size:
        raise ValueError("scores columns must match classes.")
    if class_order.size == 0:
        raise ValueError("classes must contain at least one class label.")
    return score_matrix


def _normalize_top_k(top_k: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(k) for k in top_k))
    if not normalized:
        raise ValueError("top_k must contain at least one value.")
    if any(k < 1 for k in normalized):
        raise ValueError("top_k values must be positive.")
    return normalized


def _add_group_values(frame: pd.DataFrame, group_values: Mapping[str, object] | None, n_samples: int) -> pd.DataFrame:
    if not group_values:
        return frame
    result = frame.copy()
    for column, value in group_values.items():
        column_name = str(column)
        if column_name in result.columns:
            raise ValueError(f"Group column '{column_name}' conflicts with an existing output column.")
        if _is_scalar_group_value(value):
            result[column_name] = value
            continue
        values = _as_1d_array(column_name, value)  # type: ignore[arg-type]
        if values.shape[0] != n_samples:
            raise ValueError(f"Group column '{column_name}' must contain {n_samples} values.")
        result[column_name] = values
    return result


def _is_scalar_group_value(value: object) -> bool:
    if isinstance(value, str) or value is None:
        return True
    return np.isscalar(value)


def _score_column_names(classes: np.ndarray, prefix: str) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for index, label in enumerate(classes):
        suffix = _safe_column_suffix(label, fallback=f"class_{index}")
        count = counts.get(suffix, 0)
        counts[suffix] = count + 1
        if count:
            suffix = f"{suffix}_{count + 1}"
        names.append(f"{prefix}{suffix}")
    return names


def _safe_column_suffix(value: object, *, fallback: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\W+", "_", text).strip("_")
    return text or fallback


def _iter_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    yield from frame.groupby(grouper, dropna=False, sort=True)


def _group_row(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if not group_columns:
        return {}
    if len(group_columns) == 1 and not isinstance(group_key, tuple):
        group_key = (group_key,)
    return dict(zip(group_columns, group_key))
