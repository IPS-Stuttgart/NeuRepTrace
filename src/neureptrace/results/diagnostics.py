"""Reusable prediction diagnostics for class-decoding workflows.

The functions in this module operate on the canonical NeuRepTrace
probability-observation schema, but they deliberately keep column names
configurable so project packages such as PyMEGDec can reuse them for
paper-facing prediction tables with domain-specific label names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from neureptrace.metrics import (
    confusion_category_enrichment,
    confusion_category_matrix,
    confusion_counts as metric_confusion_counts,
    confusion_pair_summary,
    per_class_accuracy,
    rank_class_scores,
)
from neureptrace.metrics.confusion import DEFAULT_METADATA_LABEL_COLUMNS
from neureptrace.observations import probability_columns as canonical_probability_columns

DEFAULT_RANK_TOP_K = (1, 2, 3)
DEFAULT_ROW_TOP_K = 3

DIAGNOSTIC_TABLES = (
    "predictions",
    "confusion",
    "per_class_recall",
    "confusion_pairs",
    "rank_summary",
    "category_enrichment",
    "category_matrix",
)

__all__ = [
    "DEFAULT_RANK_TOP_K",
    "DEFAULT_ROW_TOP_K",
    "DIAGNOSTIC_TABLES",
    "add_rank_diagnostics",
    "augment_prediction_ranks",
    "build_prediction_diagnostics",
    "category_confusion_summary",
    "confusion_counts",
    "metadata_conditioned_confusion_enrichment",
    "per_class_recall",
    "ranked_label_metrics",
    "true_label_ranks",
]


def build_prediction_diagnostics(  # pylint: disable=too-many-arguments
    frame: pd.DataFrame,
    *,
    true_column: str = "true_label",
    predicted_column: str = "predicted_label",
    group_columns: Sequence[str] | str | None = (),
    participant_column: str | None = None,
    metadata_frame: pd.DataFrame | None = None,
    metadata_label_columns: Sequence[str] = DEFAULT_METADATA_LABEL_COLUMNS,
    category_columns: Sequence[str] | str | None = None,
    label_prefix: str = "label",
    rank_top_k: Sequence[int] = DEFAULT_RANK_TOP_K,
    row_top_k: int = DEFAULT_ROW_TOP_K,
    class_values: Sequence[object] | Mapping[str, object] | None = None,
    n_permutations: int | None = 10_000,
    seed: int | None = 0,
) -> dict[str, pd.DataFrame]:
    """Build a standard bundle of prediction-diagnostic tables.

    Parameters
    ----------
    frame:
        Trial-level prediction table. For rank diagnostics, the table may also
        contain ``prob_class_*`` columns using the canonical observation schema.
    true_column / predicted_column:
        Columns holding true and predicted class labels. The default matches
        NeuRepTrace probability observations, while PyMEGDec-style tables can
        pass names such as ``true_stimulus`` and ``predicted_stimulus``.
    group_columns:
        Experimental-condition columns that should be kept separate in every
        diagnostic table, e.g. decoder, time window, or participant split.
    participant_column:
        Optional participant column used for participant counts in per-class
        recall and confusion-pair summaries.
    metadata_frame:
        Optional class/stimulus metadata. When supplied, confusion pairs are
        annotated and category-level enrichment/matrix tables are emitted.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary with the tables listed in :data:`DIAGNOSTIC_TABLES`.
    """

    group_columns = _normalize_columns(group_columns)
    ranked_predictions = augment_prediction_ranks(
        frame,
        true_column=true_column,
        top_k=rank_top_k,
        row_top_k=row_top_k,
        class_values=class_values,
    )

    diagnostics: dict[str, pd.DataFrame] = {
        "predictions": ranked_predictions,
        "confusion": metric_confusion_counts(
            ranked_predictions,
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=group_columns,
        ),
        "per_class_recall": per_class_recall(
            ranked_predictions,
            true_column=true_column,
            predicted_column=predicted_column,
            participant_column=participant_column,
            group_columns=group_columns,
        ),
        "confusion_pairs": confusion_pair_summary(
            ranked_predictions,
            true_column=true_column,
            predicted_column=predicted_column,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_frame=metadata_frame,
            metadata_label_columns=metadata_label_columns,
            label_prefix=label_prefix,
        ),
        "rank_summary": ranked_label_metrics(
            ranked_predictions,
            true_column=true_column,
            group_columns=group_columns,
            top_k=rank_top_k,
            class_values=class_values,
        ),
    }

    if metadata_frame is not None:
        diagnostics["category_enrichment"] = confusion_category_enrichment(
            ranked_predictions,
            metadata_frame=metadata_frame,
            true_column=true_column,
            predicted_column=predicted_column,
            category_columns=category_columns,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_label_columns=metadata_label_columns,
            n_permutations=n_permutations,
            seed=seed,
        )
        diagnostics["category_matrix"] = confusion_category_matrix(
            ranked_predictions,
            metadata_frame=metadata_frame,
            true_column=true_column,
            predicted_column=predicted_column,
            category_columns=category_columns,
            group_columns=group_columns,
            participant_column=participant_column,
            metadata_label_columns=metadata_label_columns,
        )
    else:
        diagnostics["category_enrichment"] = pd.DataFrame()
        diagnostics["category_matrix"] = pd.DataFrame()

    return {name: diagnostics[name] for name in DIAGNOSTIC_TABLES}


def augment_prediction_ranks(
    frame: pd.DataFrame,
    *,
    true_column: str = "true_label",
    top_k: Sequence[int] = DEFAULT_RANK_TOP_K,
    row_top_k: int = DEFAULT_ROW_TOP_K,
    class_values: Sequence[object] | Mapping[str, object] | None = None,
    class_column: str = "label",
) -> pd.DataFrame:
    """Return ``frame`` with per-row true-label rank and top-k hit columns.

    If no probability columns are present, the input rows are returned unchanged
    apart from ``top{k}_hit`` columns filled with ``NaN``. This makes the helper
    safe for prediction tables that contain hard labels only.
    """

    _require_columns(frame, [true_column])
    top_k = _normalize_top_k(top_k)
    result = frame.copy()
    prob_columns = tuple(canonical_probability_columns(result))
    if not prob_columns:
        for k in top_k:
            result[f"top{k}_hit"] = np.nan
        return result

    rank_result = _rank_class_scores_for_frame(
        result,
        true_column=true_column,
        top_k=top_k,
        row_top_k=row_top_k,
        class_values=class_values,
        class_column=class_column,
    )
    rank_rows = pd.DataFrame(rank_result["rows"], index=result.index)
    for column in rank_rows.columns:
        result[column] = rank_rows[column]

    true_label_rank = pd.to_numeric(result["true_label_rank"], errors="coerce")
    for k in top_k:
        result[f"top{k}_hit"] = (true_label_rank <= k).where(true_label_rank.notna(), np.nan)
    return result


def ranked_label_metrics(
    frame: pd.DataFrame,
    *,
    true_column: str = "true_label",
    ranked_columns: Sequence[str] | None = None,
    score_columns: Sequence[str] | Mapping[object, str] | None = None,
    group_columns: Sequence[str] | str | None = (),
    top_k: Sequence[int] = DEFAULT_RANK_TOP_K,
    top_ks: Sequence[int] | None = None,
    class_values: Sequence[object] | Mapping[str, object] | None = None,
    rank_column: str = "true_rank",
) -> pd.DataFrame:
    """Summarize true-label rank and top-k accuracy by condition group."""

    if true_column not in frame.columns and true_column == "true_label" and "true_class" in frame.columns:
        true_column = "true_class"
    if ranked_columns is not None or score_columns is not None or top_ks is not None:
        top_ks = _normalize_reporting_top_ks(DEFAULT_RANK_TOP_K if top_ks is None else top_ks)
        group_columns = _normalize_columns(group_columns)
        ranks = true_label_ranks(
            frame,
            true_column=true_column,
            ranked_columns=ranked_columns,
            score_columns=score_columns,
        )
        working = frame[[*group_columns]].copy()
        working[rank_column] = ranks
        rows: list[dict[str, object]] = []
        for group_key, group in _iter_frame_groups(working, group_columns):
            row = _group_row(group_columns, group_key)
            finite_ranks = pd.to_numeric(group[rank_column], errors="coerce")
            row["n"] = int(len(group))
            row["n_ranked"] = int(finite_ranks.notna().sum())
            row[f"{rank_column}_mean"] = float(finite_ranks.mean()) if row["n_ranked"] else np.nan
            row[f"{rank_column}_median"] = float(finite_ranks.median()) if row["n_ranked"] else np.nan
            for k in top_ks:
                row[f"top_{k}_accuracy"] = float((finite_ranks <= k).mean()) if len(group) else np.nan
                row[f"top_{k}_count"] = int((finite_ranks <= k).sum())
            rows.append(row)
        return _sort_frame(pd.DataFrame(rows), group_columns)

    _require_columns(frame, [true_column])
    group_columns = _normalize_columns(group_columns)
    _require_columns(frame, group_columns)
    top_k = _normalize_top_k(top_k)

    rows: list[dict[str, object]] = []
    for group_key, group in _iter_frame_groups(frame, group_columns):
        rank_result = _rank_class_scores_for_frame(
            group,
            true_column=true_column,
            top_k=top_k,
            row_top_k=0,
            class_values=class_values,
        )
        row = _group_row(group_columns, group_key)
        row["n_rows"] = int(len(group))
        for k in top_k:
            row[f"top{k}_accuracy"] = rank_result["top_k_accuracy"][k]
        row["mean_true_label_rank"] = rank_result["mean_true_label_rank"]
        row["median_true_label_rank"] = rank_result["median_true_label_rank"]
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def per_class_recall(
    frame: pd.DataFrame,
    *,
    true_column: str = "true_class",
    predicted_column: str = "predicted_class",
    participant_column: str | None = None,
    group_columns: Sequence[str] | str | None = (),
    class_column: str = "class_id",
) -> pd.DataFrame:
    """Return one-vs-rest recall for each true class.

    This wraps :func:`neureptrace.metrics.per_class_accuracy` but exposes the
    statistic using the standard classification term ``recall``. The original
    ``accuracy``/``n_trials`` columns are retained for backwards familiarity,
    and ``support`` is added as an alias for the number of true-class trials.
    """

    summary = per_class_accuracy(
        frame,
        true_column=true_column,
        predicted_column=predicted_column,
        participant_column=participant_column,
        group_columns=_normalize_columns(group_columns),
    ).copy()
    if summary.empty:
        return summary
    summary["support"] = summary["n_trials"]
    summary["n_true"] = summary["n_trials"]
    summary["recall"] = summary["accuracy"]
    summary["n_false_negative"] = summary["n_true"] - summary["n_correct"]
    if "true_label" in summary.columns and class_column not in summary.columns:
        summary[class_column] = summary["true_label"]
    ordered = [column for column in summary.columns if column not in {"support", "recall"}]
    insert_at = ordered.index("n_correct") + 1 if "n_correct" in ordered else len(ordered)
    return summary[[*ordered[:insert_at], "support", "recall", *ordered[insert_at:]]]


def confusion_counts(
    frame: pd.DataFrame,
    *,
    true_column: str = "true_class",
    predicted_column: str = "predicted_class",
    group_columns: Sequence[str] | str | None = None,
    count_column: str = "n",
    include_marginals: bool = True,
) -> pd.DataFrame:
    """Count true/predicted class pairs, optionally within groups."""

    group_columns = _normalize_columns(group_columns)
    _require_columns(frame, [*group_columns, true_column, predicted_column])

    keys = [*group_columns, true_column, predicted_column]
    counts = (
        frame.groupby(keys, dropna=False, sort=True)
        .size()
        .rename(count_column)
        .reset_index()
    )
    counts[count_column] = counts[count_column].astype(int)

    if not include_marginals:
        return counts

    row_totals = (
        counts.groupby([*group_columns, true_column], dropna=False, sort=False)[count_column]
        .sum()
        .rename("true_count")
        .reset_index()
    )
    column_totals = (
        counts.groupby([*group_columns, predicted_column], dropna=False, sort=False)[count_column]
        .sum()
        .rename("predicted_count")
        .reset_index()
    )
    totals = (
        counts.groupby(group_columns, dropna=False, sort=False)[count_column].sum().rename("group_count").reset_index()
        if group_columns
        else pd.DataFrame({"group_count": [int(counts[count_column].sum())]})
    )

    result = counts.merge(row_totals, on=[*group_columns, true_column], how="left", validate="many_to_one")
    result = result.merge(column_totals, on=[*group_columns, predicted_column], how="left", validate="many_to_one")
    result = result.merge(totals, on=group_columns, how="left", validate="many_to_one") if group_columns else result.assign(group_count=totals.loc[0, "group_count"])
    result["row_fraction"] = result[count_column] / result["true_count"].replace({0: np.nan})
    result["group_fraction"] = result[count_column] / result["group_count"].replace({0: np.nan})
    return _sort_frame(result, [*group_columns, true_column, predicted_column])


def add_rank_diagnostics(
    frame: pd.DataFrame,
    *,
    true_column: str = "true_class",
    ranked_columns: Sequence[str] | None = None,
    score_columns: Sequence[str] | Mapping[object, str] | None = None,
    rank_column: str = "true_rank",
    top_ks: Sequence[int] = DEFAULT_RANK_TOP_K,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with true-label rank and top-k indicator columns."""

    top_ks = _normalize_reporting_top_ks(top_ks)
    result = frame.copy()
    ranks = true_label_ranks(result, true_column=true_column, ranked_columns=ranked_columns, score_columns=score_columns)
    result[rank_column] = ranks
    for k in top_ks:
        result[f"true_in_top_{k}"] = ranks.le(k).fillna(False)
    return result


def true_label_ranks(
    frame: pd.DataFrame,
    *,
    true_column: str = "true_class",
    ranked_columns: Sequence[str] | None = None,
    score_columns: Sequence[str] | Mapping[object, str] | None = None,
) -> pd.Series:
    """Return one-based true-label ranks for each row."""

    if ranked_columns is None and score_columns is None:
        raise ValueError("Either ranked_columns or score_columns must be provided.")
    _require_columns(frame, [true_column])

    if ranked_columns is not None:
        ranked_columns = list(ranked_columns)
        _require_columns(frame, ranked_columns)
        return pd.Series(
            [
                _rank_from_values(true_value, row_values)
                for true_value, row_values in zip(frame[true_column], frame[ranked_columns].itertuples(index=False, name=None))
            ],
            index=frame.index,
            dtype="float64",
        )

    labels, columns = _score_labels_and_columns(score_columns)
    _require_columns(frame, columns)
    scores = frame[columns].apply(pd.to_numeric, errors="coerce")
    ranks: list[float] = []
    for true_value, (_, score_row) in zip(frame[true_column], scores.iterrows()):
        if pd.isna(true_value):
            ranks.append(np.nan)
            continue
        order = np.argsort(-score_row.to_numpy(dtype=float), kind="mergesort")
        ranked_labels = [labels[index] for index in order if np.isfinite(score_row.iloc[index])]
        ranks.append(_rank_from_values(true_value, ranked_labels))
    return pd.Series(ranks, index=frame.index, dtype="float64")


def metadata_conditioned_confusion_enrichment(
    confusions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    true_column: str = "true_class",
    predicted_column: str = "predicted_class",
    metadata_key: str = "class_id",
    category_columns: Sequence[str] | str,
    count_column: str = "n",
    group_columns: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Annotate confusion rows with true/predicted metadata and same-category flags."""

    group_columns = _normalize_columns(group_columns)
    category_columns = _normalize_columns(category_columns)
    _require_columns(confusions, [*group_columns, true_column, predicted_column, count_column])
    _require_columns(metadata, [metadata_key, *category_columns])

    true_metadata = metadata[[metadata_key, *category_columns]].rename(
        columns={metadata_key: true_column, **{column: f"true_{column}" for column in category_columns}}
    )
    predicted_metadata = metadata[[metadata_key, *category_columns]].rename(
        columns={metadata_key: predicted_column, **{column: f"predicted_{column}" for column in category_columns}}
    )

    enriched = confusions.merge(true_metadata, on=true_column, how="left", validate="many_to_one")
    enriched = enriched.merge(predicted_metadata, on=predicted_column, how="left", validate="many_to_one")
    for column in category_columns:
        enriched[f"same_{column}"] = enriched[f"true_{column}"].eq(enriched[f"predicted_{column}"])

    return _sort_frame(enriched, [*group_columns, true_column, predicted_column])


def category_confusion_summary(
    confusions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    true_column: str = "true_class",
    predicted_column: str = "predicted_class",
    metadata_key: str = "class_id",
    category_column: str = "category",
    count_column: str = "n",
    group_columns: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Aggregate class-level confusions to true/predicted metadata categories."""

    group_columns = _normalize_columns(group_columns)
    enriched = metadata_conditioned_confusion_enrichment(
        confusions,
        metadata,
        true_column=true_column,
        predicted_column=predicted_column,
        metadata_key=metadata_key,
        category_columns=(category_column,),
        count_column=count_column,
        group_columns=group_columns,
    )
    true_category = f"true_{category_column}"
    predicted_category = f"predicted_{category_column}"
    keys = [*group_columns, true_category, predicted_category]
    summary = enriched.groupby(keys, dropna=False, sort=True)[count_column].sum().reset_index()
    totals = (
        summary.groupby([*group_columns, true_category], dropna=False, sort=False)[count_column]
        .sum()
        .rename("true_category_count")
        .reset_index()
    )
    result = summary.merge(totals, on=[*group_columns, true_category], how="left", validate="many_to_one")
    result["category_row_fraction"] = result[count_column] / result["true_category_count"].replace({0: np.nan})
    return _sort_frame(result, keys)


def _rank_class_scores_for_frame(
    frame: pd.DataFrame,
    *,
    true_column: str,
    top_k: Sequence[int],
    row_top_k: int,
    class_values: Sequence[object] | Mapping[str, object] | None = None,
    class_column: str = "label",
) -> dict[str, object]:
    prob_columns = tuple(canonical_probability_columns(frame))
    y_true = frame[true_column].to_numpy()
    if not prob_columns:
        return rank_class_scores(None, None, y_true, top_k=top_k, row_top_k=row_top_k, class_column=class_column)
    scores = _numeric_score_matrix(frame, prob_columns)
    classes = _class_values_from_prob_columns(prob_columns, class_values=class_values)
    return rank_class_scores(scores, classes, y_true, top_k=top_k, row_top_k=row_top_k, class_column=class_column)


def _numeric_score_matrix(frame: pd.DataFrame, prob_columns: Sequence[str]) -> np.ndarray:
    scores = frame.loc[:, list(prob_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scores.ndim != 2:
        raise ValueError("Probability columns must form a two-dimensional score matrix.")
    if not np.isfinite(scores).all():
        raise ValueError("Probability columns must contain finite numeric values.")
    return scores


def _class_values_from_prob_columns(
    prob_columns: Sequence[str],
    *,
    class_values: Sequence[object] | Mapping[str, object] | None = None,
) -> np.ndarray:
    if class_values is None:
        return np.asarray([_label_from_probability_column(column) for column in prob_columns], dtype=object)
    if isinstance(class_values, Mapping):
        missing = [column for column in prob_columns if column not in class_values]
        if missing:
            raise ValueError(f"class_values mapping is missing probability columns: {missing}")
        return np.asarray([class_values[column] for column in prob_columns], dtype=object)
    values = list(class_values)
    if len(values) != len(prob_columns):
        raise ValueError("class_values must have one entry per probability column.")
    return np.asarray(values, dtype=object)


def _label_from_probability_column(column: str) -> object:
    suffix = str(column).removeprefix("prob_class_")
    return int(suffix) if suffix.isdigit() else suffix


def _normalize_columns(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(dict.fromkeys(columns))


def _normalize_top_k(top_k: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(k) for k in top_k)
    if not normalized:
        raise ValueError("top_k must contain at least one value.")
    if any(k < 1 for k in normalized):
        raise ValueError("top_k values must be positive.")
    return tuple(dict.fromkeys(normalized))


def _normalize_reporting_top_ks(top_ks: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(sorted({int(k) for k in top_ks}))
    if not parsed or any(k < 1 for k in parsed):
        raise ValueError("top_ks must contain positive integers.")
    return parsed


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")


def _sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if columns and not frame.empty:
        frame = frame.sort_values(list(columns), kind="mergesort")
    return frame.reset_index(drop=True)


def _rank_from_values(true_value: object, ranked_values: Sequence[object]) -> float:
    if pd.isna(true_value):
        return np.nan
    true_key = _label_key(true_value)
    for index, value in enumerate(ranked_values, start=1):
        if pd.isna(value):
            continue
        if _label_key(value) == true_key:
            return float(index)
    return np.nan


def _label_key(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(parsed) and parsed.is_integer():
        return int(parsed)
    return value


def _score_labels_and_columns(score_columns: Sequence[str] | Mapping[object, str] | None) -> tuple[list[object], list[str]]:
    if score_columns is None:
        raise ValueError("score_columns must not be None.")
    if isinstance(score_columns, Mapping):
        labels = list(score_columns.keys())
        columns = list(score_columns.values())
        return labels, columns

    columns = list(score_columns)
    labels: list[object] = []
    for column in columns:
        if column.startswith("prob_class_"):
            labels.append(_parse_label_suffix(column.removeprefix("prob_class_")))
        elif column.startswith("score_class_"):
            labels.append(_parse_label_suffix(column.removeprefix("score_class_")))
        else:
            labels.append(column)
    return labels, columns


def _parse_label_suffix(value: str) -> object:
    try:
        return int(value)
    except ValueError:
        return value


def _iter_frame_groups(frame: pd.DataFrame, group_columns: Sequence[str]):
    if not group_columns:
        yield (), frame
        return
    for group_key, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        if len(group_columns) == 1:
            group_key = (group_key,)
        yield tuple(group_key), group


def _group_row(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if len(group_columns) == 1 and not isinstance(group_key, tuple):
        group_key = (group_key,)
    return dict(zip(group_columns, group_key, strict=True))
