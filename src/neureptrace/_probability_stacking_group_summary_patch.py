"""Runtime patches for probability-stacking metric summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

_INSTALLED = False
_ORIGINAL_SUMMARIZE_STACKED_METRICS = None


def _stable_top_k_positions(probabilities: np.ndarray, *, k: int) -> np.ndarray:
    """Return top-k column positions with deterministic low-index tie breaks."""

    return np.argsort(-probabilities, axis=1, kind="mergesort")[:, :k]


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Return top-k accuracy with the same tie rule as core metrics."""

    from neureptrace import probability_stacking as ps

    if len(labels) == 0:
        return float("nan")
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    effective_k = min(ps._validate_positive_integer(k, name="k"), probabilities.shape[1])
    top_columns = _stable_top_k_positions(probabilities, k=effective_k)
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _top_k_accuracy_from_label_values(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_values,
    *,
    k: int,
) -> float:
    """Return top-k accuracy for arbitrary label ids with stable tie breaks."""

    from neureptrace import probability_stacking as ps

    probabilities = np.asarray(probabilities, dtype=float)
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)
    label_values_array = np.asarray(label_values, dtype=int)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")
    if probabilities.shape[0] != true_labels.shape[0]:
        raise ValueError("probabilities and true_labels must contain the same number of rows.")
    if probabilities.shape[1] != label_values_array.shape[0]:
        raise ValueError("label_values must contain one label per probability column.")
    effective_k = min(ps._validate_positive_integer(k, name="k"), probabilities.shape[1])
    top_positions = _stable_top_k_positions(probabilities, k=effective_k)
    top_labels = label_values_array[top_positions]
    return float(np.mean(np.any(top_labels == true_labels[:, None], axis=1)))


def _summarize_global_metrics(ps, observations: pd.DataFrame) -> pd.DataFrame:
    """Return one global metrics row when no metric grouping columns exist."""

    prob_columns = ps.probability_columns(observations)
    if "true_label" not in observations.columns or not prob_columns:
        raise ValueError("Stacked observations must contain true_label and prob_class_* columns.")

    label_values = ps._label_values(prob_columns)
    label_value_set = set(label_values)
    probabilities = ps._validate_probability_matrix(
        observations.loc[:, list(prob_columns)].to_numpy(dtype=float),
        context="Probability values for global metric summary",
    )
    true_label_values = ps._integer_label_array(observations["true_label"], name="true_label")
    missing_labels = sorted(set(int(label) for label in true_label_values if int(label) not in label_value_set))
    if missing_labels:
        raise ValueError(f"true_label values must index probability labels {list(label_values)}; missing labels: {missing_labels[:5]}")

    true_positions = ps._label_positions(true_label_values, label_values)
    predicted_label_values = np.asarray([label_values[position] for position in probabilities.argmax(axis=1)], dtype=int)
    row: dict[str, object] = {
        "accuracy": float(ps.accuracy_score(true_label_values, predicted_label_values)),
        "balanced_accuracy": float(ps.balanced_accuracy_score(true_label_values, predicted_label_values)),
        "top2_accuracy": ps._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=2),
        "top3_accuracy": ps._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=3),
        "log_loss": float(ps.log_loss(true_label_values, probabilities, labels=list(label_values))),
        "brier": float(ps.brier_score_multiclass(probabilities, true_positions)),
        "ece": float(ps.expected_calibration_error(probabilities, true_positions)),
        "n_test": int(len(observations)),
        "n_classes": int(len(prob_columns)),
    }
    for column in ps._STACKING_METADATA_COLUMNS:
        if column not in observations.columns:
            row[column] = ""
            continue
        values = observations[column].drop_duplicates()
        if len(values) > 1:
            raise ValueError(f"Stacking metadata column {column!r} is inconsistent within the global metric summary.")
        row[column] = "" if values.empty else values.iloc[0]
    return pd.DataFrame([row])


def install() -> None:
    """Install probability-stacking metric wrappers once."""

    global _INSTALLED, _ORIGINAL_SUMMARIZE_STACKED_METRICS
    if _INSTALLED:
        return

    from neureptrace import probability_stacking as ps

    ps._top_k_accuracy = _top_k_accuracy
    ps._top_k_accuracy_from_label_values = _top_k_accuracy_from_label_values
    _ORIGINAL_SUMMARIZE_STACKED_METRICS = ps.summarize_stacked_metrics

    def _summarize_stacked_metrics(observations: pd.DataFrame) -> pd.DataFrame:
        group_columns = [column for column in ps._METRIC_GROUP_COLUMNS if column in observations.columns]
        if group_columns:
            return _ORIGINAL_SUMMARIZE_STACKED_METRICS(observations)
        return _summarize_global_metrics(ps, observations)

    _summarize_stacked_metrics.__name__ = _ORIGINAL_SUMMARIZE_STACKED_METRICS.__name__
    _summarize_stacked_metrics.__doc__ = _ORIGINAL_SUMMARIZE_STACKED_METRICS.__doc__
    ps.summarize_stacked_metrics = _summarize_stacked_metrics
    _INSTALLED = True
