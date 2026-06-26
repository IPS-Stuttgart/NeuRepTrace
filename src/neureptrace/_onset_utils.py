from __future__ import annotations

import glob
import numbers
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace._onset_constants import GROUP_COLUMNS
from neureptrace.temporal_model import _normalize_probabilities, probability_columns


def expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_COLUMNS if column in frame.columns]


def sequence_columns(frame: pd.DataFrame) -> list[str]:
    identifier = "sequence_id" if "sequence_id" in frame.columns else "sample_index" if "sample_index" in frame.columns else None
    if identifier is None:
        raise ValueError("Observation rows must contain sequence_id or sample_index.")
    return [column for column in ("subject", "fold", identifier) if column in frame.columns]


def window_mask(frame: pd.DataFrame, window: tuple[float, float]) -> pd.Series:
    start, stop = window
    return (frame["time"] >= start) & (frame["time"] <= stop)


def _is_boolean_label(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _integer_labels(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    boolean_values = values.map(_is_boolean_label).to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = ~boolean_values & np.isfinite(numeric) & (numeric == np.floor(numeric))
    labels = np.zeros(len(numeric), dtype=int)
    labels[valid] = numeric[valid].astype(int)
    return labels, valid


def _integer_label(value: object) -> int | None:
    if _is_boolean_label(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _confidence_values(frame: pd.DataFrame) -> pd.Series:
    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    if confidence.isna().any() or not np.isfinite(confidence.to_numpy(dtype=float)).all():
        raise ValueError("confidence values must be finite.")
    if bool(((confidence < 0.0) | (confidence > 1.0)).any()):
        raise ValueError("confidence values must lie in [0, 1].")
    return confidence


def score_values(frame: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column == "confidence" and score_column in frame.columns:
        return _confidence_values(frame)
    if score_column in frame.columns:
        return pd.to_numeric(frame[score_column], errors="coerce")
    prob_columns = probability_columns(frame)
    probabilities = _normalize_probabilities(frame[prob_columns].to_numpy(dtype=float))
    if score_column == "confidence":
        return pd.Series(probabilities.max(axis=1), index=frame.index)
    if score_column == "probability_true_class" and "true_label" in frame.columns:
        true_labels, valid_labels = _integer_labels(frame["true_label"])
        scores = np.full(len(frame), np.nan, dtype=float)
        in_bounds = valid_labels & (true_labels >= 0) & (true_labels < probabilities.shape[1])
        valid_positions = np.flatnonzero(in_bounds)
        scores[valid_positions] = probabilities[valid_positions, true_labels[valid_positions]]
        return pd.Series(scores, index=frame.index)
    raise ValueError(f"Score column '{score_column}' is missing and cannot be inferred.")


def class_lookup(frame: pd.DataFrame) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for column in frame.columns:
        if not column.startswith("class_"):
            continue
        try:
            class_index = int(column.removeprefix("class_"))
        except ValueError:
            continue
        values = frame[column].dropna()
        if not values.empty:
            lookup[class_index] = str(values.iloc[0])
    return lookup


def ensure_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "predicted_label" in frame.columns and "predicted_class" in frame.columns:
        return frame
    prob_columns = probability_columns(frame)
    probabilities = _normalize_probabilities(frame[prob_columns].to_numpy(dtype=float))
    predicted_labels = probabilities.argmax(axis=1)
    if "predicted_label" in frame.columns:
        parsed_labels, valid_labels = _integer_labels(frame["predicted_label"])
        predicted_labels[valid_labels] = parsed_labels[valid_labels]
    if "predicted_label" not in frame.columns:
        frame["predicted_label"] = predicted_labels
    if "predicted_class" not in frame.columns:
        lookup = class_lookup(frame)
        frame["predicted_class"] = [lookup.get(int(label), str(int(label))) for label in predicted_labels]
    return frame


def prediction_values(frame: pd.DataFrame) -> np.ndarray:
    if "predicted_label" in frame.columns:
        return frame["predicted_label"].to_numpy(dtype=object)
    if "predicted_class" in frame.columns:
        return frame["predicted_class"].to_numpy(dtype=object)
    return np.full(len(frame), None, dtype=object)


def prediction_value(row: pd.Series) -> object:
    if "predicted_label" in row and pd.notna(row["predicted_label"]):
        return row["predicted_label"]
    if "predicted_class" in row and pd.notna(row["predicted_class"]):
        return row["predicted_class"]
    return None


def sequence_identity(row: pd.Series) -> dict:
    identity = {"sequence_id": row.get("sequence_id", row.get("sample_index", np.nan))}
    for optional_column in ("sample_index", "group", "source_file"):
        if optional_column in row:
            identity[optional_column] = row[optional_column]
    for truth_column in ("true_label", "true_class"):
        if truth_column in row:
            identity[truth_column] = row[truth_column]
    return identity


def is_correct_detection(row: pd.Series) -> bool:
    if "true_label" in row and "predicted_label" in row and pd.notna(row["true_label"]) and pd.notna(row["predicted_label"]):
        true_label = _integer_label(row["true_label"])
        predicted_label = _integer_label(row["predicted_label"])
        return true_label is not None and predicted_label is not None and true_label == predicted_label
    if "true_class" in row and "predicted_class" in row and pd.notna(row["true_class"]) and pd.notna(row["predicted_class"]):
        return str(row["true_class"]) == str(row["predicted_class"])
    return False


def validate_detection_options(
    *,
    threshold_quantile: float | None = None,
    threshold_method: str | None = None,
    threshold_methods: tuple[str, ...] | None = None,
    min_consecutive: int = 1,
    min_duration: float | None = None,
) -> None:
    if threshold_quantile is not None:
        threshold_quantile_value = _validate_real_number(threshold_quantile, name="threshold_quantile")
        if not 0.0 <= threshold_quantile_value <= 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1.")
    if threshold_method is not None and threshold_methods is not None and threshold_method not in threshold_methods:
        raise ValueError(f"threshold_method must be one of {threshold_methods}.")
    _validate_integer(min_consecutive, name="min_consecutive", minimum=1)
    if min_duration is not None:
        min_duration_value = _validate_real_number(min_duration, name="min_duration")
        if min_duration_value < 0:
            raise ValueError("min_duration must be non-negative when provided.")


def _validate_real_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real-valued number.")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _validate_integer(value: object, *, name: str, minimum: int) -> int:
    numeric = _validate_real_number(value, name=name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    integer = int(numeric)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer
