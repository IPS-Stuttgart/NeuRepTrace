"""Patch canonical probability-observation label validation.

Canonical observation rows use integer ``prob_class_<label>`` suffixes to connect
``true_label`` and ``predicted_label`` to probability columns.  Older validators
coerced these labels with ``int(...)`` inside consistency checks, which could
silently truncate fractional labels or crash on non-finite values.  This shim
keeps the public validator stable while reporting malformed labels as regular
validation errors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


_PATCH_MARKER = "_neureptrace_observation_label_patch_installed"


def _integer_label_series(observation_schema, frame: pd.DataFrame, column: str, issues: list) -> tuple[pd.Series, pd.Series]:
    labels = observation_schema._numeric_series(frame, column, issues, allow_nan=True)
    present = labels.notna()
    numeric_values = labels.to_numpy(dtype=float)
    finite = pd.Series(np.isfinite(numeric_values), index=labels.index)
    integral = pd.Series(False, index=labels.index)
    finite_indices = finite[finite].index
    if len(finite_indices):
        finite_values = labels.loc[finite_indices].to_numpy(dtype=float)
        integral.loc[finite_indices] = np.equal(finite_values, np.round(finite_values))

    raw_values = frame[column]
    boolean = raw_values.map(lambda value: isinstance(value, (bool, np.bool_))) & raw_values.notna()
    non_finite = present & ~finite & ~boolean
    non_integer = present & finite & ~integral

    for row_index, value in raw_values.loc[boolean].head(20).items():
        observation_schema._issue(
            issues,
            "error",
            "boolean_label",
            f"Column '{column}' must contain integer class labels, not booleans.",
            column=column,
            row=int(row_index),
            value=value,
        )
    for row_index, value in raw_values.loc[non_finite].head(20).items():
        observation_schema._issue(
            issues,
            "error",
            "non_finite_label",
            f"Column '{column}' must contain finite integer class labels when present.",
            column=column,
            row=int(row_index),
            value=value,
        )
    for row_index, value in raw_values.loc[non_integer].head(20).items():
        observation_schema._issue(
            issues,
            "error",
            "non_integer_label",
            f"Column '{column}' must contain integer labels when present.",
            column=column,
            row=int(row_index),
            value=value,
        )

    valid = present & finite & integral & ~boolean
    return labels, valid


def _ordered_numeric_labels(prob_columns: Sequence[str]) -> list[int | None]:
    ordered_labels: list[int | None] = []
    for column in prob_columns:
        suffix = column.removeprefix("prob_class_")
        ordered_labels.append(int(suffix) if suffix.isdigit() else None)
    return ordered_labels


def _make_safe_probability_consistency(observation_schema):
    def _validate_probability_consistency(frame: pd.DataFrame, probabilities: pd.DataFrame, prob_columns: Sequence[str], issues: list, *, tolerance: float) -> None:
        if probabilities.empty:
            return

        probability_values = probabilities.to_numpy(dtype=float)
        finite_row = np.isfinite(probability_values).any(axis=1)
        filled = np.where(np.isfinite(probability_values), probability_values, -np.inf)
        max_probabilities = filled.max(axis=1)
        max_probabilities[~finite_row] = np.nan
        argmax_positions = filled.argmax(axis=1)
        label_columns = observation_schema._numeric_label_to_probability_columns(prob_columns)
        ordered_labels = _ordered_numeric_labels(prob_columns)
        predicted_label_cache: tuple[pd.Series, pd.Series] | None = None

        def predicted_labels() -> tuple[pd.Series, pd.Series]:
            nonlocal predicted_label_cache
            if predicted_label_cache is None:
                predicted_label_cache = _integer_label_series(observation_schema, frame, "predicted_label", issues)
            return predicted_label_cache

        if "confidence" in frame.columns:
            confidence = observation_schema._numeric_series(frame, "confidence", issues, allow_nan=True)
            confidence_values = confidence.to_numpy(dtype=float)
            bad_confidence = confidence.notna() & finite_row & (np.abs(confidence_values - max_probabilities) > tolerance)
            for row_index, value in confidence.loc[bad_confidence].head(20).items():
                observation_schema._issue(
                    issues,
                    "error",
                    "confidence_probability_mismatch",
                    "Column 'confidence' must equal the maximum prob_class_* value within tolerance.",
                    column="confidence",
                    row=int(row_index),
                    value=float(value),
                )

        if "predicted_label" in frame.columns and all(label is not None for label in ordered_labels):
            predicted_label, valid_predicted_label = predicted_labels()
            expected = pd.Series([ordered_labels[position] for position in argmax_positions], index=frame.index, dtype=float)
            bad_prediction = predicted_label.notna() & valid_predicted_label & finite_row & (predicted_label.astype(float) != expected)
            for row_index, value in predicted_label.loc[bad_prediction].head(20).items():
                observation_schema._issue(
                    issues,
                    "error",
                    "predicted_label_probability_mismatch",
                    "Column 'predicted_label' must equal the argmax prob_class_* label.",
                    column="predicted_label",
                    row=int(row_index),
                    value=int(value),
                )

        if "probability_true_class" in frame.columns and "true_label" in frame.columns and label_columns:
            true_label, valid_true_label = _integer_label_series(observation_schema, frame, "true_label", issues)
            probability_true_class = observation_schema._numeric_series(frame, "probability_true_class", issues, allow_nan=True)
            for row_index, label_value in true_label.loc[valid_true_label].items():
                if pd.isna(probability_true_class.loc[row_index]):
                    continue
                label = int(label_value)
                column = label_columns.get(label)
                if column is None:
                    observation_schema._issue(
                        issues,
                        "error",
                        "missing_true_label_probability_column",
                        f"Column 'true_label' references class {label}, but 'prob_class_{label}' is missing.",
                        column="true_label",
                        row=int(row_index),
                        value=label,
                    )
                    continue
                expected_value = probabilities.loc[row_index, column]
                if pd.isna(expected_value):
                    observation_schema._issue(
                        issues,
                        "error",
                        "missing_true_label_probability_value",
                        "Column 'probability_true_class' is present, but the referenced prob_class_<true_label> value is missing.",
                        column=column,
                        row=int(row_index),
                        value=np.nan,
                    )
                    continue
                expected_probability = float(expected_value)
                observed_probability = float(probability_true_class.loc[row_index])
                if abs(observed_probability - expected_probability) > tolerance:
                    observation_schema._issue(
                        issues,
                        "error",
                        "true_probability_mismatch",
                        "Column 'probability_true_class' must match prob_class_<true_label> within tolerance.",
                        column="probability_true_class",
                        row=int(row_index),
                        value=observed_probability,
                    )

        if "predicted_class" in frame.columns and "predicted_label" in frame.columns:
            predicted_label, valid_predicted_label = predicted_labels()
            for row_index, label_value in predicted_label.loc[valid_predicted_label].items():
                class_column = f"class_{int(label_value)}"
                if class_column not in frame.columns or pd.isna(frame.loc[row_index, "predicted_class"]):
                    continue
                expected_class = str(frame.loc[row_index, class_column])
                observed_class = str(frame.loc[row_index, "predicted_class"])
                if observed_class != expected_class:
                    observation_schema._issue(
                        issues,
                        "error",
                        "predicted_class_mismatch",
                        "Column 'predicted_class' must match class_<predicted_label>.",
                        column="predicted_class",
                        row=int(row_index),
                        value=observed_class,
                    )

    return _validate_probability_consistency


def install() -> None:
    """Install stricter canonical label checks for probability observations."""

    from neureptrace import observation_schema

    if getattr(observation_schema, _PATCH_MARKER, False):
        return
    observation_schema._validate_probability_consistency = _make_safe_probability_consistency(observation_schema)
    setattr(observation_schema, _PATCH_MARKER, True)
