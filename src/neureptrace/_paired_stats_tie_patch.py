"""Runtime patches for paired-statistics input and reporting robustness."""

from __future__ import annotations

from collections import Counter
from functools import wraps
import re

import numpy as np
import pandas as pd

from neureptrace._object_label_utils import values_equal


_REQUIRED_COLUMNS = {"decoder_a_mean", "decoder_b_mean", "better_decoder_by_mean"}
_COMPLEX_DIFFERENCES_ERROR = "differences must contain only real values."
_BOOLEAN_DIFFERENCES_ERROR = "differences must not contain boolean values."


def _validate_unique_metric_names(metrics: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return metric names as a tuple after rejecting repetitions."""
    if metrics is None:
        return None
    metric_names = tuple(metrics)
    duplicates = sorted(name for name, count in Counter(metric_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"metrics must not contain duplicate names: {duplicates}")
    return metric_names


def _validate_complete_pairing_identifiers(subject_metrics: pd.DataFrame) -> None:
    """Reject missing or blank decoder/subject identifiers before string conversion."""

    invalid_rows: dict[str, list[object]] = {}
    for column in ("decoder", "subject"):
        if column not in subject_metrics.columns:
            continue
        values = subject_metrics[column]
        invalid = values.isna() | values.map(
            lambda value: isinstance(value, str) and value.strip() == ""
        )
        if bool(invalid.any()):
            invalid_rows[column] = invalid[invalid].index.tolist()[:5]
    if invalid_rows:
        raise ValueError(
            "Subject metrics must not contain missing decoder or subject identifiers; "
            f"blank strings are also invalid. Rows: {invalid_rows}"
        )


def _validate_identifier_string_collisions(subject_metrics: pd.DataFrame) -> None:
    """Reject distinct identifiers that collapse to the same normalized string."""

    for column in ("decoder", "subject"):
        if column not in subject_metrics.columns:
            continue
        seen: dict[str, tuple[object, object]] = {}
        for row_index, value in subject_metrics[column].items():
            normalized = str(value)
            previous = seen.get(normalized)
            if previous is None:
                seen[normalized] = (row_index, value)
                continue
            previous_index, previous_value = previous
            if values_equal(previous_value, value):
                continue
            raise ValueError(
                f"Subject metrics contain ambiguous {column} identifiers: "
                f"{previous_value!r} at row {previous_index!r} and {value!r} at row {row_index!r} "
                f"both normalize to {normalized!r}."
            )


def _is_boolean_scalar(value: object) -> bool:
    """Return whether a scalar-like value uses a Boolean dtype."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and isinstance(value.item(), (bool, np.bool_))
    return False


def _boolean_value_mask(values: pd.Series) -> pd.Series:
    """Return a row-aligned mask for Boolean scalar values."""
    return values.map(_is_boolean_scalar).fillna(False).astype(bool)


def _validate_non_boolean_metric_values(
    subject_metrics: pd.DataFrame,
    metric_names: tuple[str, ...],
) -> None:
    """Reject Boolean metrics before float conversion changes them to zero or one."""
    for metric in metric_names:
        if metric not in subject_metrics.columns:
            continue
        boolean_values = _boolean_value_mask(subject_metrics[metric])
        if boolean_values.any():
            bad_rows = boolean_values[boolean_values].index.tolist()[:5]
            raise ValueError(
                f"Subject metrics contain boolean values in metric '{metric}' at row(s) {bad_rows}; "
                "paired statistics require real-valued, non-boolean metrics."
            )


def _is_complex_scalar(value: object) -> bool:
    """Return whether a scalar-like value uses a complex numeric dtype."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.ndim == 0 and bool(np.iscomplexobj(array))


def _complex_value_mask(values: pd.Series) -> pd.Series:
    """Return a row-aligned mask for complex scalar values."""
    if np.iscomplexobj(values.dtype):
        return pd.Series(True, index=values.index, dtype=bool)
    return values.map(_is_complex_scalar).fillna(False).astype(bool)


def _validate_real_metric_values(
    subject_metrics: pd.DataFrame,
    metric_names: tuple[str, ...],
) -> None:
    """Reject complex metrics before pandas can discard imaginary components."""
    for metric in metric_names:
        if metric not in subject_metrics.columns:
            continue
        complex_values = _complex_value_mask(subject_metrics[metric])
        if complex_values.any():
            bad_rows = complex_values[complex_values].index.tolist()[:5]
            raise ValueError(
                f"Subject metrics contain complex values in metric '{metric}' at row(s) {bad_rows}; "
                "paired statistics require real-valued metrics."
            )


def _validate_non_boolean_differences(differences: object) -> None:
    """Reject Boolean sign-flip inputs before numeric coercion."""
    try:
        values = np.asarray(differences)
    except (TypeError, ValueError) as exc:
        raise ValueError(_BOOLEAN_DIFFERENCES_ERROR) from exc
    if any(_is_boolean_scalar(value) for value in values.flat):
        raise ValueError(_BOOLEAN_DIFFERENCES_ERROR)


def _validate_real_differences(differences: object) -> None:
    """Reject complex sign-flip inputs before real-valued statistics are computed."""
    try:
        values = np.asarray(differences)
    except (TypeError, ValueError) as exc:
        raise ValueError(_COMPLEX_DIFFERENCES_ERROR) from exc
    if np.iscomplexobj(values):
        raise ValueError(_COMPLEX_DIFFERENCES_ERROR)
    if values.dtype == object and any(_is_complex_scalar(value) for value in values.flat):
        raise ValueError(_COMPLEX_DIFFERENCES_ERROR)


def _overflow_safe_sign_flip_differences(differences: object) -> object:
    """Scale finite one-dimensional effects only when their reductions can overflow."""

    try:
        values = np.asarray(differences)
    except (TypeError, ValueError):
        return differences
    if values.ndim != 1 or values.size == 0 or values.dtype.kind in {"S", "U"}:
        return values
    try:
        numeric = values.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError):
        return values
    if not np.all(np.isfinite(numeric)):
        return numeric

    max_abs = float(np.max(np.abs(numeric)))
    if max_abs == 0.0 or max_abs <= np.finfo(float).max / numeric.size:
        return numeric
    return numeric / max_abs


def _mark_exact_mean_ties(statistics: pd.DataFrame) -> pd.DataFrame:
    """Return statistics with exact mean ties labeled explicitly."""
    if not _REQUIRED_COLUMNS.issubset(statistics.columns):
        return statistics

    tied = statistics["decoder_a_mean"].to_numpy(dtype=float) == statistics["decoder_b_mean"].to_numpy(dtype=float)
    if not np.any(tied):
        return statistics

    corrected = statistics.copy()
    corrected.loc[tied, "better_decoder_by_mean"] = "tie"
    return corrected


def install() -> None:
    """Install input validation, unbiased tie handling, and robust Markdown rendering."""
    from . import _emission_compare_empty_pairs_patch

    _emission_compare_empty_pairs_patch.install()

    import neureptrace.paired_stats as paired_stats

    if not getattr(paired_stats.sign_flip_p_value, "_paired_stats_real_differences_patched", False):
        original_sign_flip_p_value = paired_stats.sign_flip_p_value

        @wraps(original_sign_flip_p_value)
        def sign_flip_p_value(
            differences: np.ndarray,
            *,
            n_permutations: int = 10_000,
            random_state: int = 13,
        ) -> float:
            _validate_non_boolean_differences(differences)
            _validate_real_differences(differences)
            safe_differences = _overflow_safe_sign_flip_differences(differences)
            return original_sign_flip_p_value(
                safe_differences,
                n_permutations=n_permutations,
                random_state=random_state,
            )

        sign_flip_p_value._paired_stats_real_differences_patched = True  # type: ignore[attr-defined]
        paired_stats.sign_flip_p_value = sign_flip_p_value

    if not getattr(paired_stats.paired_decoder_statistics, "_paired_stats_tie_patched", False):
        original_paired_decoder_statistics = paired_stats.paired_decoder_statistics

        @wraps(original_paired_decoder_statistics)
        def paired_decoder_statistics(
            subject_metrics: pd.DataFrame,
            *,
            metrics: tuple[str, ...] | None = None,
            n_permutations: int = 10_000,
            random_state: int = 13,
        ) -> pd.DataFrame:
            metric_names = _validate_unique_metric_names(metrics)
            _validate_complete_pairing_identifiers(subject_metrics)
            _validate_identifier_string_collisions(subject_metrics)
            selected_metrics = (
                metric_names
                if metric_names is not None
                else tuple(metric for metric in paired_stats.METRIC_DIRECTIONS if metric in subject_metrics.columns)
            )
            _validate_non_boolean_metric_values(subject_metrics, selected_metrics)
            _validate_real_metric_values(subject_metrics, selected_metrics)
            statistics = original_paired_decoder_statistics(
                subject_metrics,
                metrics=metric_names,
                n_permutations=n_permutations,
                random_state=random_state,
            )
            return _mark_exact_mean_ties(statistics)

        paired_decoder_statistics._paired_stats_tie_patched = True  # type: ignore[attr-defined]
        paired_stats.paired_decoder_statistics = paired_decoder_statistics

    if not getattr(paired_stats._markdown_cell, "_paired_stats_carriage_return_patched", False):
        original_markdown_cell = paired_stats._markdown_cell

        @wraps(original_markdown_cell)
        def _markdown_cell(value: object) -> str:
            rendered = original_markdown_cell(value)
            return re.sub(r"\r ?", " ", rendered)

        _markdown_cell._paired_stats_carriage_return_patched = True  # type: ignore[attr-defined]
        paired_stats._markdown_cell = _markdown_cell


install()


__all__ = ["install"]