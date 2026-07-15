"""Runtime patches for paired-statistics reporting robustness."""

from __future__ import annotations

from functools import wraps
import re

import numpy as np
import pandas as pd

from neureptrace._object_label_utils import values_equal


_REQUIRED_COLUMNS = {"decoder_a_mean", "decoder_b_mean", "better_decoder_by_mean"}


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


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _identifier_values_equal(left: object, right: object, *, missing_equivalent: bool) -> bool:
    if missing_equivalent:
        left_missing = _is_missing_scalar(left)
        right_missing = _is_missing_scalar(right)
        if left_missing or right_missing:
            return left_missing and right_missing
    return values_equal(left, right)


def _safe_sort_key(value: object) -> tuple[str, str, str]:
    try:
        rendered = str(value)
    except Exception:  # pragma: no cover - only exotic identifiers fail during rendering
        rendered = f"<{type(value).__module__}.{type(value).__qualname__}>"
    return rendered, type(value).__module__, type(value).__qualname__


def _encode_identifier_column(
    frame: pd.DataFrame,
    column: str,
    *,
    missing_equivalent: bool,
) -> dict[str, object]:
    """Replace one identifier column with collision-safe temporary strings."""
    values = frame[column].to_numpy(dtype=object).tolist()
    representatives: list[object] = []
    for value in values:
        if not any(
            _identifier_values_equal(value, representative, missing_equivalent=missing_equivalent)
            for representative in representatives
        ):
            representatives.append(value)
    representatives.sort(key=_safe_sort_key)

    encoded = np.empty(len(values), dtype=object)
    mapping: dict[str, object] = {}
    for index, representative in enumerate(representatives):
        token = f"__neureptrace_paired_{column}_{index:012d}__"
        mask = np.asarray(
            [
                _identifier_values_equal(value, representative, missing_equivalent=missing_equivalent)
                for value in values
            ],
            dtype=bool,
        )
        encoded[mask] = token
        mapping[token] = representative
    frame[column] = pd.Series(encoded, index=frame.index, dtype=object)
    return mapping


def _decode_identifier_column(frame: pd.DataFrame, column: str, mapping: dict[str, object]) -> None:
    if column not in frame.columns:
        return
    decoded = np.empty(len(frame), dtype=object)
    for index, value in enumerate(frame[column].tolist()):
        decoded[index] = mapping.get(value, value)
    frame[column] = pd.Series(decoded, index=frame.index, dtype=object)


def install() -> None:
    """Install exact identifier handling, unbiased ties, and robust Markdown rendering."""
    from . import _emission_compare_empty_pairs_patch

    _emission_compare_empty_pairs_patch.install()

    import neureptrace.paired_stats as paired_stats

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
            if not {"decoder", "subject"}.issubset(subject_metrics.columns):
                statistics = original_paired_decoder_statistics(
                    subject_metrics,
                    metrics=metrics,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )
                return _mark_exact_mean_ties(statistics)

            encoded = paired_stats._normalise_emission_mode(subject_metrics)
            pairing_columns = paired_stats._paired_statistic_group_columns(encoded)
            mappings: dict[str, dict[str, object]] = {}
            for column in pairing_columns:
                mappings[column] = _encode_identifier_column(encoded, column, missing_equivalent=True)
            mappings["decoder"] = _encode_identifier_column(encoded, "decoder", missing_equivalent=False)
            mappings["subject"] = _encode_identifier_column(encoded, "subject", missing_equivalent=False)

            statistics = original_paired_decoder_statistics(
                encoded,
                metrics=metrics,
                n_permutations=n_permutations,
                random_state=random_state,
            )
            for column in pairing_columns:
                _decode_identifier_column(statistics, column, mappings[column])
            for column in ("decoder_a", "decoder_b", "better_decoder_by_mean"):
                _decode_identifier_column(statistics, column, mappings["decoder"])
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
