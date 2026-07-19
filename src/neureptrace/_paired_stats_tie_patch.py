"""Runtime patches for paired-statistics input and reporting robustness."""

from __future__ import annotations

from collections import Counter
from functools import wraps
import re

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {"decoder_a_mean", "decoder_b_mean", "better_decoder_by_mean"}


def _validate_unique_metric_names(metrics: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return metric names as a tuple after rejecting repetitions."""
    if metrics is None:
        return None
    metric_names = tuple(metrics)
    duplicates = sorted(name for name, count in Counter(metric_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"metrics must not contain duplicate names: {duplicates}")
    return metric_names


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
    """Install metric validation, unbiased tie handling, and robust Markdown rendering."""
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
            metric_names = _validate_unique_metric_names(metrics)
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
