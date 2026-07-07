"""Runtime patches for paired-statistics reporting robustness."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {"decoder_a_mean", "decoder_b_mean", "better_decoder_by_mean"}
_MARKDOWN_CELL_PATCH_MARKER = "_paired_stats_markdown_cell_array_safe_patched"


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


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _patch_markdown_cell(paired_stats: Any) -> None:
    original_markdown_cell = paired_stats._markdown_cell
    if getattr(original_markdown_cell, _MARKDOWN_CELL_PATCH_MARKER, False):
        return

    @wraps(original_markdown_cell)
    def _markdown_cell(value: object) -> str:
        if _is_missing_scalar(value):
            return ""
        return str(value).replace("\n", " ").replace("|", r"\|")

    setattr(_markdown_cell, _MARKDOWN_CELL_PATCH_MARKER, True)
    paired_stats._markdown_cell = _markdown_cell


def install() -> None:
    """Install unbiased tie handling and robust Markdown cell rendering."""
    from . import _emission_compare_empty_pairs_patch

    _emission_compare_empty_pairs_patch.install()

    import neureptrace.paired_stats as paired_stats

    _patch_markdown_cell(paired_stats)

    if getattr(paired_stats.paired_decoder_statistics, "_paired_stats_tie_patched", False):
        return

    original_paired_decoder_statistics = paired_stats.paired_decoder_statistics

    @wraps(original_paired_decoder_statistics)
    def paired_decoder_statistics(
        subject_metrics: pd.DataFrame,
        *,
        metrics: tuple[str, ...] | None = None,
        n_permutations: int = 10_000,
        random_state: int = 13,
    ) -> pd.DataFrame:
        statistics = original_paired_decoder_statistics(
            subject_metrics,
            metrics=metrics,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        return _mark_exact_mean_ties(statistics)

    paired_decoder_statistics._paired_stats_tie_patched = True  # type: ignore[attr-defined]
    paired_stats.paired_decoder_statistics = paired_decoder_statistics


install()


__all__ = ["install"]
