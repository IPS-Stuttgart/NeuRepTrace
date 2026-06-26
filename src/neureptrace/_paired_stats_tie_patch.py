"""Runtime patch for unbiased paired-statistics tie labels."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd


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


def install() -> None:
    """Install unbiased tie handling for paired decoder statistics."""
    import neureptrace.paired_stats as paired_stats

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


__all__ = ["install"]
