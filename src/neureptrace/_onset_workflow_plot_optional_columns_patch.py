"""Allow onset-summary plots with independently optional grouping columns."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_PATCH_MARKER = "_neureptrace_onset_workflow_plot_optional_columns_patch_installed"


def _plot_onset_summary(summary: pd.DataFrame, out_path: Path) -> Path:
    """Plot onset summaries without assuming optional columns occur together."""

    if summary.empty:
        raise ValueError("Cannot plot an empty onset summary.")
    required = {"task", "post_detection_latency_median", "false_alarm_rate", "post_zero_detected_rate"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Onset summary is missing required columns for plotting: {missing}")

    sort_columns = [column for column in ("task", "decoder", "emission_mode") if column in summary.columns]
    frame = summary.copy().sort_values(sort_columns)
    labels = frame["task"].astype(str)
    if "decoder" in frame.columns:
        labels = labels + "\n" + frame["decoder"].astype(str)
    if "emission_mode" in frame.columns:
        labels = labels + " / " + frame["emission_mode"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, 0.8 * len(frame)), 4.2))
    positions = range(len(frame))

    axes[0].bar(positions, frame["post_detection_latency_median"])
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("Median post-zero onset latency (s)")
    axes[0].set_title("Onset latency")
    axes[0].axhline(0.0, color="0.4", linewidth=1.0)
    axes[0].grid(axis="y", color="0.9", linewidth=0.8)

    width = 0.38
    axes[1].bar(
        [position - width / 2 for position in positions],
        frame["false_alarm_rate"],
        width=width,
        label="false alarm",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        frame["post_zero_detected_rate"],
        width=width,
        label="post-zero detected",
    )
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Detection quality")
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", color="0.9", linewidth=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def install() -> None:
    """Install the optional onset-summary grouping-column fix."""

    onset_workflow = importlib.import_module("neureptrace.onset_workflow")
    current = onset_workflow.plot_onset_summary
    if getattr(current, _PATCH_MARKER, False):
        return

    @wraps(current)
    def plot_onset_summary(summary: pd.DataFrame, out_path: Path) -> Path:
        return _plot_onset_summary(summary, out_path)

    setattr(plot_onset_summary, _PATCH_MARKER, True)
    onset_workflow.plot_onset_summary = plot_onset_summary


__all__ = ["install"]
