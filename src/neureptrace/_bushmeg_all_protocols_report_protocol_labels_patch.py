"""Preserve fractional protocol labels and tolerate legacy BUSH-MEG report schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import bushmeg_all_protocols_report as _report


def _numeric_protocol_categories(values: pd.Series) -> list[int | float]:
    numeric = pd.to_numeric(values, errors="coerce")
    categories: dict[float, int | float] = {}
    for value in numeric.dropna().to_numpy(dtype=float):
        if not np.isfinite(value):
            continue
        categories[float(value)] = int(value) if float(value).is_integer() else float(value)
    return [categories[key] for key in sorted(categories)]


def _format_protocol_label(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        numeric_value = float(numeric)
        if np.isfinite(numeric_value):
            return str(int(numeric_value)) if numeric_value.is_integer() else f"{numeric_value:g}"
    return str(value)


def _summary_with_method_family(summary: pd.DataFrame, method_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return ``summary`` with a method-family column required by report groupers."""

    if summary.empty:
        return summary.copy()

    normalized = summary.copy()
    if "method_family" not in normalized.columns:
        normalized["method_family"] = pd.NA
    else:
        normalized["method_family"] = normalized["method_family"].replace("", pd.NA)

    if (
        method_metadata is not None
        and not method_metadata.empty
        and "method" in normalized.columns
        and {"method", "method_family"}.issubset(method_metadata.columns)
    ):
        family_by_method = (
            method_metadata.dropna(subset=["method"])
            .assign(_method_key=lambda frame: frame["method"].astype(str))
            .drop_duplicates("_method_key")
            .set_index("_method_key")["method_family"]
        )
        mapped_family = normalized["method"].astype(str).map(family_by_method)
        normalized["method_family"] = normalized["method_family"].where(normalized["method_family"].notna(), mapped_family)

    normalized["method_family"] = normalized["method_family"].fillna("unknown")
    return normalized


def _build_protocol_summary(summary: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance and method counts by protocol without truncating fractional protocol IDs."""

    rows: list[dict[str, Any]] = []
    protocol_values = (
        _numeric_protocol_categories(leaderboard["protocol_category"])
        if "protocol_category" in leaderboard.columns
        else []
    )
    summary_protocols = pd.to_numeric(summary["protocol_category"], errors="coerce") if "protocol_category" in summary.columns else pd.Series(dtype=float)
    leaderboard_protocols = (
        pd.to_numeric(leaderboard["protocol_category"], errors="coerce") if "protocol_category" in leaderboard.columns else pd.Series(dtype=float)
    )
    for protocol_category in protocol_values:
        protocol_numeric = float(protocol_category)
        subject_rows = summary.loc[summary_protocols == protocol_numeric] if "protocol_category" in summary.columns else pd.DataFrame()
        method_rows = leaderboard.loc[leaderboard_protocols == protocol_numeric] if "protocol_category" in leaderboard.columns else pd.DataFrame()
        rows.append(
            {
                "protocol_category": protocol_category,
                "n_methods": int(method_rows["method"].nunique()),
                "n_runnable_methods": int(method_rows.loc[method_rows["n_rows"] > 0, "method"].nunique()),
                "n_skipped_methods": int(method_rows.loc[method_rows["n_skipped"] > 0, "method"].nunique()),
                "n_subjects": int(subject_rows["outer_test_subject"].nunique()) if "outer_test_subject" in subject_rows.columns else 0,
                "n_rows": int(len(subject_rows)),
                "mean_balanced_accuracy": _report._numeric_mean(subject_rows.get("balanced_accuracy", pd.Series(dtype=float))),
                "sem_balanced_accuracy": _report._numeric_sem(subject_rows.get("balanced_accuracy", pd.Series(dtype=float))),
                "mean_accuracy": _report._numeric_mean(subject_rows.get("accuracy", pd.Series(dtype=float))),
                "mean_log_loss": _report._numeric_mean(subject_rows.get("log_loss", pd.Series(dtype=float))),
                "mean_brier": _report._numeric_mean(subject_rows.get("brier", pd.Series(dtype=float))),
                "mean_ece": _report._numeric_mean(subject_rows.get("ece", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _plot_balanced_accuracy_by_protocol(protocol_summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _report.plt.subplots(figsize=(7, 4.5))
    if protocol_summary.empty:
        ax.text(0.5, 0.5, "No protocol result rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = protocol_summary.copy()
        data["_protocol_sort"] = pd.to_numeric(data["protocol_category"], errors="coerce")
        data = data.sort_values("_protocol_sort", na_position="last").drop(columns="_protocol_sort")
        labels = [f"P{_format_protocol_label(protocol)}" for protocol in data["protocol_category"]]
        x = np.arange(len(data))
        yerr = pd.to_numeric(data["sem_balanced_accuracy"], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, data["mean_balanced_accuracy"].astype(float), yerr=yerr, color="C0", alpha=0.82)
        ax.set_xticks(x, labels=labels)
        ax.set_ylabel("Mean balanced accuracy")
        ax.set_xlabel("Protocol category")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _report.plt.close(fig)


def _write_markdown_report(
    path: Path,
    leaderboard: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    skipped_methods: pd.DataFrame,
    protocol3_kshot_leaderboard: pd.DataFrame,
    protocol3_by_k: pd.DataFrame,
) -> None:
    lines = [
        "# BUSH-MEG All-Protocols Report",
        "",
        f"- Methods in leaderboard: {leaderboard['method'].nunique() if 'method' in leaderboard.columns else 0}",
        f"- Skipped methods: {len(skipped_methods)}",
        "",
        "## Top Methods",
        "",
    ]
    runnable = leaderboard.loc[leaderboard["n_rows"] > 0].copy()
    if runnable.empty:
        lines.append("No runnable result rows were found.")
    else:
        for _, row in runnable.sort_values("mean_balanced_accuracy", ascending=False).head(10).iterrows():
            lines.append(
                f"- P{_format_protocol_label(row['protocol_category'])} `{row['method']}`: "
                f"{_report._format_percent(row['mean_balanced_accuracy'])} mean BA across {int(row['n_subjects'])} subject(s)"
            )
    lines.extend(["", "## Protocol Summary", ""])
    if protocol_summary.empty:
        lines.append("No protocol summary rows were available.")
    else:
        protocol_summary = protocol_summary.copy()
        protocol_summary["_protocol_sort"] = pd.to_numeric(protocol_summary["protocol_category"], errors="coerce")
        for _, row in protocol_summary.sort_values("_protocol_sort", na_position="last").drop(columns="_protocol_sort").iterrows():
            lines.append(
                f"- P{_format_protocol_label(row['protocol_category'])}: {_report._format_percent(row['mean_balanced_accuracy'])} mean BA, "
                f"{int(row['n_runnable_methods'])} runnable method(s), {int(row['n_skipped_methods'])} skipped method(s)"
            )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Protocol 1 rows are strict source-only when `valid_for_strict_source_only` is true.",
            "- Protocol 2 rows may use unlabeled target features but not target labels for fitting.",
            "- Protocol 3 rows are calibrated target-label methods and are reported separately from zero-calibration Protocols 1/2.",
            "- Protocol 4 rows are oracle/debug upper bounds and should not be mixed into benchmark claims.",
            "",
        ]
    )
    lines.extend(["", "## Protocol 3 K-Shot Summary", ""])
    if protocol3_by_k.empty:
        lines.append("No Protocol 3 k-shot rows were available.")
    else:
        for _, row in protocol3_by_k.iterrows():
            lines.append(
                f"- k={int(row['k_per_class']) if pd.notna(row['k_per_class']) else 'NA'}: "
                f"{_report._format_percent(row['mean_balanced_accuracy'])} mean BA, "
                f"delta vs best P1 {_report._format_percent(row['mean_delta_vs_best_protocol1'])}, "
                f"{int(row['n_methods'])} method(s)"
            )
    lines.extend(["", "## Protocol 3 Top Calibrated Methods", ""])
    if protocol3_kshot_leaderboard.empty:
        lines.append("No Protocol 3 calibrated leaderboard rows were available.")
    else:
        for _, row in protocol3_kshot_leaderboard.sort_values("mean_balanced_accuracy", ascending=False).head(10).iterrows():
            lines.append(
                f"- `{row['method']}`: {_report._format_percent(row['mean_balanced_accuracy'])} mean BA, "
                f"delta vs best P1 {_report._format_percent(row['mean_delta_vs_best_protocol1'])}, "
                f"n_eval_trials={int(row['n_eval_trials'])}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def install() -> None:
    if getattr(_report, "_fractional_protocol_report_patch_installed", False):
        return

    original_build_leaderboard = _report.build_leaderboard
    original_build_subject_summary = _report.build_subject_summary
    original_build_protocol3_kshot_leaderboard = _report.build_protocol3_kshot_leaderboard

    def build_leaderboard(summary: pd.DataFrame, method_metadata: pd.DataFrame) -> pd.DataFrame:
        return original_build_leaderboard(_summary_with_method_family(summary, method_metadata), method_metadata)

    def build_subject_summary(summary: pd.DataFrame) -> pd.DataFrame:
        return original_build_subject_summary(_summary_with_method_family(summary))

    def build_protocol3_kshot_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
        return original_build_protocol3_kshot_leaderboard(_summary_with_method_family(summary))

    _report.build_leaderboard = build_leaderboard
    _report.build_subject_summary = build_subject_summary
    _report.build_protocol3_kshot_leaderboard = build_protocol3_kshot_leaderboard
    _report.build_protocol_summary = _build_protocol_summary
    _report._plot_balanced_accuracy_by_protocol = _plot_balanced_accuracy_by_protocol
    _report._write_markdown_report = _write_markdown_report
    _report._format_protocol_label = _format_protocol_label
    _report._fractional_protocol_report_patch_installed = True