"""Preserve fractional protocol labels and tolerate legacy BUSH-MEG report schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import bushmeg_all_protocols_report as _report


def _numeric_protocol_categories(values: pd.Series) -> list[int | float]:
    numeric = pd.to_numeric(values, errors="coerce")
    found: dict[float, int | float] = {}
    for value in numeric.dropna().to_numpy(dtype=float):
        if np.isfinite(value):
            found[float(value)] = int(value) if float(value).is_integer() else float(value)
    return [found[key] for key in sorted(found)]


def _format_protocol_label(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)):
        value_float = float(numeric)
        return str(int(value_float)) if value_float.is_integer() else f"{value_float:g}"
    return str(value)


def _summary_with_method_family(summary: pd.DataFrame, method_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    normalized = summary.copy()
    normalized["method_family"] = normalized.get("method_family", pd.Series(pd.NA, index=normalized.index)).replace("", pd.NA)
    if method_metadata is not None and {"method", "method_family"}.issubset(method_metadata.columns) and "method" in normalized.columns:
        family_by_method = method_metadata.dropna(subset=["method"]).assign(_key=lambda df: df["method"].astype(str)).drop_duplicates("_key").set_index("_key")["method_family"]
        normalized["method_family"] = normalized["method_family"].where(normalized["method_family"].notna(), normalized["method"].astype(str).map(family_by_method))
    normalized["method_family"] = normalized["method_family"].fillna("unknown")
    return normalized


def _build_protocol_summary(summary: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    protocol_values = _numeric_protocol_categories(leaderboard["protocol_category"]) if "protocol_category" in leaderboard.columns else []
    summary_protocols = pd.to_numeric(summary["protocol_category"], errors="coerce") if "protocol_category" in summary.columns else pd.Series(dtype=float)
    leaderboard_protocols = pd.to_numeric(leaderboard["protocol_category"], errors="coerce") if "protocol_category" in leaderboard.columns else pd.Series(dtype=float)
    for protocol_category in protocol_values:
        subject_rows = summary.loc[summary_protocols == float(protocol_category)] if "protocol_category" in summary.columns else pd.DataFrame()
        method_rows = leaderboard.loc[leaderboard_protocols == float(protocol_category)] if "protocol_category" in leaderboard.columns else pd.DataFrame()
        rows.append({
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
        })
    return pd.DataFrame(rows)


def _build_protocol3_kshot_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["method_base", "method", "method_family", "k_per_class", "mean_balanced_accuracy", "sem_balanced_accuracy", "mean_delta_vs_source_loso_logistic", "mean_delta_vs_best_protocol1", "n_subjects", "n_eval_trials", "n_calibration_trials"]
    p3 = _report._protocol3_rows(summary)
    if p3.empty:
        return pd.DataFrame(columns=columns)
    p3 = p3.copy()
    source_loso = _report._protocol1_baseline_by_subject(summary, method="source_loso_logistic")
    best_p1 = _report._protocol1_baseline_by_subject(summary, method=None)
    p3["method"] = p3["method"].astype(str)
    p3["outer_test_subject"] = p3["outer_test_subject"].astype(str)
    p3["k_per_class"] = pd.to_numeric(p3["k_per_class"], errors="coerce")
    p3["balanced_accuracy"] = pd.to_numeric(p3.get("balanced_accuracy", np.nan), errors="coerce")
    p3["delta_vs_source_loso_logistic"] = p3["balanced_accuracy"] - p3["outer_test_subject"].map(source_loso)
    p3["delta_vs_best_protocol1"] = p3["balanced_accuracy"] - p3["outer_test_subject"].map(best_p1)
    for column in ("n_target_evaluation_trials", "n_target_calibration_trials"):
        if column not in p3.columns:
            p3[column] = np.nan
        p3[column] = pd.to_numeric(p3[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for keys, group in p3.groupby(["method_base", "method", "method_family", "k_per_class"], dropna=False):
        method_base, method, method_family, k_per_class = keys
        rows.append({
            "method_base": method_base,
            "method": method,
            "method_family": method_family,
            "k_per_class": k_per_class,
            "mean_balanced_accuracy": _report._numeric_mean(group["balanced_accuracy"]),
            "sem_balanced_accuracy": _report._numeric_sem(group["balanced_accuracy"]),
            "mean_delta_vs_source_loso_logistic": _report._numeric_mean(group["delta_vs_source_loso_logistic"]),
            "mean_delta_vs_best_protocol1": _report._numeric_mean(group["delta_vs_best_protocol1"]),
            "n_subjects": int(group["outer_test_subject"].astype(str).nunique()),
            "n_eval_trials": int(group["n_target_evaluation_trials"].fillna(0).sum()),
            "n_calibration_trials": int(group["n_target_calibration_trials"].fillna(0).sum()),
        })
    result = pd.DataFrame(rows)
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns].sort_values(["k_per_class", "mean_balanced_accuracy"], ascending=[True, False], na_position="last").reset_index(drop=True)


def _plot_balanced_accuracy_by_protocol(protocol_summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _report.plt.subplots(figsize=(7, 4.5))
    if protocol_summary.empty:
        ax.text(0.5, 0.5, "No protocol result rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = protocol_summary.assign(_protocol_sort=pd.to_numeric(protocol_summary["protocol_category"], errors="coerce")).sort_values("_protocol_sort", na_position="last").drop(columns="_protocol_sort")
        ax.bar(np.arange(len(data)), data["mean_balanced_accuracy"].astype(float), yerr=pd.to_numeric(data["sem_balanced_accuracy"], errors="coerce").fillna(0.0).to_numpy(), color="C0", alpha=0.82)
        ax.set_xticks(np.arange(len(data)), labels=[f"P{_format_protocol_label(protocol)}" for protocol in data["protocol_category"]])
        ax.set_ylabel("Mean balanced accuracy")
        ax.set_xlabel("Protocol category")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _report.plt.close(fig)


def _write_markdown_report(path: Path, leaderboard: pd.DataFrame, protocol_summary: pd.DataFrame, skipped_methods: pd.DataFrame, protocol3_kshot_leaderboard: pd.DataFrame, protocol3_by_k: pd.DataFrame) -> None:
    lines = ["# BUSH-MEG All-Protocols Report", "", f"- Methods in leaderboard: {leaderboard['method'].nunique() if 'method' in leaderboard.columns else 0}", f"- Skipped methods: {len(skipped_methods)}", "", "## Top Methods", ""]
    runnable = leaderboard.loc[leaderboard["n_rows"] > 0].copy()
    if runnable.empty:
        lines.append("No runnable result rows were found.")
    else:
        for _, row in runnable.sort_values("mean_balanced_accuracy", ascending=False).head(10).iterrows():
            lines.append(f"- P{_format_protocol_label(row['protocol_category'])} `{row['method']}`: {_report._format_percent(row['mean_balanced_accuracy'])} mean BA across {int(row['n_subjects'])} subject(s)")
    lines.extend(["", "## Protocol Summary", ""])
    if protocol_summary.empty:
        lines.append("No protocol summary rows were available.")
    else:
        data = protocol_summary.assign(_protocol_sort=pd.to_numeric(protocol_summary["protocol_category"], errors="coerce")).sort_values("_protocol_sort", na_position="last").drop(columns="_protocol_sort")
        for _, row in data.iterrows():
            lines.append(f"- P{_format_protocol_label(row['protocol_category'])}: {_report._format_percent(row['mean_balanced_accuracy'])} mean BA, {int(row['n_runnable_methods'])} runnable method(s), {int(row['n_skipped_methods'])} skipped method(s)")
    lines.extend(["", "## Interpretation Notes", "", "- Protocol 1 rows are strict source-only when `valid_for_strict_source_only` is true.", "- Protocol 2 rows may use unlabeled target features but not target labels for fitting.", "- Protocol 3 rows are calibrated target-label methods and are reported separately from zero-calibration Protocols 1/2.", "- Protocol 4 rows are oracle/debug upper bounds and should not be mixed into benchmark claims.", "", "", "## Protocol 3 K-Shot Summary", ""])
    if protocol3_by_k.empty:
        lines.append("No Protocol 3 k-shot rows were available.")
    else:
        for _, row in protocol3_by_k.iterrows():
            lines.append(f"- k={int(row['k_per_class']) if pd.notna(row['k_per_class']) else 'NA'}: {_report._format_percent(row['mean_balanced_accuracy'])} mean BA, delta vs best P1 {_report._format_percent(row['mean_delta_vs_best_protocol1'])}, {int(row['n_methods'])} method(s)")
    lines.extend(["", "## Protocol 3 Top Calibrated Methods", ""])
    if protocol3_kshot_leaderboard.empty:
        lines.append("No Protocol 3 calibrated leaderboard rows were available.")
    else:
        for _, row in protocol3_kshot_leaderboard.sort_values("mean_balanced_accuracy", ascending=False).head(10).iterrows():
            lines.append(f"- `{row['method']}`: {_report._format_percent(row['mean_balanced_accuracy'])} mean BA, delta vs best P1 {_report._format_percent(row['mean_delta_vs_best_protocol1'])}, n_eval_trials={int(row['n_eval_trials'])}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def install() -> None:
    if getattr(_report, "_fractional_protocol_report_patch_installed", False):
        return
    original_build_leaderboard = _report.build_leaderboard
    original_build_subject_summary = _report.build_subject_summary

    def build_leaderboard(summary: pd.DataFrame, method_metadata: pd.DataFrame) -> pd.DataFrame:
        return original_build_leaderboard(_summary_with_method_family(summary, method_metadata), method_metadata)

    def build_subject_summary(summary: pd.DataFrame) -> pd.DataFrame:
        return original_build_subject_summary(_summary_with_method_family(summary))

    def build_protocol3_kshot_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
        return _build_protocol3_kshot_leaderboard(_summary_with_method_family(summary))

    _report.build_leaderboard = build_leaderboard
    _report.build_subject_summary = build_subject_summary
    _report.build_protocol3_kshot_leaderboard = build_protocol3_kshot_leaderboard
    _report.build_protocol_summary = _build_protocol_summary
    _report._plot_balanced_accuracy_by_protocol = _plot_balanced_accuracy_by_protocol
    _report._write_markdown_report = _write_markdown_report
    _report._format_protocol_label = _format_protocol_label
    _report._fractional_protocol_report_patch_installed = True
