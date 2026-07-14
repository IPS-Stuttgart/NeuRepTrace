"""Reporting utilities for BUSH-MEG all-protocol sweeps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = Path("results/bush_meg/all_protocols")

LEADERBOARD_COLUMNS = [
    "protocol_category",
    "method",
    "method_family",
    "mean_balanced_accuracy",
    "sem_balanced_accuracy",
    "median_balanced_accuracy",
    "mean_accuracy",
    "mean_log_loss",
    "mean_brier",
    "mean_ece",
    "n_subjects",
    "n_rows",
    "n_skipped",
    "valid_for_zero_calibration",
    "valid_for_strict_source_only",
    "debug_upper_bound",
]
SUBJECT_SUMMARY_COLUMNS = [
    "protocol_category",
    "method",
    "method_family",
    "outer_test_subject",
    "mean_balanced_accuracy",
    "mean_accuracy",
    "mean_log_loss",
    "mean_brier",
    "mean_ece",
    "n_rows",
]
PROTOCOL3_KSHOT_COLUMNS = [
    "method_base",
    "method",
    "method_family",
    "k_per_class",
    "mean_balanced_accuracy",
    "sem_balanced_accuracy",
    "mean_delta_vs_source_loso_logistic",
    "mean_delta_vs_best_protocol1",
    "n_subjects",
    "n_eval_trials",
    "n_calibration_trials",
]
PROTOCOL3_BY_K_COLUMNS = [
    "k_per_class",
    "mean_balanced_accuracy",
    "sem_balanced_accuracy",
    "mean_delta_vs_source_loso_logistic",
    "mean_delta_vs_best_protocol1",
    "n_methods",
    "n_subjects",
    "n_eval_trials",
    "n_calibration_trials",
]
PROTOCOL3_DELTA_COLUMNS = [
    "method",
    "method_base",
    "k_per_class",
    "outer_test_subject",
    "balanced_accuracy",
    "source_loso_logistic_balanced_accuracy",
    "best_protocol1_balanced_accuracy",
    "delta_vs_source_loso_logistic",
    "delta_vs_best_protocol1",
    "n_target_evaluation_trials",
    "n_target_calibration_trials",
]
BOOL_COLUMNS = ("valid_for_zero_calibration", "valid_for_strict_source_only", "debug_upper_bound")


@dataclass(frozen=True, slots=True)
class AllProtocolsReportResult:
    leaderboard_csv: Path
    protocol_summary_csv: Path
    subject_summary_csv: Path
    skipped_methods_csv: Path
    protocol3_kshot_leaderboard_csv: Path
    protocol3_by_k_csv: Path
    protocol3_delta_vs_source_only_csv: Path
    balanced_accuracy_by_method_png: Path
    balanced_accuracy_by_protocol_png: Path
    protocol3_accuracy_by_k_png: Path
    protocol3_delta_by_k_png: Path
    report_md: Path
    leaderboard: pd.DataFrame
    protocol_summary: pd.DataFrame
    subject_summary: pd.DataFrame
    skipped_methods: pd.DataFrame
    protocol3_kshot_leaderboard: pd.DataFrame
    protocol3_by_k: pd.DataFrame
    protocol3_delta_vs_source_only: pd.DataFrame


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _numeric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _numeric_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.median()) if values.notna().any() else float("nan")


def _numeric_sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sem()) if values.size > 1 else float("nan")


def _bool_like(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _status_skipped(metadata: pd.DataFrame) -> pd.Series:
    if metadata.empty:
        return pd.Series(dtype=bool)
    if "status" in metadata.columns:
        return metadata["status"].astype(str).str.lower().eq("skipped")
    if "runnable" in metadata.columns:
        return ~metadata["runnable"].map(_bool_like)
    if "skip_reason" in metadata.columns:
        return metadata["skip_reason"].fillna("").astype(str).str.len().gt(0)
    return pd.Series(False, index=metadata.index)


def _metadata_index(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty or "method" not in metadata.columns:
        return pd.DataFrame(columns=["method", "n_skipped"])
    aggregations: dict[str, Any] = {"n_skipped": ("_skipped", "sum")}
    for column in ("protocol_category", "method_family", *BOOL_COLUMNS):
        if column in metadata.columns:
            aggregations[column] = (column, "first")
    return metadata.assign(_skipped=_status_skipped(metadata).astype(int)).groupby("method", dropna=False).agg(**aggregations).reset_index()


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index), errors="coerce")


def _prepare_summary(summary: pd.DataFrame, *, unknown_family: bool) -> pd.DataFrame:
    frame = summary.copy()
    if "protocol_category" not in frame.columns:
        frame["protocol_category"] = np.nan
    if "method_family" not in frame.columns:
        frame["method_family"] = "unknown" if unknown_family else pd.NA
    elif unknown_family:
        frame["method_family"] = frame["method_family"].fillna("unknown")
    return frame


def _metric_row(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "mean_balanced_accuracy": _numeric_mean(group.get("balanced_accuracy", pd.Series(dtype=float))),
        "sem_balanced_accuracy": _numeric_sem(group.get("balanced_accuracy", pd.Series(dtype=float))),
        "median_balanced_accuracy": _numeric_median(group.get("balanced_accuracy", pd.Series(dtype=float))),
        "mean_accuracy": _numeric_mean(group.get("accuracy", pd.Series(dtype=float))),
        "mean_log_loss": _numeric_mean(group.get("log_loss", pd.Series(dtype=float))),
        "mean_brier": _numeric_mean(group.get("brier", pd.Series(dtype=float))),
        "mean_ece": _numeric_mean(group.get("ece", pd.Series(dtype=float))),
    }


def build_leaderboard(summary: pd.DataFrame, method_metadata: pd.DataFrame) -> pd.DataFrame:
    """Build a method-level leaderboard, including skipped-only methods from metadata."""

    metadata = _metadata_index(method_metadata)
    rows: list[dict[str, Any]] = []
    if not summary.empty and "method" in summary.columns:
        work = _prepare_summary(summary, unknown_family=False)
        group_columns = ["protocol_category", "method", "method_family", *(column for column in BOOL_COLUMNS if column in work.columns)]
        for keys, group in work.groupby(group_columns, dropna=False):
            key_values = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=True))
            rows.append({**key_values, **_metric_row(group), "n_subjects": int(group["outer_test_subject"].nunique()) if "outer_test_subject" in group.columns else 0, "n_rows": int(len(group))})
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        leaderboard = pd.DataFrame(columns=[column for column in LEADERBOARD_COLUMNS if column != "n_skipped"])
    if not metadata.empty:
        leaderboard = leaderboard.merge(metadata, on="method", how="outer", suffixes=("", "_metadata"))
        for column in ("protocol_category", "method_family", *BOOL_COLUMNS):
            metadata_column = f"{column}_metadata"
            if metadata_column in leaderboard.columns:
                leaderboard[column] = leaderboard[column].where(leaderboard[column].notna(), leaderboard[metadata_column]) if column in leaderboard.columns else leaderboard[metadata_column]
                leaderboard = leaderboard.drop(columns=[metadata_column])
    for column, default in (("n_skipped", 0), ("n_rows", 0), ("n_subjects", 0)):
        if column not in leaderboard.columns:
            leaderboard[column] = default
        leaderboard[column] = leaderboard[column].fillna(default).astype(int)
    leaderboard["method_family"] = leaderboard.get("method_family", pd.Series("unknown", index=leaderboard.index)).fillna("unknown")
    for column in BOOL_COLUMNS:
        if column not in leaderboard.columns:
            leaderboard[column] = False
        leaderboard[column] = leaderboard[column].map(_bool_like)
    for column in LEADERBOARD_COLUMNS:
        if column not in leaderboard.columns:
            leaderboard[column] = np.nan
    leaderboard = leaderboard[LEADERBOARD_COLUMNS].assign(_protocol_sort=pd.to_numeric(leaderboard["protocol_category"], errors="coerce"))
    return leaderboard.sort_values(["_protocol_sort", "mean_balanced_accuracy"], ascending=[True, False], na_position="last", kind="mergesort").drop(columns=["_protocol_sort"]).reset_index(drop=True)


def build_protocol_summary(summary: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance and method counts by protocol."""

    if "protocol_category" not in leaderboard.columns:
        return pd.DataFrame()
    summary_protocols = pd.to_numeric(summary["protocol_category"], errors="coerce") if "protocol_category" in summary.columns else pd.Series(dtype=float)
    leaderboard_protocols = pd.to_numeric(leaderboard["protocol_category"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for protocol_category in sorted(set(leaderboard_protocols.dropna().astype(int))):
        subject_rows = summary.loc[summary_protocols == protocol_category] if "protocol_category" in summary.columns else pd.DataFrame()
        method_rows = leaderboard.loc[leaderboard_protocols == protocol_category]
        rows.append(
            {
                "protocol_category": int(protocol_category),
                "n_methods": int(method_rows["method"].nunique()),
                "n_runnable_methods": int(method_rows.loc[method_rows["n_rows"] > 0, "method"].nunique()),
                "n_skipped_methods": int(method_rows.loc[method_rows["n_skipped"] > 0, "method"].nunique()),
                "n_subjects": int(subject_rows["outer_test_subject"].nunique()) if "outer_test_subject" in subject_rows.columns else 0,
                "n_rows": int(len(subject_rows)),
                "mean_balanced_accuracy": _numeric_mean(subject_rows.get("balanced_accuracy", pd.Series(dtype=float))),
                "sem_balanced_accuracy": _numeric_sem(subject_rows.get("balanced_accuracy", pd.Series(dtype=float))),
                "mean_accuracy": _numeric_mean(subject_rows.get("accuracy", pd.Series(dtype=float))),
                "mean_log_loss": _numeric_mean(subject_rows.get("log_loss", pd.Series(dtype=float))),
                "mean_brier": _numeric_mean(subject_rows.get("brier", pd.Series(dtype=float))),
                "mean_ece": _numeric_mean(subject_rows.get("ece", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def build_subject_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Build per-method, per-subject summary rows."""

    if summary.empty or "method" not in summary.columns or "outer_test_subject" not in summary.columns:
        return pd.DataFrame(columns=SUBJECT_SUMMARY_COLUMNS)
    work = _prepare_summary(summary, unknown_family=True)
    work["outer_test_subject"] = work["outer_test_subject"].astype(str)
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(["protocol_category", "method", "method_family", "outer_test_subject"], dropna=False):
        protocol_category, method, method_family, subject = keys
        rows.append({"protocol_category": protocol_category, "method": method, "method_family": method_family, "outer_test_subject": subject, **{key: value for key, value in _metric_row(group).items() if key != "sem_balanced_accuracy" and key != "median_balanced_accuracy"}, "n_rows": int(len(group))})
    return pd.DataFrame(rows)[SUBJECT_SUMMARY_COLUMNS].sort_values(["protocol_category", "method", "outer_test_subject"]).reset_index(drop=True)


def build_skipped_methods(method_metadata: pd.DataFrame) -> pd.DataFrame:
    if method_metadata.empty:
        return pd.DataFrame()
    return method_metadata.loc[_status_skipped(method_metadata)].copy().reset_index(drop=True)


def _protocol3_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "protocol_category" not in summary.columns:
        return pd.DataFrame(columns=summary.columns)
    rows = summary.loc[pd.to_numeric(summary["protocol_category"], errors="coerce") == 3].copy()
    if rows.empty:
        return rows
    if "method" not in rows.columns:
        rows["method"] = "unknown"
    if "outer_test_subject" not in rows.columns:
        rows["outer_test_subject"] = np.nan
    rows["method_family"] = rows["method_family"].fillna("unknown") if "method_family" in rows.columns else "unknown"
    rows["k_per_class"] = pd.to_numeric(rows["k_per_class"] if "k_per_class" in rows.columns else rows.get("target_calibration_per_class", np.nan), errors="coerce")
    if "method_base" not in rows.columns:
        rows["method_base"] = rows["method"].astype(str).str.replace(r"_k\d+$", "", regex=True)
    return rows


def _protocol1_baseline_by_subject(summary: pd.DataFrame, *, method: str | None = None) -> pd.Series:
    if summary.empty or "protocol_category" not in summary.columns or "outer_test_subject" not in summary.columns:
        return pd.Series(dtype=float)
    rows = summary.loc[pd.to_numeric(summary["protocol_category"], errors="coerce") == 1].copy()
    if method is not None and "method" in rows.columns:
        rows = rows.loc[rows["method"].astype(str) == method].copy()
    if rows.empty or "balanced_accuracy" not in rows.columns:
        return pd.Series(dtype=float)
    rows["outer_test_subject"] = rows["outer_test_subject"].astype(str)
    rows["balanced_accuracy"] = pd.to_numeric(rows["balanced_accuracy"], errors="coerce")
    if method is None:
        per_method = rows.groupby(["outer_test_subject", "method"], dropna=False)["balanced_accuracy"].mean().reset_index()
        return per_method.groupby("outer_test_subject")["balanced_accuracy"].max()
    return rows.groupby("outer_test_subject")["balanced_accuracy"].mean()


def _protocol3_delta_frame(summary: pd.DataFrame) -> pd.DataFrame:
    p3 = _protocol3_rows(summary)
    if p3.empty:
        return pd.DataFrame(columns=PROTOCOL3_DELTA_COLUMNS)
    source_loso = _protocol1_baseline_by_subject(summary, method="source_loso_logistic")
    best_p1 = _protocol1_baseline_by_subject(summary)
    p3["outer_test_subject"] = p3["outer_test_subject"].astype(str)
    p3["balanced_accuracy"] = _numeric_column(p3, "balanced_accuracy")
    p3["source_loso_logistic_balanced_accuracy"] = p3["outer_test_subject"].map(source_loso)
    p3["best_protocol1_balanced_accuracy"] = p3["outer_test_subject"].map(best_p1)
    p3["delta_vs_source_loso_logistic"] = p3["balanced_accuracy"] - p3["source_loso_logistic_balanced_accuracy"]
    p3["delta_vs_best_protocol1"] = p3["balanced_accuracy"] - p3["best_protocol1_balanced_accuracy"]
    for column in ("n_target_evaluation_trials", "n_target_calibration_trials"):
        p3[column] = _numeric_column(p3, column)
    for column in PROTOCOL3_DELTA_COLUMNS:
        if column not in p3.columns:
            p3[column] = np.nan
    return p3[PROTOCOL3_DELTA_COLUMNS].reset_index(drop=True)


def build_protocol3_delta_vs_source_only(summary: pd.DataFrame) -> pd.DataFrame:
    """Build per-subject Protocol 3 deltas against source-only baselines."""

    return _protocol3_delta_frame(summary)


def build_protocol3_kshot_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize Protocol 3 performance by method/k without mixing with zero-calibration methods."""

    p3 = _protocol3_rows(summary).reset_index(drop=True)
    if p3.empty:
        return pd.DataFrame(columns=PROTOCOL3_KSHOT_COLUMNS)
    deltas = _protocol3_delta_frame(summary)
    p3["balanced_accuracy"] = _numeric_column(p3, "balanced_accuracy")
    p3["delta_vs_source_loso_logistic"] = deltas["delta_vs_source_loso_logistic"].to_numpy()
    p3["delta_vs_best_protocol1"] = deltas["delta_vs_best_protocol1"].to_numpy()
    for column in ("n_target_evaluation_trials", "n_target_calibration_trials"):
        p3[column] = _numeric_column(p3, column)
    rows: list[dict[str, Any]] = []
    for keys, group in p3.groupby(["method_base", "method", "method_family", "k_per_class"], dropna=False):
        method_base, method, method_family, k_per_class = keys
        rows.append(
            {
                "method_base": method_base,
                "method": method,
                "method_family": method_family,
                "k_per_class": k_per_class,
                "mean_balanced_accuracy": _numeric_mean(group["balanced_accuracy"]),
                "sem_balanced_accuracy": _numeric_sem(group["balanced_accuracy"]),
                "mean_delta_vs_source_loso_logistic": _numeric_mean(group["delta_vs_source_loso_logistic"]),
                "mean_delta_vs_best_protocol1": _numeric_mean(group["delta_vs_best_protocol1"]),
                "n_subjects": int(group["outer_test_subject"].astype(str).nunique()),
                "n_eval_trials": int(group["n_target_evaluation_trials"].fillna(0).sum()),
                "n_calibration_trials": int(group["n_target_calibration_trials"].fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)[PROTOCOL3_KSHOT_COLUMNS].sort_values(["k_per_class", "mean_balanced_accuracy"], ascending=[True, False], na_position="last").reset_index(drop=True)


def build_protocol3_by_k(protocol3_kshot_leaderboard: pd.DataFrame) -> pd.DataFrame:
    if protocol3_kshot_leaderboard.empty:
        return pd.DataFrame(columns=PROTOCOL3_BY_K_COLUMNS)
    rows = []
    for k, group in protocol3_kshot_leaderboard.groupby("k_per_class", dropna=False):
        rows.append(
            {
                "k_per_class": k,
                "mean_balanced_accuracy": _numeric_mean(group["mean_balanced_accuracy"]),
                "sem_balanced_accuracy": _numeric_sem(group["mean_balanced_accuracy"]),
                "mean_delta_vs_source_loso_logistic": _numeric_mean(group["mean_delta_vs_source_loso_logistic"]),
                "mean_delta_vs_best_protocol1": _numeric_mean(group["mean_delta_vs_best_protocol1"]),
                "n_methods": int(group["method"].nunique()),
                "n_subjects": int(pd.to_numeric(group["n_subjects"], errors="coerce").fillna(0).max()),
                "n_eval_trials": int(pd.to_numeric(group["n_eval_trials"], errors="coerce").fillna(0).sum()),
                "n_calibration_trials": int(pd.to_numeric(group["n_calibration_trials"], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)[PROTOCOL3_BY_K_COLUMNS].sort_values("k_per_class", na_position="last").reset_index(drop=True)


def _plot_balanced_accuracy_by_method(leaderboard: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = leaderboard.loc[leaderboard["n_rows"] > 0].copy().sort_values(["protocol_category", "mean_balanced_accuracy"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(10, max(4.0, min(18.0, 0.36 * max(len(plot_data), 1) + 1.5))))
    if plot_data.empty:
        ax.text(0.5, 0.5, "No runnable result rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        y = np.arange(len(plot_data))
        protocol_numbers = pd.to_numeric(plot_data["protocol_category"], errors="coerce")
        colors = [f"C{int(protocol) % 10}" if pd.notna(protocol) else "C0" for protocol in protocol_numbers]
        ax.barh(y, plot_data["mean_balanced_accuracy"].astype(float), xerr=pd.to_numeric(plot_data["sem_balanced_accuracy"], errors="coerce").fillna(0.0).to_numpy(), color=colors, alpha=0.82)
        ax.set_yticks(y, labels=plot_data["method"].astype(str))
        ax.set_xlabel("Mean balanced accuracy")
        ax.set_ylabel("Method")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_balanced_accuracy_by_protocol(protocol_summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if protocol_summary.empty:
        ax.text(0.5, 0.5, "No protocol result rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = protocol_summary.sort_values("protocol_category")
        x = np.arange(len(data))
        ax.bar(x, data["mean_balanced_accuracy"].astype(float), yerr=pd.to_numeric(data["sem_balanced_accuracy"], errors="coerce").fillna(0.0).to_numpy(), color="C0", alpha=0.82)
        ax.set_xticks(x, labels=[f"P{int(protocol)}" for protocol in data["protocol_category"]])
        ax.set_ylabel("Mean balanced accuracy")
        ax.set_xlabel("Protocol category")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_protocol3_accuracy_by_k(protocol3_kshot_leaderboard: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if protocol3_kshot_leaderboard.empty:
        ax.text(0.5, 0.5, "No Protocol 3 k-shot rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = protocol3_kshot_leaderboard.copy()
        data["k_per_class"] = pd.to_numeric(data["k_per_class"], errors="coerce")
        for method_base, group in data.sort_values("k_per_class").groupby("method_base", dropna=False):
            ax.plot(group["k_per_class"], group["mean_balanced_accuracy"], marker="o", label=str(method_base))
        ax.set_xlabel("Target calibration rows per class (k)")
        ax.set_ylabel("Mean balanced accuracy")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_protocol3_delta_by_k(protocol3_kshot_leaderboard: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if protocol3_kshot_leaderboard.empty:
        ax.text(0.5, 0.5, "No Protocol 3 k-shot rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = protocol3_kshot_leaderboard.copy()
        data["k_per_class"] = pd.to_numeric(data["k_per_class"], errors="coerce")
        for method_base, group in data.sort_values("k_per_class").groupby("method_base", dropna=False):
            ax.plot(group["k_per_class"], group["mean_delta_vs_best_protocol1"], marker="o", label=str(method_base))
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_xlabel("Target calibration rows per class (k)")
        ax.set_ylabel("Mean BA delta vs best Protocol 1")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _format_percent(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value) * 100:.2f}%"


def _write_markdown_report(path: Path, leaderboard: pd.DataFrame, protocol_summary: pd.DataFrame, skipped_methods: pd.DataFrame, protocol3_kshot_leaderboard: pd.DataFrame, protocol3_by_k: pd.DataFrame) -> None:
    lines = ["# BUSH-MEG All-Protocols Report", "", f"- Methods in leaderboard: {leaderboard['method'].nunique() if 'method' in leaderboard.columns else 0}", f"- Skipped methods: {len(skipped_methods)}", "", "## Top Methods", ""]
    runnable = leaderboard.loc[leaderboard["n_rows"] > 0].copy()
    if runnable.empty:
        lines.append("No runnable result rows were found.")
    else:
        for _, row in runnable.sort_values("mean_balanced_accuracy", ascending=False).head(10).iterrows():
            lines.append(f"- P{int(row['protocol_category'])} `{row['method']}`: {_format_percent(row['mean_balanced_accuracy'])} mean BA across {int(row['n_subjects'])} subject(s)")
    lines.extend(["", "## Protocol Summary", ""])
    if protocol_summary.empty:
        lines.append("No protocol summary rows were available.")
    else:
        for _, row in protocol_summary.sort_values("protocol_category").iterrows():
            lines.append(f"- P{int(row['protocol_category'])}: {_format_percent(row['mean_balanced_accuracy'])} mean BA, {int(row['n_runnable_methods'])} runnable method(s), {int(row['n_skipped_methods'])} skipped method(s)")
    lines.extend(["", "## Interpretation Notes", "", "- Protocol 1 rows are strict source-only when `valid_for_strict_source_only` is true.", "- Protocol 2 rows may use unlabeled target features but not target labels for fitting.", "- Protocol 3 rows are calibrated target-label methods and are reported separately from zero-calibration Protocols 1/2.", "- Protocol 4 rows are oracle/debug upper bounds and should not be mixed into benchmark claims.", "", "", "## Protocol 3 K-Shot Summary", ""])
    if protocol3_by_k.empty:
        lines.append("No Protocol 3 k-shot rows were available.")
    else:
        for _, row in protocol3_by_k.iterrows():
            lines.append(f"- k={int(row['k_per_class']) if pd.notna(row['k_per_class']) else 'NA'}: {_format_percent(row['mean_balanced_accuracy'])} mean BA, delta vs best P1 {_format_percent(row['mean_delta_vs_best_protocol1'])}, {int(row['n_methods'])} method(s)")
    lines.extend(["", "## Protocol 3 Top Calibrated Methods", ""])
    if protocol3_kshot_leaderboard.empty:
        lines.append("No Protocol 3 calibrated leaderboard rows were available.")
    else:
        for _, row in protocol3_kshot_leaderboard.sort_values("mean_balanced_accuracy", ascending=False).head(10).iterrows():
            lines.append(f"- `{row['method']}`: {_format_percent(row['mean_balanced_accuracy'])} mean BA, delta vs best P1 {_format_percent(row['mean_delta_vs_best_protocol1'])}, n_eval_trials={int(row['n_eval_trials'])}")
    path.write_text("\n".join([*lines, ""]), encoding="utf-8")


def build_bushmeg_all_protocols_report(*, summary_csv: str | Path = DEFAULT_RESULTS_DIR / "summary.csv", method_metadata_csv: str | Path = DEFAULT_RESULTS_DIR / "method_metadata.csv", out_dir: str | Path = DEFAULT_RESULTS_DIR) -> AllProtocolsReportResult:
    """Read all-protocol artifacts and write CSV, figure, and Markdown reports."""

    summary_path, metadata_path, output_dir = Path(summary_csv), Path(method_metadata_csv), Path(out_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary, method_metadata = _read_csv_or_empty(summary_path), _read_csv_or_empty(metadata_path)
    leaderboard = build_leaderboard(summary, method_metadata)
    protocol_summary = build_protocol_summary(summary, leaderboard)
    subject_summary = build_subject_summary(summary)
    skipped_methods = build_skipped_methods(method_metadata)
    protocol3_kshot_leaderboard = build_protocol3_kshot_leaderboard(summary)
    protocol3_by_k = build_protocol3_by_k(protocol3_kshot_leaderboard)
    protocol3_delta_vs_source_only = build_protocol3_delta_vs_source_only(summary)
    paths = {
        "leaderboard_csv": output_dir / "leaderboard.csv",
        "protocol_summary_csv": output_dir / "protocol_summary.csv",
        "subject_summary_csv": output_dir / "subject_summary.csv",
        "skipped_methods_csv": output_dir / "skipped_methods.csv",
        "protocol3_kshot_leaderboard_csv": output_dir / "protocol3_kshot_leaderboard.csv",
        "protocol3_by_k_csv": output_dir / "protocol3_by_k.csv",
        "protocol3_delta_vs_source_only_csv": output_dir / "protocol3_delta_vs_source_only.csv",
        "balanced_accuracy_by_method_png": figures_dir / "balanced_accuracy_by_method.png",
        "balanced_accuracy_by_protocol_png": figures_dir / "balanced_accuracy_by_protocol.png",
        "protocol3_accuracy_by_k_png": figures_dir / "protocol3_accuracy_by_k.png",
        "protocol3_delta_by_k_png": figures_dir / "protocol3_delta_by_k.png",
        "report_md": output_dir / "report.md",
    }
    leaderboard.to_csv(paths["leaderboard_csv"], index=False)
    protocol_summary.to_csv(paths["protocol_summary_csv"], index=False)
    subject_summary.to_csv(paths["subject_summary_csv"], index=False)
    skipped_methods.to_csv(paths["skipped_methods_csv"], index=False)
    protocol3_kshot_leaderboard.to_csv(paths["protocol3_kshot_leaderboard_csv"], index=False)
    protocol3_by_k.to_csv(paths["protocol3_by_k_csv"], index=False)
    protocol3_delta_vs_source_only.to_csv(paths["protocol3_delta_vs_source_only_csv"], index=False)
    _plot_balanced_accuracy_by_method(leaderboard, paths["balanced_accuracy_by_method_png"])
    _plot_balanced_accuracy_by_protocol(protocol_summary, paths["balanced_accuracy_by_protocol_png"])
    _plot_protocol3_accuracy_by_k(protocol3_kshot_leaderboard, paths["protocol3_accuracy_by_k_png"])
    _plot_protocol3_delta_by_k(protocol3_kshot_leaderboard, paths["protocol3_delta_by_k_png"])
    _write_markdown_report(paths["report_md"], leaderboard, protocol_summary, skipped_methods, protocol3_kshot_leaderboard, protocol3_by_k)
    return AllProtocolsReportResult(**paths, leaderboard=leaderboard, protocol_summary=protocol_summary, subject_summary=subject_summary, skipped_methods=skipped_methods, protocol3_kshot_leaderboard=protocol3_kshot_leaderboard, protocol3_by_k=protocol3_by_k, protocol3_delta_vs_source_only=protocol3_delta_vs_source_only)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", "--summary", dest="summary_csv", default=str(DEFAULT_RESULTS_DIR / "summary.csv"))
    parser.add_argument("--method-metadata-csv", "--metadata", dest="method_metadata_csv", default=str(DEFAULT_RESULTS_DIR / "method_metadata.csv"))
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_bushmeg_all_protocols_report(summary_csv=args.summary_csv, method_metadata_csv=args.method_metadata_csv, out_dir=args.out_dir)
    print(f"Wrote leaderboard: {result.leaderboard_csv}")
    print(f"Wrote report: {result.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
