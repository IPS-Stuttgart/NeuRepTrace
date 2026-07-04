"""Runtime safeguards for BUSH-MEG all-protocol report helpers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BOOL_PATCH_ATTR = "_neureptrace_bushmeg_report_bool_like_lists"
_PATCH_ATTR = _BOOL_PATCH_ATTR
_CSV_PATCH_ATTR = "_neureptrace_bushmeg_report_empty_csv_guard"
_METHOD_FAMILY_PATCH_ATTR = "_neureptrace_bushmeg_report_method_family_guard"
_PROTOCOL3_ROWS_PATCH_ATTR = "_neureptrace_bushmeg_report_protocol3_rows_method_family_guard"
_PROTOCOL3_KSHOT_PATCH_ATTR = "_neureptrace_bushmeg_report_protocol3_kshot_no_m2m_merge"
_TRUE_STRINGS = {"1", "true", "yes", "y"}
_EMPTY_STRINGS = {"", "nan", "none", "null", "<na>"}


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return [value.item()] if value.ndim == 0 else list(value.ravel())
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _scalar_bool_like(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        return False if text in _EMPTY_STRINGS else text in _TRUE_STRINGS
    if value is None or _is_missing_scalar(value):
        return False
    return bool(value)


def _bool_like(value: Any) -> bool:
    values = []
    for item in _items(value):
        if isinstance(item, str):
            if item.strip().lower() in _EMPTY_STRINGS:
                continue
        elif item is None or _is_missing_scalar(item):
            continue
        values.append(item)
    return any(_scalar_bool_like(item) for item in values)


def _read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _method_family_lookup(method_metadata: pd.DataFrame | None) -> pd.Series:
    if method_metadata is None or method_metadata.empty or "method" not in method_metadata.columns or "method_family" not in method_metadata.columns:
        return pd.Series(dtype=object)
    metadata = method_metadata.loc[:, ["method", "method_family"]].copy()
    metadata = metadata.dropna(subset=["method"])
    if metadata.empty:
        return pd.Series(dtype=object)
    metadata["method"] = metadata["method"].astype(str)
    metadata["method_family"] = metadata["method_family"].where(metadata["method_family"].notna(), "unknown").astype(str)
    metadata = metadata.drop_duplicates("method", keep="first")
    return metadata.set_index("method")["method_family"]


def _with_method_family(summary: pd.DataFrame, method_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    enriched = summary.copy()
    if "method_family" in enriched.columns:
        enriched["method_family"] = enriched["method_family"].where(enriched["method_family"].notna(), "unknown").astype(str)
        return enriched
    enriched["method_family"] = "unknown"
    if not enriched.empty and "method" in enriched.columns:
        lookup = _method_family_lookup(method_metadata)
        if not lookup.empty:
            enriched["method_family"] = enriched["method"].astype(str).map(lookup).fillna("unknown")
    return enriched


def _protocol3_method_families(report: Any, summary: pd.DataFrame) -> pd.DataFrame:
    p3 = report._protocol3_rows(_with_method_family(summary))
    columns = ["method", "method_base", "k_per_class", "method_family"]
    if p3.empty:
        return pd.DataFrame(columns=columns)
    p3 = p3.copy()
    if "method_base" not in p3.columns and "method" in p3.columns:
        p3["method_base"] = p3["method"].astype(str).str.replace(r"_k\d+$", "", regex=True)
    if "k_per_class" not in p3.columns:
        p3["k_per_class"] = p3.get("target_calibration_per_class", np.nan)
    if "method_family" not in p3.columns:
        p3["method_family"] = "unknown"
    for column in columns:
        if column not in p3.columns:
            p3[column] = np.nan
    p3["method"] = p3["method"].astype(str)
    p3["method_base"] = p3["method_base"].astype(str)
    p3["k_per_class"] = pd.to_numeric(p3["k_per_class"], errors="coerce")
    p3["method_family"] = p3["method_family"].where(p3["method_family"].notna(), "unknown").astype(str)
    return p3.loc[:, columns].drop_duplicates(["method", "method_base", "k_per_class"], keep="first").reset_index(drop=True)


def _build_protocol3_kshot_leaderboard(report: Any, summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    enriched_summary = _with_method_family(summary)
    p3 = report._protocol3_rows(enriched_summary)
    if p3.empty:
        return pd.DataFrame(columns=columns)
    deltas = report._protocol3_delta_frame(enriched_summary)
    if deltas.empty:
        merged = p3.copy()
        for column in ("delta_vs_source_loso_logistic", "delta_vs_best_protocol1"):
            merged[column] = np.nan
    else:
        merged = deltas.copy()
    if "method_base" not in merged.columns and "method" in merged.columns:
        merged["method_base"] = merged["method"].astype(str).str.replace(r"_k\d+$", "", regex=True)
    if "k_per_class" not in merged.columns:
        merged["k_per_class"] = np.nan
    if "method_family" in merged.columns:
        merged = merged.drop(columns=["method_family"])
    merged["method"] = merged["method"].astype(str)
    merged["method_base"] = merged["method_base"].astype(str)
    merged["k_per_class"] = pd.to_numeric(merged["k_per_class"], errors="coerce")
    families = _protocol3_method_families(report, enriched_summary)
    merged = merged.merge(families, on=["method", "method_base", "k_per_class"], how="left", validate="many_to_one")
    merged["method_family"] = merged["method_family"].where(merged["method_family"].notna(), "unknown").astype(str)
    for column in (
        "balanced_accuracy",
        "delta_vs_source_loso_logistic",
        "delta_vs_best_protocol1",
        "n_target_evaluation_trials",
        "n_target_calibration_trials",
    ):
        if column not in merged.columns:
            merged[column] = np.nan
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    group_columns = ["method_base", "method", "method_family", "k_per_class"]
    for keys, group in merged.groupby(group_columns, dropna=False):
        key_values = dict(zip(group_columns, keys, strict=True))
        rows.append(
            {
                **key_values,
                "mean_balanced_accuracy": report._numeric_mean(group["balanced_accuracy"]),
                "sem_balanced_accuracy": report._numeric_sem(group["balanced_accuracy"]),
                "mean_delta_vs_source_loso_logistic": report._numeric_mean(group["delta_vs_source_loso_logistic"]),
                "mean_delta_vs_best_protocol1": report._numeric_mean(group["delta_vs_best_protocol1"]),
                "n_subjects": int(group["outer_test_subject"].astype(str).nunique()) if "outer_test_subject" in group.columns else 0,
                "n_eval_trials": int(group["n_target_evaluation_trials"].fillna(0).sum()),
                "n_calibration_trials": int(group["n_target_calibration_trials"].fillna(0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[columns].sort_values(["k_per_class", "mean_balanced_accuracy"], ascending=[True, False], na_position="last").reset_index(drop=True)


def _install_bool_patch(report: Any) -> None:
    original = getattr(report, "_bool_like", None)
    if getattr(original, _BOOL_PATCH_ATTR, False):
        return
    setattr(_bool_like, _BOOL_PATCH_ATTR, True)
    if original is not None:
        _bool_like.__wrapped__ = original
    report._bool_like = _bool_like


def _install_empty_csv_patch(report: Any) -> None:
    original = getattr(report, "_read_csv_or_empty", None)
    if getattr(original, _CSV_PATCH_ATTR, False):
        return
    setattr(_read_csv_or_empty, _CSV_PATCH_ATTR, True)
    if original is not None:
        _read_csv_or_empty.__wrapped__ = original
    report._read_csv_or_empty = _read_csv_or_empty


def _install_method_family_patch(report: Any) -> None:
    original_leaderboard = getattr(report, "build_leaderboard", None)
    if original_leaderboard is not None and not getattr(original_leaderboard, _METHOD_FAMILY_PATCH_ATTR, False):

        @wraps(original_leaderboard)
        def build_leaderboard(summary: pd.DataFrame, method_metadata: pd.DataFrame) -> pd.DataFrame:
            return original_leaderboard(_with_method_family(summary, method_metadata), method_metadata)

        setattr(build_leaderboard, _METHOD_FAMILY_PATCH_ATTR, True)
        build_leaderboard.__wrapped__ = original_leaderboard
        report.build_leaderboard = build_leaderboard

    original_subject_summary = getattr(report, "build_subject_summary", None)
    if original_subject_summary is not None and not getattr(original_subject_summary, _METHOD_FAMILY_PATCH_ATTR, False):

        @wraps(original_subject_summary)
        def build_subject_summary(summary: pd.DataFrame) -> pd.DataFrame:
            return original_subject_summary(_with_method_family(summary))

        setattr(build_subject_summary, _METHOD_FAMILY_PATCH_ATTR, True)
        build_subject_summary.__wrapped__ = original_subject_summary
        report.build_subject_summary = build_subject_summary


def _install_protocol3_rows_patch(report: Any) -> None:
    original = getattr(report, "_protocol3_rows", None)
    if original is None or getattr(original, _PROTOCOL3_ROWS_PATCH_ATTR, False):
        return

    @wraps(original)
    def _protocol3_rows(summary: pd.DataFrame) -> pd.DataFrame:
        return original(_with_method_family(summary))

    setattr(_protocol3_rows, _PROTOCOL3_ROWS_PATCH_ATTR, True)
    _protocol3_rows.__wrapped__ = original
    report._protocol3_rows = _protocol3_rows


def _install_protocol3_kshot_patch(report: Any) -> None:
    original = getattr(report, "build_protocol3_kshot_leaderboard", None)
    if original is None or getattr(original, _PROTOCOL3_KSHOT_PATCH_ATTR, False):
        return

    @wraps(original)
    def build_protocol3_kshot_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
        return _build_protocol3_kshot_leaderboard(report, summary)

    setattr(build_protocol3_kshot_leaderboard, _PROTOCOL3_KSHOT_PATCH_ATTR, True)
    build_protocol3_kshot_leaderboard.__wrapped__ = original
    report.build_protocol3_kshot_leaderboard = build_protocol3_kshot_leaderboard


def install() -> None:
    report = importlib.import_module("neureptrace.bushmeg_all_protocols_report")
    _install_bool_patch(report)
    _install_empty_csv_patch(report)
    _install_method_family_patch(report)
    _install_protocol3_rows_patch(report)
    _install_protocol3_kshot_patch(report)


__all__ = ["install"]
