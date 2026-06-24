"""Audit BUSH-MEG all-protocol evaluation artifacts for protocol leakage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.bushmeg_all_protocols import method_registry
from neureptrace.dataset_config import load_config

DEFAULT_CONFIG = Path("configs/bush_meg/all_protocols.yml")
DEFAULT_RESULTS_DIR = Path("results/bush_meg/all_protocols/full")
DEFAULT_AUDIT_MD = Path("results/bush_meg/all_protocols/audit.md")

PROTOCOL3_REQUIRED_SUMMARY_FLAGS = {
    "uses_target_data": True,
    "uses_target_labels_for_fitting": True,
    "calibration_rows_disjoint_from_evaluation": True,
    "valid_for_zero_calibration": False,
    "valid_for_strict_source_only": False,
    "debug_upper_bound": False,
}
PROTOCOL3_REQUIRED_PREDICTION_COLUMNS = (
    "is_calibration_row",
    "target_calibration_per_class",
    "n_target_calibration_trials",
    "n_target_evaluation_trials",
)
PROTOCOL3_CALIBRATION_INDEX_COLUMNS = (
    "target_calibration_indices",
    "target_calibration_row_indices",
    "calibration_indices",
    "calibration_row_indices",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _bool_like(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _status_line(ok: bool, message: str) -> str:
    return f"- [{'PASS' if ok else 'FAIL'}] {message}"


def _protocol_rows(frame: pd.DataFrame, protocol_category: int) -> pd.DataFrame:
    if frame.empty or "protocol_category" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    protocol = pd.to_numeric(frame["protocol_category"], errors="coerce")
    return frame.loc[protocol == int(protocol_category)].copy()


def _expected_methods(config_path: Path, *, include_oracle: bool) -> set[str]:
    registry = method_registry()
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return {method for method, spec in registry.items() if spec.runnable and (include_oracle or spec.protocol_category != 4)}
    groups = ((config.get("all_protocols", {}) or {}).get("method_groups", {}) or {})
    names: list[str] = []
    group_names = (
        ("protocol4_oracle_debug",)
        if include_oracle
        else ("protocol1_source_only", "protocol2_unlabeled_target_adaptive", "protocol3_few_shot_calibrated")
    )
    for group in group_names:
        if not group:
            continue
        value = groups.get(group, [])
        names.extend(str(item) for item in value)
    return {name for name in names if name in registry}


def _first_present(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return pd.NA


def _numeric_or_na(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if not pd.isna(parsed) else float("nan")


def _row_label(row: pd.Series) -> str:
    parts = []
    for column in ("method", "outer_test_subject", "fold_index", "k_per_class", "target_calibration_per_class"):
        if column in row.index and not pd.isna(row[column]):
            parts.append(f"{column}={row[column]}")
    return ", ".join(parts) if parts else f"row_index={row.name}"


def _row_is_explicitly_skipped(row: pd.Series) -> bool:
    for column in ("target_calibration_skipped", "fold_skipped", "skipped", "is_skipped"):
        if column in row.index and _bool_like(row[column]):
            return True
    for column in ("skip_reason", "target_calibration_skip_reason", "fold_skip_reason"):
        if column in row.index and not pd.isna(row[column]) and str(row[column]).strip():
            return True
    return False


def _class_count_from_row(row: pd.Series) -> float:
    if "n_classes" in row.index:
        value = _numeric_or_na(row["n_classes"])
        if not pd.isna(value):
            return value
    if "class_names" in row.index and not pd.isna(row["class_names"]):
        labels = [token for token in str(row["class_names"]).replace(",", "|").split("|") if token != ""]
        if labels:
            return float(len(labels))
    return float("nan")


def _parse_index_set(value: Any) -> set[int]:
    if pd.isna(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        tokens = value
    else:
        text = str(value).strip()
        if not text:
            return set()
        for character in "[](){}":
            text = text.replace(character, " ")
        for separator in ("|", ";", ","):
            text = text.replace(separator, " ")
        tokens = text.split()
    indices: set[int] = set()
    for token in tokens:
        try:
            indices.add(int(float(str(token))))
        except ValueError:
            continue
    return indices


def _protocol3_group_key(row: pd.Series) -> tuple[str, str, str, str]:
    k_value = _first_present(row, ("k_per_class", "target_calibration_per_class"))
    return (
        str(row.get("method", "")),
        str(row.get("outer_test_subject", "")),
        str(row.get("fold_index", "")),
        str(k_value),
    )


def _protocol3_summary_failures(summary: pd.DataFrame) -> list[str]:
    p3 = _protocol_rows(summary, 3)
    if p3.empty:
        return []
    failures: list[str] = []
    for column, expected in PROTOCOL3_REQUIRED_SUMMARY_FLAGS.items():
        if column not in p3.columns:
            failures.append(f"Protocol 3 summary rows are missing required column `{column}`.")
            continue
        actual = p3[column].map(_bool_like)
        bad = p3.loc[actual.ne(expected)]
        if not bad.empty:
            preview = "; ".join(_row_label(row) for _, row in bad.head(5).iterrows())
            failures.append(f"Protocol 3 summary column `{column}` must be {str(expected).lower()}; bad rows: {preview}.")
    return failures


def _protocol3_calibration_count_failures(summary: pd.DataFrame) -> list[str]:
    p3 = _protocol_rows(summary, 3)
    if p3.empty:
        return []
    failures: list[str] = []
    if "n_target_calibration_trials" not in p3.columns:
        return ["Protocol 3 summary rows are missing required column `n_target_calibration_trials`."]
    for _, row in p3.iterrows():
        if _row_is_explicitly_skipped(row):
            continue
        k_value = _numeric_or_na(_first_present(row, ("k_per_class", "target_calibration_per_class")))
        n_classes = _class_count_from_row(row)
        n_calibration = _numeric_or_na(row["n_target_calibration_trials"])
        if pd.isna(k_value) or pd.isna(n_classes) or pd.isna(n_calibration):
            failures.append(f"Protocol 3 calibration count cannot be checked for {_row_label(row)}; missing k, n_classes, or n_target_calibration_trials.")
            continue
        expected = int(k_value) * int(n_classes)
        if int(n_calibration) != expected:
            failures.append(
                f"Protocol 3 calibration count mismatch for {_row_label(row)}: "
                f"n_target_calibration_trials={int(n_calibration)} but k*n_classes={expected}."
            )
    return failures


def _protocol3_prediction_required_failures(summary: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    p3_summary = _protocol_rows(summary, 3)
    if p3_summary.empty:
        return []
    if predictions.empty:
        return ["Protocol 3 summary rows exist but `predictions.csv` is missing or empty."]
    if "protocol_category" not in predictions.columns:
        return ["Protocol 3 predictions cannot be audited because `predictions.csv` is missing `protocol_category`."]
    p3_predictions = _protocol_rows(predictions, 3)
    if p3_predictions.empty:
        return ["Protocol 3 summary rows exist but `predictions.csv` has no Protocol 3 rows."]
    failures: list[str] = []
    for column in PROTOCOL3_REQUIRED_PREDICTION_COLUMNS:
        if column not in p3_predictions.columns:
            failures.append(f"Protocol 3 prediction rows are missing required column `{column}`.")
    if "is_calibration_row" in p3_predictions.columns:
        calibration_rows = p3_predictions.loc[p3_predictions["is_calibration_row"].map(_bool_like)]
        if not calibration_rows.empty:
            preview = "; ".join(_row_label(row) for _, row in calibration_rows.head(5).iterrows())
            failures.append(f"Protocol 3 predictions include rows marked as calibration rows: {preview}.")
    return failures


def _protocol3_prediction_overlap_failures(summary: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    p3_summary = _protocol_rows(summary, 3)
    if p3_summary.empty or predictions.empty or "protocol_category" not in predictions.columns:
        return []
    p3_predictions = _protocol_rows(predictions, 3)
    if p3_predictions.empty:
        return []
    index_column = next((column for column in PROTOCOL3_CALIBRATION_INDEX_COLUMNS if column in p3_summary.columns), "")
    if not index_column:
        return []
    prediction_index_column = "target_row_index" if "target_row_index" in p3_predictions.columns else "trial_index" if "trial_index" in p3_predictions.columns else ""
    if not prediction_index_column:
        return ["Protocol 3 summary exposes calibration row indices, but predictions lack `target_row_index`/`trial_index` for overlap auditing."]

    calibration_by_key: dict[tuple[str, str, str, str], set[int]] = {}
    for _, row in p3_summary.iterrows():
        indices = _parse_index_set(row[index_column])
        if indices:
            calibration_by_key.setdefault(_protocol3_group_key(row), set()).update(indices)
    if not calibration_by_key:
        return []

    failures: list[str] = []
    for _, row in p3_predictions.iterrows():
        key = _protocol3_group_key(row)
        calibration_indices = calibration_by_key.get(key, set())
        if not calibration_indices or pd.isna(row.get(prediction_index_column)):
            continue
        row_index = int(float(row[prediction_index_column]))
        if row_index in calibration_indices:
            failures.append(
                f"Protocol 3 prediction uses calibration row {row_index} for {_row_label(row)}."
            )
    return failures[:10]


def _protocol3_prediction_failures(summary: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    return _protocol3_prediction_required_failures(summary, predictions) + _protocol3_prediction_overlap_failures(summary, predictions)


def _leaderboard_calibrated_failures(leaderboard: pd.DataFrame, *, include_calibrated: bool) -> list[str]:
    if include_calibrated or leaderboard.empty or "protocol_category" not in leaderboard.columns:
        return []
    p3 = _protocol_rows(leaderboard, 3)
    if p3.empty:
        return []
    methods = ", ".join(sorted(p3.get("method", pd.Series(dtype=str)).astype(str).unique())[:10])
    return [
        "Protocol 3 calibrated methods appear in the leaderboard, but this audit was not marked "
        f"with `--include-calibrated`; methods: {methods}."
    ]


def _best_methods(leaderboard: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    if not leaderboard.empty and {"protocol_category", "method", "mean_balanced_accuracy", "n_rows"}.issubset(leaderboard.columns):
        data = leaderboard.loc[pd.to_numeric(leaderboard["n_rows"], errors="coerce").fillna(0) > 0].copy()
    elif not summary.empty and {"protocol_category", "method", "balanced_accuracy"}.issubset(summary.columns):
        data = (
            summary.groupby(["protocol_category", "method"], dropna=False)["balanced_accuracy"]
            .mean()
            .reset_index(name="mean_balanced_accuracy")
        )
    else:
        return ["- No balanced-accuracy rows available."]
    if data.empty:
        return ["- No balanced-accuracy rows available."]
    rows: list[str] = []
    protocols = sorted(pd.to_numeric(data["protocol_category"], errors="coerce").dropna().astype(int).unique())
    for protocol in protocols:
        group = data.loc[pd.to_numeric(data["protocol_category"], errors="coerce") == protocol].copy()
        if group.empty:
            continue
        row = group.sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
        rows.append(f"- Protocol {protocol}: `{row['method']}` ({float(row['mean_balanced_accuracy']) * 100:.2f}% mean BA)")
    return rows or ["- No balanced-accuracy rows available."]


def _append_failures(lines: list[str], failures: list[str], *, limit: int = 10) -> None:
    for failure in failures[:limit]:
        lines.append(f"  - {failure}")
    if len(failures) > limit:
        lines.append(f"  - ... {len(failures) - limit} additional failure(s) omitted.")


def build_audit_markdown(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
    out_path: str | Path = DEFAULT_AUDIT_MD,
    oracle_debug: bool = False,
    include_calibrated: bool = False,
) -> Path:
    results_dir = Path(results_dir)
    config_path = Path(config_path)
    out_path = Path(out_path)
    summary = _read_csv(results_dir / "summary.csv")
    predictions = _read_csv(results_dir / "predictions.csv")
    method_metadata = _read_csv(results_dir / "method_metadata.csv")
    skipped = _read_csv(results_dir / "skipped_methods.csv")
    leaderboard = _read_csv(results_dir / "leaderboard.csv")

    expected = set(method_metadata["method"].astype(str)) if "method" in method_metadata.columns and not method_metadata.empty else _expected_methods(config_path, include_oracle=oracle_debug)
    summary_methods = set(summary["method"].astype(str)) if "method" in summary.columns else set()
    skipped_methods = set(skipped["method"].astype(str)) if "method" in skipped.columns else set()
    accounted = summary_methods | skipped_methods
    missing_methods = sorted(expected.difference(accounted))
    skipped_missing_reason: list[str] = []
    if not skipped.empty and {"method", "skip_reason"}.issubset(skipped.columns):
        skipped_missing_reason = sorted(skipped.loc[skipped["skip_reason"].fillna("").astype(str).str.strip().eq(""), "method"].astype(str).unique())

    lines = [
        "# BUSH-MEG All-Protocols Audit",
        "",
        f"- Results directory: `{results_dir}`",
        f"- Oracle/debug audit: `{bool(oracle_debug)}`",
        f"- Calibrated leaderboard explicitly included: `{bool(include_calibrated)}`",
        "",
        "## Checks",
        "",
    ]

    lines.append(_status_line(not missing_methods and not skipped_missing_reason, "Every requested method is present in `summary.csv` or `skipped_methods.csv` with `skip_reason`."))
    if missing_methods:
        lines.append(f"  Missing methods: {', '.join(missing_methods)}")
    if skipped_missing_reason:
        lines.append(f"  Skipped methods without reason: {', '.join(skipped_missing_reason)}")

    leaderboard_failures = _leaderboard_calibrated_failures(leaderboard, include_calibrated=include_calibrated)
    lines.append(
        _status_line(
            not leaderboard_failures,
            "Protocol 3 calibrated rows are absent from default Protocol 1/2 leaderboards unless explicitly included.",
        )
    )
    _append_failures(lines, leaderboard_failures)

    if summary.empty:
        lines.append(_status_line(False, "`summary.csv` is missing or empty; row-level protocol checks cannot be completed."))
    else:
        protocol = pd.to_numeric(summary.get("protocol_category", pd.Series(dtype=float)), errors="coerce")
        p1 = summary.loc[protocol == 1]
        p2 = summary.loc[protocol == 2]
        p4 = summary.loc[protocol == 4]
        p1_ok = p1.empty or (
            p1.get("uses_target_data", pd.Series(False, index=p1.index)).map(_bool_like).eq(False).all()
            and p1.get("uses_target_labels_for_fitting", pd.Series(False, index=p1.index)).map(_bool_like).eq(False).all()
        )
        p2_ok = p2.empty or (
            p2.get("uses_target_data", pd.Series(False, index=p2.index)).map(_bool_like).eq(True).all()
            and p2.get("uses_target_labels_for_fitting", pd.Series(False, index=p2.index)).map(_bool_like).eq(False).all()
        )
        p3_summary_failures = _protocol3_summary_failures(summary)
        p3_prediction_failures = _protocol3_prediction_failures(summary, predictions)
        p3_count_failures = _protocol3_calibration_count_failures(summary)
        p4_ok = oracle_debug or p4.empty
        target_accuracy_columns = [column for column in summary.columns if "target_accuracy" in str(column).lower()]
        target_accuracy_ok = not target_accuracy_columns or not summary[target_accuracy_columns].applymap(_bool_like).any().any()
        outer_ok = "outer_test_subject" in summary.columns and summary["outer_test_subject"].notna().all()
        lines.append(_status_line(p1_ok, "Protocol 1 rows have `uses_target_data=false` and `uses_target_labels_for_fitting=false`."))
        lines.append(_status_line(p2_ok, "Protocol 2 rows have `uses_target_data=true` and `uses_target_labels_for_fitting=false`."))
        lines.append(
            _status_line(
                not p3_summary_failures,
                "Protocol 3 summary rows declare calibrated target use, no strict/zero-calibration validity, and no debug upper-bound status.",
            )
        )
        _append_failures(lines, p3_summary_failures)
        lines.append(_status_line(not p3_prediction_failures, "Protocol 3 prediction rows exclude calibration rows and include calibration/evaluation count metadata."))
        _append_failures(lines, p3_prediction_failures)
        lines.append(_status_line(not p3_count_failures, "Protocol 3 calibration counts equal `k * n_classes` for every non-skipped method/fold/k."))
        _append_failures(lines, p3_count_failures)
        lines.append(_status_line(p4_ok, "Protocol 4 rows are absent from the main leaderboard/output."))
        lines.append(_status_line(target_accuracy_ok, "No method selected hyperparameters using held-out target accuracy."))
        if target_accuracy_columns:
            lines.append(f"  Target-accuracy-related columns inspected: {', '.join(target_accuracy_columns)}")
        lines.append(_status_line(outer_ok, "All folds expose `outer_test_subject` as the held-out outer unit."))

    lines.extend(["", "## Best Methods By Protocol", ""])
    lines.extend(_best_methods(leaderboard, summary))
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_AUDIT_MD))
    parser.add_argument("--oracle-debug", action="store_true")
    parser.add_argument(
        "--include-calibrated",
        action="store_true",
        help="Allow Protocol 3 calibrated methods in leaderboard checks for reports that explicitly include calibrated methods.",
    )
    args = parser.parse_args(argv)
    path = build_audit_markdown(
        results_dir=args.results_dir,
        config_path=args.config,
        out_path=args.out,
        oracle_debug=args.oracle_debug,
        include_calibrated=args.include_calibrated,
    )
    print(f"Wrote audit: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
