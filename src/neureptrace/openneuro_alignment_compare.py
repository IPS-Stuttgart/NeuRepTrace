"""Compare OpenNeuro source-alignment variants and emit debug decisions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

MINIMIZE_METRICS = {"log_loss", "brier", "ece"}
IDENTITY_ANCHOR_MODES = {
    "stimulus_id_mean",
    "stimulus_id_repetition",
    "event_code_mean",
    "run_event_index_within_stimulus",
}
CLASS_REPETITION_ANCHOR = "class_repetition"
STRICT_TARGET_PROJECTION = "group_projection"
TARGET_CALIBRATED_TARGET_PROJECTION = "target_calibrated_alignment"
ORACLE_TARGET_PROJECTION = "oracle_target_calibrated_alignment"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _median(frame: pd.DataFrame, column: str) -> float | str:
    values = _numeric_series(frame, column).dropna()
    return "" if values.empty else float(values.median())


def _min(frame: pd.DataFrame, column: str) -> float | str:
    values = _numeric_series(frame, column).dropna()
    return "" if values.empty else float(values.min())


def _max(frame: pd.DataFrame, column: str) -> float | str:
    values = _numeric_series(frame, column).dropna()
    return "" if values.empty else float(values.max())


def _compact_unique(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = [
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    ]
    return "|".join(dict.fromkeys(values))


def _single_unique(frame: pd.DataFrame, column: str, *, artifact: str) -> str:
    if column not in frame.columns:
        return ""
    values = [
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    ]
    unique = list(dict.fromkeys(values))
    if len(unique) > 1:
        raise ValueError(f"Artifact {artifact!r} has inconsistent {column!r} values: {unique}")
    return unique[0] if unique else ""


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value not in {"", None}:
            text = str(value).strip()
            if text:
                return text
    return ""


def _output_dir_from_summary(summary_path: Path) -> Path:
    return summary_path.parent.parent


def discover_output_dirs(paths: Sequence[str | Path]) -> list[Path]:
    """Return unique OpenNeuro output dirs containing decode/time_decode_summary.csv."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        candidates: list[Path] = []
        direct_summary = path / "decode" / "time_decode_summary.csv"
        if direct_summary.is_file():
            candidates.append(path)
        if path.is_file() and path.name == "time_decode_summary.csv" and path.parent.name == "decode":
            candidates.append(_output_dir_from_summary(path))
        if path.exists() and path.is_dir():
            candidates.extend(_output_dir_from_summary(summary) for summary in path.rglob("decode/time_decode_summary.csv"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(candidate)
    return sorted(discovered, key=lambda item: item.as_posix())


def _select_metric(summary: pd.DataFrame, *, metric: str, fixed_time: float | None) -> dict[str, Any]:
    if "time" not in summary.columns:
        raise ValueError("time_decode_summary.csv is missing required column 'time'.")
    if metric not in summary.columns:
        raise ValueError(f"time_decode_summary.csv is missing requested metric column {metric!r}.")

    values = summary[["time", metric]].copy()
    values["time"] = pd.to_numeric(values["time"], errors="coerce")
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna(subset=["time", metric])
    if values.empty:
        return {
            "selection_metric": metric,
            "selection_time": "",
            "selection_value": "",
            "selection_score": "",
            "selection_mode": "missing_metric_rows",
        }
    by_time = values.groupby("time", dropna=False)[metric].mean().reset_index()
    minimize = metric in MINIMIZE_METRICS
    if fixed_time is not None:
        index = (by_time["time"] - float(fixed_time)).abs().idxmin()
        mode = "nearest_fixed_time"
    else:
        index = by_time[metric].idxmin() if minimize else by_time[metric].idxmax()
        mode = "best_time"
    row = by_time.loc[index]
    value = float(row[metric])
    return {
        "selection_metric": metric,
        "selection_time": float(row["time"]),
        "selection_value": value,
        "selection_score": -value if minimize else value,
        "selection_mode": mode,
    }


def _alignment_diagnostic_summary(diagnostics: pd.DataFrame) -> dict[str, Any]:
    if diagnostics.empty:
        return {
            "alignment_diagnostics_present": False,
            "alignment_diagnostics_rows": 0,
        }
    collapse = (
        diagnostics["uses_channel_projection_collapse"].map(_as_bool)
        if "uses_channel_projection_collapse" in diagnostics.columns
        else pd.Series(dtype=bool)
    )
    if not collapse.empty and {
        "alignment_window_center",
        "alignment_window_size",
        "decode_window_center",
        "decode_window_size",
    }.issubset(diagnostics.columns):
        same_window = (
            pd.to_numeric(diagnostics["alignment_window_center"], errors="coerce")
            .sub(pd.to_numeric(diagnostics["decode_window_center"], errors="coerce"))
            .abs()
            .le(1e-9)
            & pd.to_numeric(diagnostics["alignment_window_size"], errors="coerce")
            .sub(pd.to_numeric(diagnostics["decode_window_size"], errors="coerce"))
            .abs()
            .le(1e-9)
        )
        collapse = collapse & ~same_window.fillna(False)
    if "alignment_dimensionality_reduction" in diagnostics.columns:
        reduction = diagnostics["alignment_dimensionality_reduction"].map(_as_bool)
    elif {"decode_feature_dim", "feature_dim"}.issubset(diagnostics.columns):
        reduction = pd.to_numeric(diagnostics["decode_feature_dim"], errors="coerce") < pd.to_numeric(
            diagnostics["feature_dim"],
            errors="coerce",
        )
    else:
        reduction = pd.Series(dtype=bool)
    before = _median(diagnostics, "anchor_row_correlation_before")
    after = _median(diagnostics, "anchor_row_correlation_after")
    inner_before = _median(diagnostics, "source_inner_decoding_before_alignment")
    inner_after = _median(diagnostics, "source_inner_decoding_after_alignment")
    return {
        "alignment_diagnostics_present": True,
        "alignment_diagnostics_rows": int(len(diagnostics)),
        "diagnostic_actual_components_median": _median(diagnostics, "actual_components"),
        "diagnostic_actual_components_min": _min(diagnostics, "actual_components"),
        "diagnostic_actual_components_max": _max(diagnostics, "actual_components"),
        "diagnostic_n_alignment_rows_median": _median(diagnostics, "n_alignment_rows"),
        "diagnostic_n_alignment_rows_min": _min(diagnostics, "n_alignment_rows"),
        "diagnostic_n_repetitions_per_class_median": _median(diagnostics, "n_repetitions_per_class"),
        "diagnostic_feature_dim_median": _median(diagnostics, "feature_dim"),
        "diagnostic_decode_feature_dim_median": _median(diagnostics, "decode_feature_dim"),
        "diagnostic_channel_projection_collapse_fraction": "" if collapse.empty else float(collapse.mean()),
        "diagnostic_uses_channel_projection_collapse_any": bool(collapse.any()) if not collapse.empty else "",
        "diagnostic_dimensionality_reduction_fraction": "" if reduction.empty else float(reduction.mean()),
        "diagnostic_uses_dimensionality_reduction_any": bool(reduction.any()) if not reduction.empty else "",
        "diagnostic_target_transform_type": _compact_unique(diagnostics, "target_transform_type"),
        "diagnostic_anchor_row_correlation_before_median": before,
        "diagnostic_anchor_row_correlation_after_median": after,
        "diagnostic_anchor_row_correlation_gain_median": (
            ""
            if before == "" or after == ""
            else float(after) - float(before)
        ),
        "diagnostic_source_inner_decoding_before_median": inner_before,
        "diagnostic_source_inner_decoding_after_median": inner_after,
        "diagnostic_source_inner_decoding_gain_median": (
            ""
            if inner_before == "" or inner_after == ""
            else float(inner_after) - float(inner_before)
        ),
    }


def summarize_alignment_variant(
    output_dir: str | Path,
    *,
    metric: str = "balanced_accuracy",
    fixed_time: float | None = None,
) -> dict[str, Any]:
    """Summarize one OpenNeuro output directory as an alignment variant row."""

    output = Path(output_dir)
    decode_dir = output / "decode"
    summary_path = decode_dir / "time_decode_summary.csv"
    diagnostics_path = decode_dir / "alignment_diagnostics.csv"
    diagnostics_time_course_path = decode_dir / "diagnostics" / "time_course_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing decode summary: {summary_path}")

    manifest = _read_json(output / "run_manifest.json")
    summary = pd.read_csv(summary_path)
    metric_summary = (
        pd.read_csv(diagnostics_time_course_path)
        if diagnostics_time_course_path.is_file() and diagnostics_time_course_path.stat().st_size > 0
        else summary
    )
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.is_file() and diagnostics_path.stat().st_size > 0 else pd.DataFrame()
    selection = _select_metric(metric_summary, metric=metric, fixed_time=fixed_time)
    artifact_name = manifest.get("artifact_name", output.name)
    method = _first_nonempty(_single_unique(summary, "alignment_method", artifact=artifact_name), manifest.get("alignment_method", ""))
    anchor_mode = _first_nonempty(
        _single_unique(summary, "alignment_anchor_mode", artifact=artifact_name),
        manifest.get("alignment_anchor_mode", ""),
    )
    target_projection = _first_nonempty(
        _single_unique(summary, "alignment_target_projection", artifact=artifact_name),
        manifest.get("alignment_target_projection", ""),
    )
    oracle = target_projection == ORACLE_TARGET_PROJECTION
    target_calibrated = target_projection == TARGET_CALIBRATED_TARGET_PROJECTION
    explicit_valid_text = _single_unique(summary, "alignment_valid_for_benchmark", artifact=artifact_name)
    explicit_valid = None if explicit_valid_text == "" else _as_bool(explicit_valid_text)
    projection_valid = bool(not oracle and not target_calibrated)
    valid_for_benchmark = projection_valid if explicit_valid is None else bool(projection_valid and explicit_valid)
    row = {
        "output_dir": output.as_posix(),
        "artifact_name": artifact_name,
        "github_run_id": manifest.get("github_run_id", ""),
        "dataset": _first_nonempty(manifest.get("dataset", ""), _compact_unique(summary, "dataset")),
        "mode": manifest.get("mode", ""),
        "subjects": manifest.get("subjects", ""),
        "runs": manifest.get("runs", ""),
        "n_subjects": manifest.get("n_subjects", ""),
        "alignment_method": method,
        "alignment_anchor_mode": anchor_mode,
        "alignment_anchor_column": _first_nonempty(
            _single_unique(summary, "alignment_anchor_column", artifact=artifact_name),
            manifest.get("alignment_anchor_column", ""),
        ),
        "alignment_target_projection": target_projection,
        "alignment_target_calibrated": target_calibrated,
        "alignment_oracle_target_calibrated": oracle,
        "alignment_valid_for_benchmark": valid_for_benchmark,
        "identity_anchor": anchor_mode in IDENTITY_ANCHOR_MODES,
        "class_repetition_anchor": anchor_mode == CLASS_REPETITION_ANCHOR,
        "time_decode_summary_rows": int(len(summary)),
        "selection_source": "diagnostics_time_course" if metric_summary is not summary else "time_decode_summary",
        **selection,
        **_alignment_diagnostic_summary(diagnostics),
    }
    return row


def build_variant_summary(
    output_dirs: Sequence[str | Path],
    *,
    metric: str = "balanced_accuracy",
    fixed_time: float | None = None,
) -> pd.DataFrame:
    rows = [
        summarize_alignment_variant(output_dir, metric=metric, fixed_time=fixed_time)
        for output_dir in output_dirs
    ]
    return pd.DataFrame(rows)


def _best_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(["selection_score", "selection_value"], ascending=[False, False]).iloc[0]


def _valid_strict_rows(group: pd.DataFrame) -> pd.DataFrame:
    return group[
        (group["alignment_target_projection"] == STRICT_TARGET_PROJECTION)
        & (group["alignment_valid_for_benchmark"].map(_as_bool))
    ]


def _valid_raw_rows(group: pd.DataFrame) -> pd.DataFrame:
    return group[
        group["alignment_method"].isin(["", "none"])
        & (group["alignment_valid_for_benchmark"].map(_as_bool))
    ]


def build_anchor_comparison(variants: pd.DataFrame, *, min_delta: float = 0.0) -> pd.DataFrame:
    """Compare true identity anchors against class_repetition within matched groups."""

    if variants.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "alignment_method", "alignment_target_projection", "selection_metric"]
    benchmark_variants = _valid_strict_rows(variants)
    for group_values, group in benchmark_variants.groupby(group_columns, dropna=False):
        group_map = dict(zip(group_columns, group_values, strict=False))
        class_rows = group[group["alignment_anchor_mode"] == CLASS_REPETITION_ANCHOR]
        identity_rows = group[group["alignment_anchor_mode"].isin(IDENTITY_ANCHOR_MODES)]
        if class_rows.empty or identity_rows.empty:
            continue
        class_row = _best_row(class_rows)
        identity_row = _best_row(identity_rows)
        delta = float(identity_row["selection_score"]) - float(class_row["selection_score"])
        if delta > min_delta:
            decision = "true_identity_anchor_better_than_class_repetition"
            interpretation = "anchor_semantics_likely_issue"
        elif delta < -min_delta:
            decision = "class_repetition_not_worse_than_true_identity"
            interpretation = "anchor_semantics_not_primary_from_current_runs"
        else:
            decision = "no_clear_identity_anchor_gain"
            interpretation = "anchor_semantics_inconclusive"
        rows.append(
            {
                **group_map,
                "class_repetition_artifact": class_row["artifact_name"],
                "class_repetition_value": class_row["selection_value"],
                "best_identity_anchor_mode": identity_row["alignment_anchor_mode"],
                "best_identity_artifact": identity_row["artifact_name"],
                "best_identity_value": identity_row["selection_value"],
                "score_delta_identity_minus_class_repetition": delta,
                "min_delta": float(min_delta),
                "decision": decision,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_oracle_comparison(variants: pd.DataFrame, *, min_delta: float = 0.0) -> pd.DataFrame:
    """Compare oracle target-calibrated projection against strict group projection."""

    if variants.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "alignment_method", "alignment_anchor_mode", "selection_metric"]
    for group_values, group in variants.groupby(group_columns, dropna=False):
        group_map = dict(zip(group_columns, group_values, strict=False))
        strict_rows = _valid_strict_rows(group)
        oracle_rows = group[group["alignment_target_projection"] == ORACLE_TARGET_PROJECTION]
        if strict_rows.empty or oracle_rows.empty:
            continue
        strict_row = _best_row(strict_rows)
        oracle_row = _best_row(oracle_rows)
        delta = float(oracle_row["selection_score"]) - float(strict_row["selection_score"])
        if delta > min_delta:
            decision = "oracle_target_calibration_helps"
            interpretation = "strict_source_only_target_projection_likely_bottleneck"
        elif delta < -min_delta:
            decision = "oracle_target_calibration_does_not_help"
            interpretation = "focus_feature_space_anchor_or_alignment_implementation"
        else:
            decision = "no_clear_oracle_target_calibration_gain"
            interpretation = "target_projection_inconclusive"
        rows.append(
            {
                **group_map,
                "strict_artifact": strict_row["artifact_name"],
                "strict_value": strict_row["selection_value"],
                "oracle_artifact": oracle_row["artifact_name"],
                "oracle_value": oracle_row["selection_value"],
                "score_delta_oracle_minus_strict": delta,
                "min_delta": float(min_delta),
                "decision": decision,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_target_calibrated_comparison(variants: pd.DataFrame, *, min_delta: float = 0.0) -> pd.DataFrame:
    """Compare disjoint target-calibrated projection against strict and raw rows."""

    if variants.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    raw_groups = {
        group_values: group
        for group_values, group in _valid_raw_rows(variants).groupby(
            ["dataset", "selection_metric"],
            dropna=False,
        )
    }
    group_columns = ["dataset", "alignment_method", "alignment_anchor_mode", "selection_metric"]
    for group_values, group in variants.groupby(group_columns, dropna=False):
        group_map = dict(zip(group_columns, group_values, strict=False))
        target_rows = group[group["alignment_target_projection"] == TARGET_CALIBRATED_TARGET_PROJECTION]
        if target_rows.empty:
            continue
        strict_rows = _valid_strict_rows(group)
        raw_rows = raw_groups.get((group_map["dataset"], group_map["selection_metric"]), pd.DataFrame())
        target_row = _best_row(target_rows)
        strict_row = _best_row(strict_rows) if not strict_rows.empty else None
        raw_row = _best_row(raw_rows) if not raw_rows.empty else None
        delta_vs_strict = (
            ""
            if strict_row is None
            else float(target_row["selection_score"]) - float(strict_row["selection_score"])
        )
        delta_vs_raw = "" if raw_row is None else float(target_row["selection_score"]) - float(raw_row["selection_score"])
        if delta_vs_strict == "":
            decision = "target_calibrated_without_strict_pair"
            interpretation = "strict_source_only_pair_missing"
        elif float(delta_vs_strict) > min_delta:
            decision = "target_calibrated_beats_strict_source_only"
            interpretation = "small_target_calibration_can_help_this_alignment"
        elif float(delta_vs_strict) < -min_delta:
            decision = "target_calibrated_hurts_strict_source_only"
            interpretation = "target_calibration_not_sufficient_for_this_alignment"
        else:
            decision = "no_clear_target_calibration_gain_over_strict"
            interpretation = "target_calibration_inconclusive"
        rows.append(
            {
                **group_map,
                "strict_artifact": "" if strict_row is None else strict_row["artifact_name"],
                "strict_value": "" if strict_row is None else strict_row["selection_value"],
                "target_calibrated_artifact": target_row["artifact_name"],
                "target_calibrated_value": target_row["selection_value"],
                "raw_artifact": "" if raw_row is None else raw_row["artifact_name"],
                "raw_value": "" if raw_row is None else raw_row["selection_value"],
                "score_delta_target_calibrated_minus_strict": delta_vs_strict,
                "score_delta_target_calibrated_minus_raw": delta_vs_raw,
                "min_delta": float(min_delta),
                "decision": decision,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_raw_alignment_comparison(variants: pd.DataFrame, *, min_delta: float = 0.0) -> pd.DataFrame:
    """Compare the best alignment variant against the raw/no-alignment row."""

    if variants.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "selection_metric"]
    for group_values, group in variants.groupby(group_columns, dropna=False):
        group_map = dict(zip(group_columns, group_values, strict=False))
        raw_rows = _valid_raw_rows(group)
        aligned_rows = group[
            (~group["alignment_method"].isin(["", "none"]))
            & (group["alignment_valid_for_benchmark"].map(_as_bool))
            & (group["alignment_target_projection"] == STRICT_TARGET_PROJECTION)
        ]
        if raw_rows.empty or aligned_rows.empty:
            continue
        raw_row = _best_row(raw_rows)
        aligned_row = _best_row(aligned_rows)
        delta = float(aligned_row["selection_score"]) - float(raw_row["selection_score"])
        if delta > min_delta:
            decision = "alignment_improves_raw"
            interpretation = "alignment_condition_has_positive_smoke_signal"
        elif delta < -min_delta:
            decision = "alignment_hurts_raw"
            interpretation = "current_alignment_conditions_are_not_helping"
        else:
            decision = "no_clear_alignment_gain"
            interpretation = "alignment_vs_raw_inconclusive"
        rows.append(
            {
                **group_map,
                "raw_artifact": raw_row["artifact_name"],
                "raw_value": raw_row["selection_value"],
                "best_alignment_artifact": aligned_row["artifact_name"],
                "best_alignment_method": aligned_row["alignment_method"],
                "best_alignment_anchor_mode": aligned_row["alignment_anchor_mode"],
                "best_alignment_target_projection": aligned_row["alignment_target_projection"],
                "best_alignment_value": aligned_row["selection_value"],
                "score_delta_alignment_minus_raw": delta,
                "min_delta": float(min_delta),
                "decision": decision,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_alignment_debug_note(
    variants: pd.DataFrame,
    raw_comparison: pd.DataFrame,
    anchor_comparison: pd.DataFrame,
    oracle_comparison: pd.DataFrame,
    target_calibrated_comparison: pd.DataFrame,
    *,
    metric: str,
    fixed_time: float | None,
) -> str:
    lines = [
        "# OpenNeuro Alignment Debug Summary",
        "",
        f"- Selection metric: `{metric}`",
        f"- Selection mode: `{'nearest_fixed_time' if fixed_time is not None else 'best_time'}`",
    ]
    if fixed_time is not None:
        lines.append(f"- Requested fixed time: `{fixed_time}`")
    lines.extend(
        [
            f"- Variant artifacts: `{len(variants)}`",
            "",
            "## Alignment Versus Raw",
        ]
    )
    if raw_comparison.empty:
        lines.append("No matched raw/no-alignment versus alignment pairs were available.")
    else:
        for row in raw_comparison.itertuples(index=False):
            lines.append(
                "- "
                f"{row.dataset}: `{row.decision}` ({row.interpretation}), "
                f"delta={row.score_delta_alignment_minus_raw:.4g}, "
                f"best aligned=`{row.best_alignment_method}/{row.best_alignment_anchor_mode}`."
            )
    lines.extend(
        [
            "",
            "## Anchor Semantics",
        ]
    )
    if anchor_comparison.empty:
        lines.append("No matched class-repetition versus true-identity anchor pairs were available.")
    else:
        for row in anchor_comparison.itertuples(index=False):
            lines.append(
                "- "
                f"{row.dataset} / {row.alignment_method} / {row.alignment_target_projection}: "
                f"`{row.decision}` ({row.interpretation}), "
                f"delta={row.score_delta_identity_minus_class_repetition:.4g}, "
                f"best identity=`{row.best_identity_anchor_mode}`."
            )
    lines.extend(["", "## Oracle Target Calibration"])
    if oracle_comparison.empty:
        lines.append("No matched strict versus oracle target-calibrated pairs were available.")
    else:
        for row in oracle_comparison.itertuples(index=False):
            lines.append(
                "- "
                f"{row.dataset} / {row.alignment_method} / {row.alignment_anchor_mode}: "
                f"`{row.decision}` ({row.interpretation}), "
                f"delta={row.score_delta_oracle_minus_strict:.4g}."
            )
    lines.extend(["", "## Disjoint Target Calibration"])
    if target_calibrated_comparison.empty:
        lines.append("No matched disjoint target-calibrated runs were available.")
    else:
        for row in target_calibrated_comparison.itertuples(index=False):
            delta_strict = row.score_delta_target_calibrated_minus_strict
            delta_text = "missing strict pair" if delta_strict == "" else f"delta_vs_strict={float(delta_strict):.4g}"
            lines.append(
                "- "
                f"{row.dataset} / {row.alignment_method} / {row.alignment_anchor_mode}: "
                f"`{row.decision}` ({row.interpretation}), {delta_text}."
            )
    if not variants.empty:
        collapse = (
            variants["diagnostic_uses_channel_projection_collapse_any"]
            if "diagnostic_uses_channel_projection_collapse_any" in variants.columns
            else pd.Series(False, index=variants.index)
        )
        collapse_count = collapse.map(_as_bool).sum()
        reduction = (
            variants["diagnostic_uses_dimensionality_reduction_any"]
            if "diagnostic_uses_dimensionality_reduction_any" in variants.columns
            else pd.Series(False, index=variants.index)
        )
        reduction_count = reduction.map(_as_bool).sum()
        diagnostic_count = (
            int(variants["alignment_diagnostics_present"].map(_as_bool).sum())
            if "alignment_diagnostics_present" in variants.columns
            else 0
        )
        lines.extend(
            [
                "",
                "## Dimensionality Flags",
                f"- Artifacts with alignment diagnostics: `{diagnostic_count}/{len(variants)}`",
                f"- Artifacts with any channel projection collapse: `{int(collapse_count)}/{len(variants)}`",
                f"- Artifacts with any aligned-space dimensionality reduction: `{int(reduction_count)}/{len(variants)}`",
                "- Inspect `alignment_variant_summary.csv` for `diagnostic_actual_components_median`, "
                "`diagnostic_n_alignment_rows_median`, and anchor correlation before/after.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run_alignment_comparison(
    paths: Sequence[str | Path],
    *,
    out_dir: str | Path,
    metric: str = "balanced_accuracy",
    fixed_time: float | None = None,
    min_delta: float = 0.0,
) -> dict[str, Path]:
    output_dirs = discover_output_dirs(paths)
    if not output_dirs:
        raise FileNotFoundError("No OpenNeuro output directories with decode/time_decode_summary.csv were found.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    variants = build_variant_summary(output_dirs, metric=metric, fixed_time=fixed_time)
    raw_comparison = build_raw_alignment_comparison(variants, min_delta=min_delta)
    anchor_comparison = build_anchor_comparison(variants, min_delta=min_delta)
    oracle_comparison = build_oracle_comparison(variants, min_delta=min_delta)
    target_calibrated_comparison = build_target_calibrated_comparison(variants, min_delta=min_delta)
    note = build_alignment_debug_note(
        variants,
        raw_comparison,
        anchor_comparison,
        oracle_comparison,
        target_calibrated_comparison,
        metric=metric,
        fixed_time=fixed_time,
    )

    paths_out = {
        "variant_summary": out / "alignment_variant_summary.csv",
        "raw_comparison": out / "alignment_vs_raw_comparison.csv",
        "anchor_comparison": out / "alignment_anchor_comparison.csv",
        "oracle_comparison": out / "alignment_oracle_comparison.csv",
        "target_calibrated_comparison": out / "alignment_target_calibrated_comparison.csv",
        "note": out / "alignment_debug_summary.md",
    }
    variants.to_csv(paths_out["variant_summary"], index=False)
    raw_comparison.to_csv(paths_out["raw_comparison"], index=False)
    anchor_comparison.to_csv(paths_out["anchor_comparison"], index=False)
    oracle_comparison.to_csv(paths_out["oracle_comparison"], index=False)
    target_calibrated_comparison.to_csv(paths_out["target_calibrated_comparison"], index=False)
    paths_out["note"].write_text(note, encoding="utf-8")
    return paths_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="OpenNeuro output dirs or parent dirs to scan.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for comparison CSVs and Markdown note.")
    parser.add_argument("--metric", default="balanced_accuracy", help="Metric from time_decode_summary.csv to compare.")
    parser.add_argument("--fixed-time", type=float, help="Use the nearest decoded time to this value instead of each run's best time.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum signed score delta required for a positive decision.")
    args = parser.parse_args(argv)

    written = run_alignment_comparison(
        args.paths,
        out_dir=args.out_dir,
        metric=args.metric,
        fixed_time=args.fixed_time,
        min_delta=args.min_delta,
    )
    for name, path in written.items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
