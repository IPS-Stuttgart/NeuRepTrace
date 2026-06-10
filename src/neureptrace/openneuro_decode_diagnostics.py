"""Summarize and diagnose OpenNeuro MEG LOSO workflow outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.loso_observation_diagnostics import write_loso_observation_diagnostics

METRIC_COLUMNS = ("balanced_accuracy", "accuracy", "top2_accuracy", "top3_accuracy", "log_loss", "brier", "ece")
MINIMIZE_METRICS = {"log_loss", "brier", "ece"}
NULL_CHANCE_TOLERANCE = 0.03
POSITIVE_CHANCE_MARGIN = 0.05
SUMMARY_PROVENANCE_COLUMNS = (
    "alignment_method",
    "alignment_anchor_mode",
    "alignment_anchor_column",
    "alignment_repetition_cap",
    "alignment_components",
    "alignment_times",
    "alignment_target_projection",
    "alignment_target_calibration_per_anchor",
    "alignment_target_calibration_seed",
    "source_decoders",
    "ensemble_weights",
    "ensemble_source_temperatures",
    "ensemble_score_mode",
    "ensemble_source_baseline_debiasing",
    "ensemble_baseline_window_start",
    "ensemble_baseline_window_stop",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    return None if numeric is None else int(numeric)


def _as_percent(value: Any) -> float | str:
    numeric = _as_float(value)
    return "" if numeric is None else 100.0 * numeric


def _join_manifest_list(value: Any) -> str:
    if isinstance(value, list | tuple):
        return "|".join(str(item) for item in value if str(item))
    return "" if value is None else str(value)


def _top_k_evidence_role(interpretation: Any) -> str:
    normalized = str(interpretation).strip().lower()
    if normalized == "automatic_ceiling":
        return "uninformative_automatic_ceiling"
    if normalized == "informative":
        return "chance_adjusted_supporting"
    return ""


def _quality_decision(
    *,
    decode_summary_exists: bool,
    quality_summary_exists: bool,
    label_shuffle_control: bool,
    fixed_balanced_minus_chance: Any,
    subjects_fixed_above_chance: Any,
    n_subjects: Any,
) -> str:
    if not decode_summary_exists:
        return "missing_decode_summary"
    if not quality_summary_exists:
        return "missing_quality_summary"

    delta = _as_float(fixed_balanced_minus_chance)
    if delta is None:
        return "missing_fixed_balanced_delta"

    if label_shuffle_control:
        if abs(delta) <= NULL_CHANCE_TOLERANCE:
            return "null_near_chance"
        if delta > NULL_CHANCE_TOLERANCE:
            return "null_above_chance"
        return "null_below_chance"

    subjects_above = _as_int(subjects_fixed_above_chance)
    subject_count = _as_int(n_subjects)
    majority_above = subjects_above is not None and subject_count is not None and subjects_above > subject_count / 2.0
    if delta >= POSITIVE_CHANCE_MARGIN and majority_above:
        return "promising_above_chance_consistent"
    if delta >= POSITIVE_CHANCE_MARGIN:
        return "promising_above_chance"
    if delta > 0:
        return "weak_above_chance"
    return "not_above_chance"


def _relative_files(root: Path, *, max_files: int = 200) -> list[str]:
    if not root.exists():
        return []
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if len(files) > max_files:
        return [*files[:max_files], f"... {len(files) - max_files} more file(s) omitted"]
    return files


def _csv_shape(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    frame = pd.read_csv(path)
    return {"exists": True, "rows": int(len(frame)), "columns": list(frame.columns)}


def _compact_unique_value(values: pd.Series) -> str:
    unique_values = tuple(
        str(value).strip()
        for value in values.dropna().astype(str)
        if str(value).strip()
    )
    return "|".join(dict.fromkeys(unique_values))


def _summary_provenance(summary_path: Path) -> dict[str, str]:
    if not summary_path.is_file() or summary_path.stat().st_size <= 0:
        return {}
    frame = pd.read_csv(summary_path)
    return {
        column: _compact_unique_value(frame[column])
        for column in SUMMARY_PROVENANCE_COLUMNS
        if column in frame.columns
    }


def _provenance_value(
    manifest: dict[str, Any],
    summary_provenance: dict[str, str],
    manifest_key: str,
    summary_key: str | None = None,
) -> Any:
    value = manifest.get(manifest_key, "")
    if value not in {"", None}:
        return value
    return summary_provenance.get(summary_key or manifest_key, "")


def _concat_existing_csvs(
    output_dirs: Sequence[Path],
    relative_path: str,
    out_path: Path,
    *,
    drop_duplicate_columns: Sequence[str] = (),
) -> Path | None:
    frames = []
    for output_dir in output_dirs:
        path = output_dir / relative_path
        if path.is_file() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if drop_duplicate_columns and all(column in combined.columns for column in drop_duplicate_columns):
        combined = combined.drop_duplicates(subset=list(drop_duplicate_columns), keep="first").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return out_path


def _aggregate_manifest(output_dirs: Sequence[Path]) -> dict[str, Any]:
    manifests = [_read_json(output_dir / "run_manifest.json") for output_dir in output_dirs]
    first = next((manifest for manifest in manifests if manifest), {})
    dataset = first.get("dataset", "")
    mode = first.get("mode", "")
    artifact_base = first.get("artifact_name") or "-".join(part for part in ("openneuro-meg", dataset, mode) if part)
    if artifact_base.endswith("-shard-aggregate"):
        artifact_name = artifact_base
    else:
        artifact_name = f"{artifact_base}-shard-aggregate"
    outer_test_groups = [str(manifest.get("outer_test_groups", "")).strip() for manifest in manifests if str(manifest.get("outer_test_groups", "")).strip()]
    aggregate = dict(first)
    aggregate.update(
        {
            "artifact_name": artifact_name,
            "shard_count": len(output_dirs),
            "source_output_dirs": [output_dir.as_posix() for output_dir in output_dirs],
            "source_artifacts": [manifest.get("artifact_name", "") for manifest in manifests],
            "source_github_run_ids": [manifest.get("github_run_id", "") for manifest in manifests],
            "aggregate_outer_test_groups": outer_test_groups,
            "outer_test_groups": "|".join(outer_test_groups),
            "aggregate_source": "openneuro_decode_diagnostics",
        }
    )
    return aggregate


def _diagnostics_best_time(output_dirs: Sequence[Path], explicit_best_time: float | None) -> float | None:
    if explicit_best_time is not None:
        return explicit_best_time
    for output_dir in output_dirs:
        value = _read_json(output_dir / "run_manifest.json").get("diagnostics_best_time", "")
        if value not in {"", None}:
            numeric = _as_float(value)
            if numeric is not None:
                return numeric
    return None


def _stage_summary(stage_summary_path: Path) -> dict[str, Any]:
    if not stage_summary_path.is_file():
        return {"exists": False}
    frame = pd.read_csv(stage_summary_path)
    summary: dict[str, Any] = {
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "subject" in frame.columns:
        subjects = [str(subject) for subject in frame["subject"].dropna().unique()]
        summary["subjects"] = subjects
        summary["n_subjects"] = len(subjects)
    if "n_trials" in frame.columns:
        n_trials = pd.to_numeric(frame["n_trials"], errors="coerce")
        summary["total_trials"] = int(n_trials.fillna(0).sum())
        summary["subjects_with_no_trials"] = frame.loc[n_trials.fillna(0) <= 0, "subject"].astype(str).tolist() if "subject" in frame.columns else []
    if "labels" in frame.columns:
        labels = sorted({label for value in frame["labels"].dropna().astype(str) for label in value.split("|") if label})
        summary["labels"] = labels
    return summary


def best_metric_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one best-time row per available metric from a decode summary."""

    if summary.empty or "time" not in summary.columns:
        return pd.DataFrame(columns=["selection_metric", "selection_value"])

    available_metrics = [metric for metric in METRIC_COLUMNS if metric in summary.columns]
    if not available_metrics:
        return pd.DataFrame(columns=["selection_metric", "selection_value"])

    metrics = summary[["time", *available_metrics]].copy()
    metrics["time"] = pd.to_numeric(metrics["time"], errors="coerce")
    for metric in available_metrics:
        metrics[metric] = pd.to_numeric(metrics[metric], errors="coerce")
    metrics = metrics.dropna(subset=["time"])
    if metrics.empty:
        return pd.DataFrame(columns=["selection_metric", "selection_value"])

    by_time = metrics.groupby("time", dropna=False)[available_metrics].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for metric in available_metrics:
        values = by_time[metric].dropna()
        if values.empty:
            continue
        best_index = values.idxmin() if metric in MINIMIZE_METRICS else values.idxmax()
        row = by_time.loc[best_index].to_dict()
        row["selection_metric"] = metric
        row["selection_value"] = float(row[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def workflow_quality_summary(
    output_dir: str | Path,
    diagnostics: dict[str, Any],
    best_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return compact paper-table rows for an OpenNeuro workflow artifact."""

    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "run_manifest.json")
    raw_quality_path = output_dir / "decode" / "diagnostics" / "quality_summary.csv"
    raw_summary_path = output_dir / "decode" / "time_decode_summary.csv"
    best_by_metric = best_rows.set_index("selection_metric") if not best_rows.empty else pd.DataFrame()
    stage_summary = diagnostics.get("stage_summary", {})
    rows = [
        _workflow_quality_row(
            manifest=manifest,
            stage_summary=stage_summary,
            decode_summary=diagnostics.get("decode_summary", {}),
            quality_path=raw_quality_path,
            summary_provenance=_summary_provenance(raw_summary_path),
            best_by_metric=best_by_metric,
            result_variant="raw",
        )
    ]

    smoothed_summary_path = output_dir / "decode" / "temporal_smoothing" / "time_decode_summary.csv"
    smoothed_quality_path = output_dir / "decode" / "temporal_smoothing" / "diagnostics" / "quality_summary.csv"
    if smoothed_summary_path.is_file() or smoothed_quality_path.is_file():
        smoothed_summary = _csv_shape(smoothed_summary_path)
        if smoothed_summary_path.is_file():
            smoothed_best = best_metric_rows(pd.read_csv(smoothed_summary_path)).set_index("selection_metric")
        else:
            smoothed_best = pd.DataFrame()
        rows.append(
            _workflow_quality_row(
                manifest=manifest,
                stage_summary=stage_summary,
                decode_summary=smoothed_summary,
                quality_path=smoothed_quality_path,
                summary_provenance=_summary_provenance(smoothed_summary_path),
                best_by_metric=smoothed_best,
                result_variant="temporal_smoothing",
            )
        )

    response_summary_path = output_dir / "decode" / "response_window" / "time_decode_summary.csv"
    response_quality_path = output_dir / "decode" / "response_window" / "diagnostics" / "quality_summary.csv"
    if response_summary_path.is_file() or response_quality_path.is_file():
        response_summary = _csv_shape(response_summary_path)
        if response_summary_path.is_file():
            response_best = best_metric_rows(pd.read_csv(response_summary_path)).set_index("selection_metric")
        else:
            response_best = pd.DataFrame()
        rows.append(
            _workflow_quality_row(
                manifest=manifest,
                stage_summary=stage_summary,
                decode_summary=response_summary,
                quality_path=response_quality_path,
                summary_provenance=_summary_provenance(response_summary_path),
                best_by_metric=response_best,
                result_variant="response_window",
            )
        )

    return pd.DataFrame(rows)


def _workflow_quality_row(
    *,
    manifest: dict[str, Any],
    stage_summary: dict[str, Any],
    decode_summary: dict[str, Any],
    quality_path: Path,
    summary_provenance: dict[str, str],
    best_by_metric: pd.DataFrame,
    result_variant: str,
) -> dict[str, Any]:
    quality = pd.read_csv(quality_path).iloc[0].to_dict() if quality_path.is_file() else {}
    preferred_metric = "balanced_accuracy" if "balanced_accuracy" in best_by_metric.index else "accuracy"
    has_best_metric = preferred_metric in best_by_metric.index
    best = best_by_metric.loc[preferred_metric].to_dict() if has_best_metric else {}
    decode_summary_exists = bool(decode_summary.get("exists", False))
    quality_summary_exists = bool(quality_path.is_file())
    n_subjects = stage_summary.get("n_subjects", manifest.get("n_subjects", ""))
    label_shuffle_control = _as_bool(manifest.get("label_shuffle_control", ""))
    alignment_method = _provenance_value(manifest, summary_provenance, "alignment_method")
    alignment_anchor_mode = _provenance_value(manifest, summary_provenance, "alignment_anchor_mode")
    alignment_anchor_column = _provenance_value(manifest, summary_provenance, "alignment_anchor_column")
    alignment_target_projection = _provenance_value(manifest, summary_provenance, "alignment_target_projection")
    alignment_enabled = str(alignment_method).strip().lower() not in {"", "none"}
    normalized_target_projection = str(alignment_target_projection).strip().lower()
    oracle_alignment = normalized_target_projection == "oracle_target_calibrated_alignment"
    target_calibrated_alignment = normalized_target_projection == "target_calibrated_alignment"
    alignment_protocol = (
        "oracle_target_calibrated_alignment"
        if oracle_alignment
        else "target_calibrated_alignment"
        if target_calibrated_alignment
        else "strict_source_only"
        if alignment_enabled
        else ""
    )
    quality_decision = _quality_decision(
        decode_summary_exists=decode_summary_exists,
        quality_summary_exists=quality_summary_exists,
        label_shuffle_control=label_shuffle_control,
        fixed_balanced_minus_chance=quality.get("fixed_balanced_minus_chance", ""),
        subjects_fixed_above_chance=quality.get("subjects_fixed_above_chance", ""),
        n_subjects=n_subjects,
    )

    return {
        "dataset": manifest.get("dataset", ""),
        "mode": manifest.get("mode", ""),
        "result_variant": result_variant,
        "artifact_name": manifest.get("artifact_name", ""),
        "shard_count": manifest.get("shard_count", ""),
        "aggregate_outer_test_groups": _join_manifest_list(manifest.get("aggregate_outer_test_groups", "")),
        "source_artifacts": _join_manifest_list(manifest.get("source_artifacts", "")),
        "source_github_run_ids": _join_manifest_list(manifest.get("source_github_run_ids", "")),
        "github_run_id": manifest.get("github_run_id", ""),
        "github_sha": manifest.get("github_sha", ""),
        "runner_type_input": manifest.get("runner_type_input", ""),
        "runner_environment": manifest.get("runner_environment", ""),
        "subjects": manifest.get("subjects", ""),
        "runs": manifest.get("runs", ""),
        "n_subjects_requested": manifest.get("n_subjects", ""),
        "n_subjects_staged": n_subjects,
        "total_trials_staged": stage_summary.get("total_trials", ""),
        "label_shuffle_control": label_shuffle_control,
        "label_shuffle_seed": manifest.get("label_shuffle_seed", ""),
        "time_decode_backend": manifest.get("time_decode_backend", ""),
        "source_calibration": _provenance_value(manifest, summary_provenance, "source_calibration"),
        "alignment_method": alignment_method,
        "alignment_anchor_mode": alignment_anchor_mode,
        "alignment_anchor_column": alignment_anchor_column,
        "alignment_repetition_cap": _provenance_value(manifest, summary_provenance, "alignment_repetition_cap"),
        "alignment_components": _provenance_value(manifest, summary_provenance, "alignment_components"),
        "alignment_times": _provenance_value(manifest, summary_provenance, "alignment_times"),
        "alignment_target_projection": alignment_target_projection,
        "alignment_target_calibration_per_anchor": _provenance_value(
            manifest,
            summary_provenance,
            "alignment_target_calibration_per_anchor",
        ),
        "alignment_target_calibration_seed": _provenance_value(
            manifest,
            summary_provenance,
            "alignment_target_calibration_seed",
        ),
        "alignment_strict_source_only": bool(alignment_enabled and not oracle_alignment and not target_calibrated_alignment),
        "alignment_target_calibrated": bool(target_calibrated_alignment),
        "alignment_oracle_target_calibrated": bool(oracle_alignment),
        "alignment_debug_upper_bound": bool(oracle_alignment),
        "alignment_valid_for_benchmark": bool(not oracle_alignment and not target_calibrated_alignment),
        "alignment_protocol": alignment_protocol,
        "alignment_protocol_note": (
            "debug upper bound only; not valid for benchmark"
            if oracle_alignment
            else "uses disjoint target calibration rows; not valid for strict source-only benchmark"
            if target_calibrated_alignment
            else ""
        ),
        "decoder_override": manifest.get("decoder_override", ""),
        "ensemble_weights": _provenance_value(manifest, summary_provenance, "ensemble_weights"),
        "ensemble_source_decoders": _provenance_value(
            manifest,
            summary_provenance,
            "ensemble_source_decoders",
            "source_decoders",
        ),
        "ensemble_source_temperatures": _provenance_value(manifest, summary_provenance, "ensemble_source_temperatures"),
        "ensemble_score_mode": _provenance_value(manifest, summary_provenance, "ensemble_score_mode"),
        "ensemble_source_baseline_debiasing": _provenance_value(
            manifest,
            summary_provenance,
            "ensemble_source_baseline_debiasing",
        ),
        "ensemble_baseline_window": manifest.get("ensemble_baseline_window", ""),
        "ensemble_baseline_window_start": summary_provenance.get("ensemble_baseline_window_start", ""),
        "ensemble_baseline_window_stop": summary_provenance.get("ensemble_baseline_window_stop", ""),
        "ensemble_min_probability": manifest.get("ensemble_min_probability", ""),
        "temporal_smoothing": _as_bool(manifest.get("temporal_smoothing", "")),
        "temporal_smoothing_fit_window": manifest.get("temporal_smoothing_fit_window", ""),
        "temporal_smoothing_mode": manifest.get("temporal_smoothing_mode", ""),
        "temporal_smoothing_stay_grid_size": manifest.get("temporal_smoothing_stay_grid_size", ""),
        "response_window_ensemble": _as_bool(manifest.get("response_window_ensemble", "")),
        "response_window_mode": manifest.get("response_window_mode", ""),
        "response_window_combine": manifest.get("response_window_combine", ""),
        "response_window_times": manifest.get("response_window_times", ""),
        "decode_summary_exists": decode_summary_exists,
        "quality_summary_exists": quality_summary_exists,
        "quality_decision": quality_decision,
        "null_chance_tolerance": NULL_CHANCE_TOLERANCE,
        "positive_chance_margin": POSITIVE_CHANCE_MARGIN,
        "n_classes": quality.get("n_classes", ""),
        "chance_accuracy": quality.get("chance_accuracy", ""),
        "top2_chance": quality.get("top2_chance", ""),
        "top3_chance": quality.get("top3_chance", ""),
        "top2_interpretation": quality.get("top2_interpretation", ""),
        "top3_interpretation": quality.get("top3_interpretation", ""),
        "top2_evidence_role": _top_k_evidence_role(quality.get("top2_interpretation", "")),
        "top3_evidence_role": _top_k_evidence_role(quality.get("top3_interpretation", "")),
        "fixed_time": quality.get("fixed_time", ""),
        "fixed_accuracy": quality.get("fixed_accuracy", ""),
        "fixed_balanced_accuracy": quality.get("fixed_balanced_accuracy", ""),
        "fixed_balanced_minus_chance": quality.get("fixed_balanced_minus_chance", ""),
        "fixed_balanced_minus_chance_pct": _as_percent(quality.get("fixed_balanced_minus_chance", "")),
        "fixed_top2_accuracy": quality.get("fixed_top2_accuracy", ""),
        "fixed_top2_minus_chance": quality.get("fixed_top2_minus_chance", ""),
        "fixed_top2_minus_chance_pct": _as_percent(quality.get("fixed_top2_minus_chance", "")),
        "fixed_top3_accuracy": quality.get("fixed_top3_accuracy", ""),
        "fixed_top3_minus_chance": quality.get("fixed_top3_minus_chance", ""),
        "fixed_top3_minus_chance_pct": _as_percent(quality.get("fixed_top3_minus_chance", "")),
        "subjects_fixed_above_chance": quality.get("subjects_fixed_above_chance", ""),
        "best_selection_metric": preferred_metric if has_best_metric else "",
        "best_time": best.get("time", ""),
        "best_selection_value": best.get("selection_value", ""),
    }


def summarize_decode_outputs(output_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return diagnostic metadata and best metric rows for an OpenNeuro workflow output directory."""

    output_dir = Path(output_dir)
    decode_dir = output_dir / "decode"
    stage_summary_path = output_dir / "stage_summary.csv"
    summary_path = decode_dir / "time_decode_summary.csv"
    observations_path = decode_dir / "observations.csv"
    calibration_path = decode_dir / "calibration.csv"
    alignment_anchor_availability_path = decode_dir / "alignment_anchor_availability.csv"
    alignment_diagnostics_path = decode_dir / "alignment_diagnostics.csv"

    diagnostics: dict[str, Any] = {
        "output_dir": output_dir.as_posix(),
        "output_dir_exists": output_dir.exists(),
        "files": _relative_files(output_dir),
        "run_manifest": _read_json(output_dir / "run_manifest.json"),
        "stage_summary": _stage_summary(stage_summary_path),
        "decode_summary": _csv_shape(summary_path),
        "observations": _csv_shape(observations_path),
        "calibration": _csv_shape(calibration_path),
        "alignment_anchor_availability": _csv_shape(alignment_anchor_availability_path),
        "alignment_diagnostics": _csv_shape(alignment_diagnostics_path),
        "temporal_smoothing_summary": _csv_shape(decode_dir / "temporal_smoothing" / "time_decode_summary.csv"),
        "temporal_smoothing_observations": _csv_shape(decode_dir / "temporal_smoothing" / "observations.csv"),
    }

    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        best_rows = best_metric_rows(summary)
        diagnostics["decode_summary"]["n_times"] = int(pd.to_numeric(summary.get("time", pd.Series(dtype=float)), errors="coerce").nunique()) if "time" in summary.columns else 0
    else:
        best_rows = pd.DataFrame(columns=["selection_metric", "selection_value"])
        diagnostics["warning"] = f"Missing decode summary: {summary_path.as_posix()}"

    return diagnostics, best_rows


def write_decode_diagnostics(
    output_dir: str | Path,
    *,
    diagnostics_out: str | Path | None = None,
    best_out: str | Path | None = None,
    quality_out: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Write JSON diagnostics and best-time metric CSVs for an OpenNeuro workflow run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, best_rows = summarize_decode_outputs(output_dir)

    diagnostics_path = Path(diagnostics_out) if diagnostics_out is not None else output_dir / "decode_diagnostics.json"
    best_path = Path(best_out) if best_out is not None else output_dir / "decode_best_metrics.csv"
    quality_path = Path(quality_out) if quality_out is not None else output_dir / "workflow_quality_summary.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    best_rows.to_csv(best_path, index=False)
    quality = workflow_quality_summary(output_dir, diagnostics, best_rows)
    quality.to_csv(quality_path, index=False)

    print(f"Wrote OpenNeuro decode diagnostics: {diagnostics_path}")
    if not best_rows.empty:
        print(best_rows.to_string(index=False))
        print(f"Wrote OpenNeuro best-metric table: {best_path}")
    else:
        print(diagnostics.get("warning", "No best-metric rows were available."))
    print(f"Wrote OpenNeuro workflow quality summary: {quality_path}")
    return diagnostics, best_rows


def aggregate_workflow_outputs(
    output_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path,
    diagnostics_best_time: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Aggregate sharded OpenNeuro workflow outputs into one diagnosable directory."""

    source_dirs = [Path(output_dir) for output_dir in output_dirs]
    if not source_dirs:
        raise ValueError("At least one source output directory is required.")
    aggregate_dir = Path(out_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    decode_dir = aggregate_dir / "decode"

    manifest = _aggregate_manifest(source_dirs)
    (aggregate_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    stage_summary_path = _concat_existing_csvs(
        source_dirs,
        "stage_summary.csv",
        aggregate_dir / "stage_summary.csv",
        drop_duplicate_columns=("dataset_id", "subject", "epochs_path"),
    )
    summary_path = _concat_existing_csvs(source_dirs, "decode/time_decode_summary.csv", decode_dir / "time_decode_summary.csv")
    observations_path = _concat_existing_csvs(source_dirs, "decode/observations.csv", decode_dir / "observations.csv")
    _concat_existing_csvs(
        source_dirs,
        "decode/alignment_anchor_availability.csv",
        decode_dir / "alignment_anchor_availability.csv",
    )
    _concat_existing_csvs(source_dirs, "decode/alignment_diagnostics.csv", decode_dir / "alignment_diagnostics.csv")

    best_time = _diagnostics_best_time(source_dirs, diagnostics_best_time)
    if observations_path is not None and summary_path is not None:
        write_loso_observation_diagnostics(
            observations_path,
            out_dir=decode_dir / "diagnostics",
            summary_csv=summary_path,
            stage_summary_csv=stage_summary_path,
            best_time=best_time,
        )

    smoothed_dir = decode_dir / "temporal_smoothing"
    smoothed_summary = _concat_existing_csvs(
        source_dirs,
        "decode/temporal_smoothing/time_decode_summary.csv",
        smoothed_dir / "time_decode_summary.csv",
    )
    smoothed_observations = _concat_existing_csvs(
        source_dirs,
        "decode/temporal_smoothing/observations.csv",
        smoothed_dir / "observations.csv",
    )
    if smoothed_observations is not None and smoothed_summary is not None:
        write_loso_observation_diagnostics(
            smoothed_observations,
            out_dir=smoothed_dir / "diagnostics",
            summary_csv=smoothed_summary,
            stage_summary_csv=stage_summary_path,
            best_time=best_time,
        )

    return write_decode_diagnostics(aggregate_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="+", help="OpenNeuro workflow output directory, e.g. outputs/openneuro_ds006629_full.")
    parser.add_argument("--aggregate-out", type=Path, help="Aggregate multiple sharded workflow output directories before writing diagnostics.")
    parser.add_argument("--diagnostics-out", type=Path)
    parser.add_argument("--best-out", type=Path)
    parser.add_argument("--quality-out", type=Path)
    parser.add_argument("--best-time", type=float, help="Fixed time for aggregate LOSO observation diagnostics.")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit status when the decode summary is missing.")
    args = parser.parse_args(argv)

    if args.aggregate_out is not None or len(args.output_dir) > 1:
        if args.aggregate_out is None:
            parser.error("--aggregate-out is required when multiple output directories are provided.")
        diagnostics, _best_rows = aggregate_workflow_outputs(
            args.output_dir,
            out_dir=args.aggregate_out,
            diagnostics_best_time=args.best_time,
        )
    else:
        diagnostics, _best_rows = write_decode_diagnostics(
            args.output_dir[0],
            diagnostics_out=args.diagnostics_out,
            best_out=args.best_out,
            quality_out=args.quality_out,
        )
    return int(args.strict and not diagnostics["decode_summary"].get("exists", False))


if __name__ == "__main__":
    raise SystemExit(main())
