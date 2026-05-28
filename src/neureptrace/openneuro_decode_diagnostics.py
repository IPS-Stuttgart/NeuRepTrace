"""Summarize and diagnose OpenNeuro MEG LOSO workflow outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_COLUMNS = ("balanced_accuracy", "accuracy", "top2_accuracy", "top3_accuracy", "log_loss", "brier", "ece")
MINIMIZE_METRICS = {"log_loss", "brier", "ece"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    """Return one compact paper-table row for an OpenNeuro workflow artifact."""

    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "run_manifest.json")
    quality_path = output_dir / "decode" / "diagnostics" / "quality_summary.csv"
    quality = pd.read_csv(quality_path).iloc[0].to_dict() if quality_path.is_file() else {}
    best_by_metric = best_rows.set_index("selection_metric") if not best_rows.empty else pd.DataFrame()
    preferred_metric = "balanced_accuracy" if "balanced_accuracy" in best_by_metric.index else "accuracy"
    has_best_metric = preferred_metric in best_by_metric.index
    best = best_by_metric.loc[preferred_metric].to_dict() if has_best_metric else {}
    stage_summary = diagnostics.get("stage_summary", {})
    decode_summary = diagnostics.get("decode_summary", {})

    return pd.DataFrame(
        [
            {
                "dataset": manifest.get("dataset", ""),
                "mode": manifest.get("mode", ""),
                "artifact_name": manifest.get("artifact_name", ""),
                "github_run_id": manifest.get("github_run_id", ""),
                "github_sha": manifest.get("github_sha", ""),
                "runner_type_input": manifest.get("runner_type_input", ""),
                "runner_environment": manifest.get("runner_environment", ""),
                "subjects": manifest.get("subjects", ""),
                "runs": manifest.get("runs", ""),
                "n_subjects_requested": manifest.get("n_subjects", ""),
                "n_subjects_staged": stage_summary.get("n_subjects", ""),
                "total_trials_staged": stage_summary.get("total_trials", ""),
                "label_shuffle_control": manifest.get("label_shuffle_control", ""),
                "label_shuffle_seed": manifest.get("label_shuffle_seed", ""),
                "time_decode_backend": manifest.get("time_decode_backend", ""),
                "decoder_override": manifest.get("decoder_override", ""),
                "decode_summary_exists": bool(decode_summary.get("exists", False)),
                "quality_summary_exists": bool(quality_path.is_file()),
                "n_classes": quality.get("n_classes", ""),
                "chance_accuracy": quality.get("chance_accuracy", ""),
                "top2_chance": quality.get("top2_chance", ""),
                "top3_chance": quality.get("top3_chance", ""),
                "top2_interpretation": quality.get("top2_interpretation", ""),
                "top3_interpretation": quality.get("top3_interpretation", ""),
                "fixed_time": quality.get("fixed_time", ""),
                "fixed_accuracy": quality.get("fixed_accuracy", ""),
                "fixed_balanced_accuracy": quality.get("fixed_balanced_accuracy", ""),
                "fixed_balanced_minus_chance": quality.get("fixed_balanced_minus_chance", ""),
                "fixed_top2_accuracy": quality.get("fixed_top2_accuracy", ""),
                "fixed_top2_minus_chance": quality.get("fixed_top2_minus_chance", ""),
                "fixed_top3_accuracy": quality.get("fixed_top3_accuracy", ""),
                "fixed_top3_minus_chance": quality.get("fixed_top3_minus_chance", ""),
                "subjects_fixed_above_chance": quality.get("subjects_fixed_above_chance", ""),
                "best_selection_metric": preferred_metric if has_best_metric else "",
                "best_time": best.get("time", ""),
                "best_selection_value": best.get("selection_value", ""),
            }
        ]
    )


def summarize_decode_outputs(output_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return diagnostic metadata and best metric rows for an OpenNeuro workflow output directory."""

    output_dir = Path(output_dir)
    decode_dir = output_dir / "decode"
    stage_summary_path = output_dir / "stage_summary.csv"
    summary_path = decode_dir / "time_decode_summary.csv"
    observations_path = decode_dir / "observations.csv"
    calibration_path = decode_dir / "calibration.csv"

    diagnostics: dict[str, Any] = {
        "output_dir": output_dir.as_posix(),
        "output_dir_exists": output_dir.exists(),
        "files": _relative_files(output_dir),
        "run_manifest": _read_json(output_dir / "run_manifest.json"),
        "stage_summary": _stage_summary(stage_summary_path),
        "decode_summary": _csv_shape(summary_path),
        "observations": _csv_shape(observations_path),
        "calibration": _csv_shape(calibration_path),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="OpenNeuro workflow output directory, e.g. outputs/openneuro_ds006629_full.")
    parser.add_argument("--diagnostics-out", type=Path)
    parser.add_argument("--best-out", type=Path)
    parser.add_argument("--quality-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="Return a non-zero exit status when the decode summary is missing.")
    args = parser.parse_args(argv)

    diagnostics, _best_rows = write_decode_diagnostics(
        args.output_dir,
        diagnostics_out=args.diagnostics_out,
        best_out=args.best_out,
        quality_out=args.quality_out,
    )
    return int(args.strict and not diagnostics["decode_summary"].get("exists", False))


if __name__ == "__main__":
    raise SystemExit(main())
