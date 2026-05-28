"""Resilient OpenNeuro MEG staging helpers.

The standard ``neureptrace.openneuro_meg stage`` command is intentionally strict:
any subject-level staging error aborts the run.  This module adds an opt-in
wrapper for long OpenNeuro sweeps where a single corrupt FIF file or malformed
sidecar should be recorded and skipped while the rest of the cohort is staged.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from neureptrace import openneuro_meg as openneuro
from neureptrace.openneuro_meg import StageResult


@dataclass(frozen=True)
class StageFailure:
    """Subject-level OpenNeuro staging failure."""

    dataset_id: str
    subject: str
    error_type: str
    error: str


def _split_csv_or_space(value: str) -> list[str]:
    return [part.strip() for chunk in str(value).split(",") for part in chunk.split() if part.strip()]


def _baseline(value: str) -> tuple[float | None, float | None] | None:
    text = str(value).strip().lower()
    if text in {"none", "off", "false"}:
        return None
    parts = _split_csv_or_space(value)
    if len(parts) != 2:
        raise ValueError("--baseline must be 'none' or two values such as 'None,0'.")
    start, stop = parts
    return (None if start.lower() == "none" else float(start), None if stop.lower() == "none" else float(stop))


def stage_dataset_resilient(
    dataset_id: str,
    *,
    bids_root: Path,
    staged_dir: Path,
    subjects: str | Iterable[str | int] | None = None,
    runs: str | Iterable[str] | None = None,
    label_column: str | None = None,
    include_labels: Sequence[str] | None = None,
    max_events_per_label: int | None = None,
    selection: str = "random",
    seed: int = 13,
    tmin: float | None = None,
    tmax: float | None = None,
    baseline: tuple[float | None, float | None] | None = (None, 0.0),
    picks: str = "meg",
    resample_sfreq: float | None = None,
    on_mismatch: str = "warn",
    overwrite: bool = False,
    skip_failed_subjects: bool = False,
) -> tuple[list[StageResult], list[StageFailure]]:
    """Stage a dataset while optionally recording and skipping failed subjects."""

    spec = openneuro.DATASET_SPECS[openneuro.normalize_dataset_id(dataset_id)]
    results: list[StageResult] = []
    failures: list[StageFailure] = []

    for subject in openneuro.parse_subjects(spec, subjects):
        subject_name = openneuro.subject_label(spec, subject)
        try:
            result = openneuro.stage_subject(
                spec.dataset_id,
                bids_root=bids_root,
                staged_dir=staged_dir,
                subject=subject,
                runs=runs,
                label_column=label_column,
                include_labels=include_labels,
                max_events_per_label=max_events_per_label,
                selection=selection,
                seed=seed,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks=picks,
                resample_sfreq=resample_sfreq,
                on_mismatch=on_mismatch,
                overwrite=overwrite,
            )
        except Exception as exc:
            if not skip_failed_subjects:
                raise
            failures.append(StageFailure(spec.dataset_id, subject_name, type(exc).__name__, str(exc)))
        else:
            results.append(result)

    return results, failures


def write_stage_failures(failures: Sequence[StageFailure], out_path: Path) -> None:
    """Write skipped subject diagnostics as CSV."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "subject", "error_type", "error"])
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "dataset_id": failure.dataset_id,
                    "subject": failure.subject,
                    "error_type": failure.error_type,
                    "error": failure.error,
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(openneuro.DATASET_SPECS))
    parser.add_argument("--subjects", default="all", help="Subject ids/ranges, or all.")
    parser.add_argument("--runs", default="all", help="Run ids, comma/space separated, or all.")
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--failures-out", type=Path, help="Optional CSV file for skipped subject diagnostics.")
    parser.add_argument("--label-column")
    parser.add_argument("--include-label", action="append", dest="include_labels")
    parser.add_argument("--max-events-per-label", type=int)
    parser.add_argument("--selection", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--baseline", default="None,0")
    parser.add_argument("--picks", default="meg")
    parser.add_argument("--resample-sfreq", type=float)
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--on-mismatch", choices=["raise", "warn", "ignore"], default="warn")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed-subjects", action="store_true", help="Continue after subject-level staging errors and record them as failures.")
    parser.add_argument("--fail-on-skipped-subject", action="store_true", help="Return non-zero when any subject was skipped.")
    args = parser.parse_args(argv)

    include_labels = None
    if args.include_labels:
        include_labels = [label for value in args.include_labels for label in _split_csv_or_space(value)]

    results, failures = stage_dataset_resilient(
        args.dataset,
        bids_root=args.bids_root,
        staged_dir=args.staged_dir,
        subjects=args.subjects,
        runs=args.runs,
        label_column=args.label_column,
        include_labels=include_labels,
        max_events_per_label=args.max_events_per_label,
        selection=args.selection,
        seed=args.seed,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=_baseline(args.baseline),
        picks=args.picks,
        resample_sfreq=0.0 if args.no_resample else args.resample_sfreq,
        on_mismatch=args.on_mismatch,
        overwrite=args.overwrite,
        skip_failed_subjects=args.skip_failed_subjects,
    )

    for result in results:
        print(f"Staged {result.subject}: {result.n_trials} trials, labels={','.join(result.labels)}, runs={','.join(result.runs)}")

    if args.summary_out is not None:
        openneuro.write_stage_summary(results, args.summary_out)
        print(f"Wrote OpenNeuro MEG stage summary: {args.summary_out}")

    if args.failures_out is not None:
        write_stage_failures(failures, args.failures_out)
        print(f"Wrote OpenNeuro MEG stage failures: {args.failures_out}")

    for failure in failures:
        print(f"Skipped {failure.subject}: {failure.error_type}: {failure.error}", file=sys.stderr)

    if failures and not results:
        print("No OpenNeuro MEG subjects were staged successfully.", file=sys.stderr)
        return 1
    if failures and args.fail_on_skipped_subject:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
