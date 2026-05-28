"""Resilient OpenNeuro MEG staging helpers.

This module wraps the existing per-subject OpenNeuro staging code with
failure accounting so large workflow runs can continue when a selected
subject has a corrupt or unreadable raw file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from neureptrace.openneuro_meg import (
    DATASET_SPECS,
    StageResult,
    _baseline,
    _split_csv_or_space,
    normalize_dataset_id,
    parse_subjects,
    stage_subject,
    subject_label,
    write_stage_summary,
)


@dataclass(frozen=True)
class StageFailure:
    """Description of one subject that could not be staged."""

    dataset_id: str
    subject: str
    error_type: str
    error: str


def participant_override_token(subject: int | str) -> str:
    """Return a config-friendly participant token for a staged subject.

    The OpenNeuro decode configs use numeric participant placeholders such as
    ``{subject02d}`` and ``{subject03d}``.  A user may pass either ``6`` or a
    BIDS label such as ``sub-06`` to staging; for decode overrides we normalize
    BIDS labels back to their numeric token when possible.
    """

    text = str(subject).strip()
    if text.startswith("sub-"):
        suffix = text.removeprefix("sub-")
        if suffix.isdigit():
            return str(int(suffix))
    return text


def successful_subjects_override(subjects: Sequence[int | str]) -> str:
    """Return a compact participants.ids override for successful subjects."""

    return ",".join(participant_override_token(subject) for subject in subjects)


def write_stage_failure_summary(failures: Sequence[StageFailure], out_path: Path) -> None:
    """Write failed subjects to a CSV file."""

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


def _append_github_env(path: Path, *, successful_subjects: str, n_splits: int, failed_subjects: Sequence[str]) -> None:
    """Append successful cohort variables for subsequent GitHub Actions steps."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"OPENNEURO_SUBJECTS={successful_subjects}\n")
        handle.write(f"OPENNEURO_N_SPLITS={n_splits}\n")
        handle.write(f"OPENNEURO_FAILED_SUBJECTS={','.join(failed_subjects)}\n")


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
) -> tuple[list[StageResult], list[StageFailure], tuple[int | str, ...]]:
    """Stage subjects while optionally skipping failed subjects.

    Returns successful staging results, failed subject records, and the original
    participant tokens for the successful subjects.  The third return value is
    intended for ``participants.ids`` overrides in the subsequent decode step.
    """

    spec = DATASET_SPECS[normalize_dataset_id(dataset_id)]
    results: list[StageResult] = []
    failures: list[StageFailure] = []
    successful_subjects: list[int | str] = []

    for subject in parse_subjects(spec, subjects):
        try:
            result = stage_subject(
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
            failures.append(
                StageFailure(
                    dataset_id=spec.dataset_id,
                    subject=subject_label(spec, subject),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            continue
        results.append(result)
        successful_subjects.append(subject)

    return results, failures, tuple(successful_subjects)


def _add_dataset_subject_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--subjects", default="all", help="Subject ids/ranges, or all.")
    parser.add_argument("--runs", default="all", help="Run ids, comma/space separated, or all.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_parser = subparsers.add_parser(
        "stage",
        help="Stage selected OpenNeuro MEG subjects and optionally skip failed subjects.",
    )
    _add_dataset_subject_run_args(stage_parser)
    stage_parser.add_argument("--bids-root", type=Path, required=True)
    stage_parser.add_argument("--staged-dir", type=Path, required=True)
    stage_parser.add_argument("--summary-out", type=Path)
    stage_parser.add_argument("--failure-summary-out", type=Path)
    stage_parser.add_argument("--successful-subjects-out", type=Path)
    stage_parser.add_argument("--github-env", type=Path, help="Append successful cohort variables to this GitHub Actions env file.")
    stage_parser.add_argument("--skip-failed-subjects", action="store_true")
    stage_parser.add_argument("--min-successful-subjects", type=int, default=1)
    stage_parser.add_argument("--label-column")
    stage_parser.add_argument("--include-label", action="append", dest="include_labels")
    stage_parser.add_argument("--max-events-per-label", type=int)
    stage_parser.add_argument("--selection", choices=["first", "random"], default="random")
    stage_parser.add_argument("--seed", type=int, default=13)
    stage_parser.add_argument("--tmin", type=float)
    stage_parser.add_argument("--tmax", type=float)
    stage_parser.add_argument("--baseline", default="None,0")
    stage_parser.add_argument("--picks", default="meg")
    stage_parser.add_argument("--resample-sfreq", type=float)
    stage_parser.add_argument("--no-resample", action="store_true")
    stage_parser.add_argument("--on-mismatch", choices=["raise", "warn", "ignore"], default="warn")
    stage_parser.add_argument("--overwrite", action="store_true")
    stage_parser.set_defaults(func=_main_stage)

    args = parser.parse_args(argv)
    return args.func(args)


def _main_stage(args: argparse.Namespace) -> int:
    include_labels = None
    if args.include_labels:
        include_labels = [label for value in args.include_labels for label in _split_csv_or_space(value)]
    resample_sfreq = 0.0 if args.no_resample else args.resample_sfreq

    results, failures, successful_subjects = stage_dataset_resilient(
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
        resample_sfreq=resample_sfreq,
        on_mismatch=args.on_mismatch,
        overwrite=args.overwrite,
        skip_failed_subjects=args.skip_failed_subjects,
    )

    for result in results:
        print(f"Staged {result.subject}: {result.n_trials} trials, labels={','.join(result.labels)}, runs={','.join(result.runs)}")
    for failure in failures:
        print(f"Failed {failure.subject}: {failure.error_type}: {failure.error}", file=sys.stderr)

    if args.summary_out is not None:
        write_stage_summary(results, args.summary_out)
        print(f"Wrote OpenNeuro MEG stage summary: {args.summary_out}")

    failure_summary_out = args.failure_summary_out
    if failure_summary_out is None and args.summary_out is not None and failures:
        failure_summary_out = args.summary_out.with_name("stage_failures.csv")
    if failure_summary_out is not None:
        write_stage_failure_summary(failures, failure_summary_out)
        print(f"Wrote OpenNeuro MEG stage failure summary: {failure_summary_out}")

    successful_override = successful_subjects_override(successful_subjects)
    if args.successful_subjects_out is not None:
        args.successful_subjects_out.parent.mkdir(parents=True, exist_ok=True)
        args.successful_subjects_out.write_text(successful_override + "\n", encoding="utf-8")
    if args.github_env is not None:
        _append_github_env(
            args.github_env,
            successful_subjects=successful_override,
            n_splits=len(successful_subjects),
            failed_subjects=[failure.subject for failure in failures],
        )

    if len(results) < args.min_successful_subjects:
        print(
            f"Only {len(results)} subject(s) staged successfully; required at least {args.min_successful_subjects}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
