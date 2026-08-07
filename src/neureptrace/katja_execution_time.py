"""Add execution/go-cue-relative timestamps to a Katja press-timing audit.

The raw timing audit stores timestamps on the MEG epoch clock, whose zero is the
fractal-cue onset. Julia's 0--6 s online grid is described as an execution-period
grid. This module subtracts each trial's cue duration and retains both clocks so
the final reference implementation can choose explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neureptrace.katja_spm_feature_cache import load_subject_behavior

if TYPE_CHECKING:
    from collections.abc import Sequence


def add_execution_time_reference(
    timing: pd.DataFrame,
    cue_durations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return timing rows with MEG-cue and execution-relative aliases."""

    timing_required = {
        "subject",
        "trial_id",
        "behavior_time_seconds",
        "trigger_time_seconds",
        "recommended_time_seconds",
    }
    cue_required = {"subject", "trial_id", "cue_duration_seconds"}
    missing_timing = sorted(timing_required.difference(timing.columns))
    missing_cue = sorted(cue_required.difference(cue_durations.columns))
    if missing_timing:
        raise ValueError(f"Timing table is missing required columns: {missing_timing}.")
    if missing_cue:
        raise ValueError(
            f"Cue-duration table is missing required columns: {missing_cue}."
        )

    cue = cue_durations[list(cue_required)].copy()
    cue["subject"] = cue["subject"].astype(str)
    if cue.duplicated(["subject", "trial_id"]).any():
        raise ValueError("Cue-duration table contains duplicate subject/trial rows.")
    durations = cue["cue_duration_seconds"].to_numpy(dtype=float)
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("cue_duration_seconds must be finite and non-negative.")

    frame = timing.copy()
    frame["subject"] = frame["subject"].astype(str)
    frame = frame.merge(
        cue,
        on=["subject", "trial_id"],
        how="left",
        validate="many_to_one",
    )
    if frame["cue_duration_seconds"].isna().any():
        missing = (
            frame.loc[
                frame["cue_duration_seconds"].isna(), ["subject", "trial_id"]
            ]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"Cue duration is unavailable for timing rows: {missing}.")

    cue_seconds = frame["cue_duration_seconds"].to_numpy(dtype=float)
    aliases = (
        ("behavior_time_seconds", "behavior_time"),
        ("trigger_time_seconds", "trigger_time"),
        ("recommended_time_seconds", "recommended_time"),
        ("expected_trigger_time_seconds", "expected_trigger_time"),
    )
    for source, prefix in aliases:
        if source not in frame.columns:
            continue
        values = frame[source].to_numpy(dtype=float)
        frame[f"{prefix}_meg_seconds"] = values
        frame[f"{prefix}_execution_seconds"] = values - cue_seconds

    metadata: dict[str, object] = {
        "format": "neureptrace_katja_dual_time_reference_v1",
        "input_meg_reference": "fractal-cue onset",
        "execution_reference": "go-cue/execution onset",
        "conversion": "execution_seconds = meg_seconds - cue_duration_seconds",
        "recommended_execution_column": "recommended_time_execution_seconds",
        "n_rows": int(frame.shape[0]),
        "n_trials": int(
            frame[["subject", "trial_id"]].drop_duplicates().shape[0]
        ),
        "n_subjects": int(frame["subject"].nunique()),
    }
    return frame, metadata


def load_cue_duration_registry(
    dataset_root: str | Path,
    participants: Sequence[str],
) -> pd.DataFrame:
    """Load one cue duration for every participant and one-based trial ID."""

    root = Path(dataset_root)
    rows: list[pd.DataFrame] = []
    for subject in participants:
        behavior = load_subject_behavior(root / "beh_data" / str(subject))
        durations = np.asarray(behavior["cue_duration_ms"], dtype=float).reshape(-1)
        rows.append(
            pd.DataFrame(
                {
                    "subject": str(subject),
                    "trial_id": np.arange(1, durations.shape[0] + 1, dtype=int),
                    "cue_duration_seconds": durations / 1000.0,
                }
            )
        )
    if not rows:
        raise ValueError("participants must not be empty.")
    return pd.concat(rows, ignore_index=True)


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated participant list must not be empty.")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--press-timing", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--participants", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    timing = pd.read_csv(args.press_timing)
    participants = _parse_csv(args.participants)
    cue = load_cue_duration_registry(args.dataset_root, participants)
    enriched, metadata = add_execution_time_reference(timing, cue)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output.suffix == ".gz" else None
    enriched.to_csv(output, index=False, compression=compression)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output.with_suffix(".metadata.json")
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
