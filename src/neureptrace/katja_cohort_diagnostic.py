"""Summarize early-versus-late Katja target behavior from per-target results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_REFERENCE_CONFIGURATION = "nine_single_protocol"
DEFAULT_CANDIDATE_CONFIGURATION = "hybrid_source_model_ensemble6"


def participant_number(identifier: str) -> int:
    """Extract the numeric suffix from a participant ID such as ``s28``."""

    match = re.fullmatch(r"[A-Za-z_-]*([0-9]+)", str(identifier).strip())
    if match is None:
        raise ValueError(f"Cannot extract participant number from {identifier!r}.")
    return int(match.group(1))


def summarize_cohort_shift(
    per_target: pd.DataFrame,
    *,
    reference_configuration: str = DEFAULT_REFERENCE_CONFIGURATION,
    candidate_configuration: str = DEFAULT_CANDIDATE_CONFIGURATION,
    calibration_count: int = 20,
    early_maximum_participant_number: int = 18,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare a candidate with a reference across early and late participants."""

    required = {"configuration", "target", "k", "independent_accuracy"}
    missing = sorted(required.difference(per_target.columns))
    if missing:
        raise ValueError(f"Per-target table is missing columns: {missing!r}.")
    frame = per_target[
        (per_target["k"].astype(int) == int(calibration_count))
        & per_target["configuration"].isin(
            [reference_configuration, candidate_configuration]
        )
    ].copy()
    pivot = frame.pivot(
        index="target",
        columns="configuration",
        values="independent_accuracy",
    )
    expected_columns = {reference_configuration, candidate_configuration}
    if set(pivot.columns) != expected_columns:
        raise ValueError(
            "Per-target table does not contain exactly one reference and candidate "
            f"value for every target; found columns {list(pivot.columns)!r}."
        )
    if pivot.isna().any().any():
        raise ValueError("Reference/candidate target coverage is incomplete.")

    result = pivot.reset_index().rename(
        columns={
            reference_configuration: "reference_accuracy",
            candidate_configuration: "candidate_accuracy",
        }
    )
    result["participant_number"] = result["target"].map(participant_number)
    result["cohort"] = np.where(
        result["participant_number"] <= int(early_maximum_participant_number),
        "early_s05_to_s18",
        "late_s20_to_s28",
    )
    result["candidate_minus_reference"] = (
        result["candidate_accuracy"] - result["reference_accuracy"]
    )
    result = result.sort_values("participant_number").reset_index(drop=True)

    summary = (
        result.groupby("cohort", as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            reference_accuracy_mean=("reference_accuracy", "mean"),
            candidate_accuracy_mean=("candidate_accuracy", "mean"),
            candidate_minus_reference_mean=("candidate_minus_reference", "mean"),
            candidate_minus_reference_min=("candidate_minus_reference", "min"),
            candidate_minus_reference_max=("candidate_minus_reference", "max"),
        )
        .sort_values("cohort")
        .reset_index(drop=True)
    )
    return result, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-configuration",
        default=DEFAULT_REFERENCE_CONFIGURATION,
    )
    parser.add_argument(
        "--candidate-configuration",
        default=DEFAULT_CANDIDATE_CONFIGURATION,
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--early-maximum", type=int, default=18)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    per_target = pd.read_csv(args.per_target)
    targets, cohorts = summarize_cohort_shift(
        per_target,
        reference_configuration=args.reference_configuration,
        candidate_configuration=args.candidate_configuration,
        calibration_count=args.k,
        early_maximum_participant_number=args.early_maximum,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "katja_cohort_diagnostic_per_target.csv", index=False)
    cohorts.to_csv(output / "katja_cohort_diagnostic_summary.csv", index=False)
    metadata: dict[str, Any] = {
        "reference_configuration": args.reference_configuration,
        "candidate_configuration": args.candidate_configuration,
        "k": int(args.k),
        "early_maximum_participant_number": int(args.early_maximum),
        "interpretation_boundary": (
            "Event-conditioned cohort diagnostic; not directly comparable with "
            "Julia's continuous online-window endpoint."
        ),
    }
    (output / "katja_cohort_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(cohorts.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
