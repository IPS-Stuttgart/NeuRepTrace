"""Protocol scaffolding for Julia-compatible Katja online decoding.

This module provides two pieces that do not depend on the still-pending reference
window-label function:

* deterministic trial-level calibration/evaluation manifests for several split
  conventions; and
* fold, subject, and population aggregation for window-level predictions.

It does not create finger/null labels from overlaps. Those labels must be added
by the exact external window function before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

SPLIT_MODES = (
    "nested_rest",
    "independent_rest",
    "fixed_max_complement",
)


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = repr((int(seed), *(str(part) for part in parts))).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def _unique_trial_registry(
    trials: pd.DataFrame,
    *,
    subject_col: str,
    trial_col: str,
    sequence_col: str,
) -> pd.DataFrame:
    required = {subject_col, trial_col, sequence_col}
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Trial table is missing required columns: {missing}.")
    registry = trials[[subject_col, trial_col, sequence_col]].drop_duplicates()
    duplicate_sequences = (
        registry.groupby([subject_col, trial_col], dropna=False)[sequence_col]
        .nunique(dropna=False)
        .reset_index(name="n_sequences")
    )
    invalid = duplicate_sequences[duplicate_sequences["n_sequences"] != 1]
    if not invalid.empty:
        raise ValueError("Each subject/trial identifier must map to exactly one sequence ID.")
    return registry.sort_values([subject_col, sequence_col, trial_col]).reset_index(drop=True)


def build_trial_split_manifest(
    trials: pd.DataFrame,
    *,
    calibration_counts: Sequence[int] = (1, 3, 5, 10, 15, 20),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    mode: str = "nested_rest",
    subject_col: str = "subject",
    trial_col: str = "trial_id",
    sequence_col: str = "sequence_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create trial-level calibration/evaluation assignments.

    ``nested_rest``
        Draw one deterministic permutation per subject, seed, and sequence.
        Calibration uses its first ``k`` trials and evaluation uses the rest.
        Lower-k calibration sets are nested, but evaluation changes with k.

    ``independent_rest``
        Redraw the permutation for every k and evaluate on that k-specific rest.

    ``fixed_max_complement``
        Reserve the largest-k pool first. Lower-k calibration sets are nested;
        the unused part of the maximum pool is marked ``reserved_unused`` and
        every k uses the same evaluation complement.
    """

    split_mode = str(mode).strip().lower()
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES!r}.")
    counts = tuple(sorted({int(value) for value in calibration_counts}))
    seed_values = tuple(int(value) for value in seeds)
    if not counts or counts[0] < 1:
        raise ValueError("calibration_counts must contain positive integers.")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must contain unique integers.")
    registry = _unique_trial_registry(
        trials,
        subject_col=subject_col,
        trial_col=trial_col,
        sequence_col=sequence_col,
    )
    maximum = max(counts)
    rows: list[pd.DataFrame] = []

    for subject, subject_frame in registry.groupby(subject_col, sort=True, dropna=False):
        for seed in seed_values:
            permutations: dict[Any, np.ndarray] = {}
            for sequence_id, sequence_frame in subject_frame.groupby(
                sequence_col, sort=True, dropna=False
            ):
                trial_values = sequence_frame[trial_col].to_numpy()
                if trial_values.shape[0] <= maximum:
                    raise ValueError(
                        f"Subject {subject!r}, sequence {sequence_id!r} needs more than "
                        f"{maximum} trials to leave an evaluation set; got {trial_values.shape[0]}."
                    )
                rng = np.random.default_rng(
                    _stable_seed(seed, split_mode, subject, sequence_id)
                )
                permutations[sequence_id] = rng.permutation(trial_values)

            for calibration_count in counts:
                assignment_parts: list[pd.DataFrame] = []
                for sequence_id, sequence_frame in subject_frame.groupby(
                    sequence_col, sort=True, dropna=False
                ):
                    if split_mode == "independent_rest":
                        rng = np.random.default_rng(
                            _stable_seed(
                                seed,
                                split_mode,
                                subject,
                                sequence_id,
                                calibration_count,
                            )
                        )
                        permutation = rng.permutation(
                            sequence_frame[trial_col].to_numpy()
                        )
                    else:
                        permutation = permutations[sequence_id]
                    calibration = set(permutation[:calibration_count].tolist())
                    maximum_pool = set(permutation[:maximum].tolist())
                    part = sequence_frame.copy()
                    if split_mode == "fixed_max_complement":
                        roles = np.asarray(
                            [
                                "calibration"
                                if value in calibration
                                else (
                                    "reserved_unused"
                                    if value in maximum_pool
                                    else "evaluation"
                                )
                                for value in part[trial_col].tolist()
                            ],
                            dtype=object,
                        )
                    else:
                        roles = np.asarray(
                            [
                                "calibration"
                                if value in calibration
                                else "evaluation"
                                for value in part[trial_col].tolist()
                            ],
                            dtype=object,
                        )
                    part["seed"] = int(seed)
                    part["k"] = int(calibration_count)
                    part["split_role"] = roles
                    part["split_mode"] = split_mode
                    assignment_parts.append(part)
                rows.append(pd.concat(assignment_parts, ignore_index=True))

    manifest = pd.concat(rows, ignore_index=True).sort_values(
        [subject_col, "seed", "k", sequence_col, trial_col]
    )
    for keys, group in manifest.groupby(
        [subject_col, "seed", "k"], sort=True, dropna=False
    ):
        calibration = group[group["split_role"] == "calibration"]
        evaluation = group[group["split_role"] == "evaluation"]
        if calibration.empty or evaluation.empty:
            raise RuntimeError(f"Split {keys!r} has an empty calibration or evaluation set.")
        if set(calibration[trial_col]).intersection(evaluation[trial_col]):
            raise RuntimeError(f"Split {keys!r} has calibration/evaluation trial overlap.")
        per_sequence = calibration.groupby(sequence_col, dropna=False)[trial_col].nunique()
        expected = int(group["k"].iloc[0])
        if not np.all(per_sequence.to_numpy() == expected):
            raise RuntimeError(f"Split {keys!r} does not contain k trials per sequence.")

    metadata = {
        "format": "neureptrace_katja_trial_split_manifest_v1",
        "mode": split_mode,
        "calibration_counts": list(counts),
        "seeds": list(seed_values),
        "subject_column": subject_col,
        "trial_column": trial_col,
        "sequence_column": sequence_col,
        "n_subjects": int(registry[subject_col].nunique(dropna=False)),
        "n_trials": int(registry.shape[0]),
        "exact_julia_split_status": "pending_reference_split_function",
    }
    return manifest.reset_index(drop=True), metadata


def _accuracy(true_values: pd.Series, predicted_values: pd.Series) -> float:
    if true_values.empty:
        return np.nan
    return float(np.mean(true_values.to_numpy() == predicted_values.to_numpy()))


def score_window_predictions(
    predictions: pd.DataFrame,
    *,
    subject_col: str = "subject",
    seed_col: str = "seed",
    k_col: str = "k",
    true_label_col: str = "true_label",
    predicted_label_col: str = "predicted_label",
    trial_col: str = "trial_id",
    press_mask_col: str | None = None,
    null_label: str | int | float | None = "null",
    true_sequence_col: str | None = "true_sequence_id",
    predicted_sequence_col: str | None = "predicted_sequence_id",
    true_position_col: str | None = "true_serial_position",
    predicted_position_col: str | None = "predicted_serial_position",
    true_overlap_col: str | None = "true_overlap_ratio",
    predicted_overlap_col: str | None = "predicted_overlap_ratio",
) -> pd.DataFrame:
    """Calculate one metric row per subject, seed, and k fold."""

    required = {
        subject_col,
        seed_col,
        k_col,
        true_label_col,
        predicted_label_col,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}.")
    if predictions[list(required)].isna().any().any():
        raise ValueError("Fold identifiers and primary labels must not be missing.")

    frame = predictions.copy()
    if press_mask_col is not None:
        if press_mask_col not in frame.columns:
            raise ValueError(f"press_mask_col {press_mask_col!r} is absent.")
        frame["_is_press"] = frame[press_mask_col].astype(bool)
    elif null_label is not None:
        frame["_is_press"] = frame[true_label_col] != null_label
    else:
        frame["_is_press"] = True
    frame["_correct"] = frame[true_label_col] == frame[predicted_label_col]

    fold_rows: list[dict[str, Any]] = []
    fold_columns = [subject_col, seed_col, k_col]
    for keys, group in frame.groupby(fold_columns, sort=True, dropna=False):
        subject, seed, calibration_count = keys
        press = group[group["_is_press"]]
        null = group[~group["_is_press"]]
        row: dict[str, Any] = {
            "subject": subject,
            "seed": int(seed),
            "k": int(calibration_count),
            "n_windows": int(group.shape[0]),
            "overall_accuracy": float(group["_correct"].mean()),
            "n_press_windows": int(press.shape[0]),
            "press_window_accuracy": (
                float(press["_correct"].mean()) if not press.empty else np.nan
            ),
            "n_null_windows": int(null.shape[0]),
            "null_window_accuracy": (
                float(null["_correct"].mean()) if not null.empty else np.nan
            ),
        }
        if trial_col in group.columns:
            trial_scores = group.groupby(trial_col, dropna=False)["_correct"].mean()
            row["n_trials"] = int(trial_scores.shape[0])
            row["trial_macro_accuracy"] = float(trial_scores.mean())
        optional_accuracy_pairs = (
            ("sequence_accuracy", true_sequence_col, predicted_sequence_col),
            ("serial_position_accuracy", true_position_col, predicted_position_col),
        )
        for metric_name, true_col, predicted_col in optional_accuracy_pairs:
            if (
                true_col is not None
                and predicted_col is not None
                and true_col in group.columns
                and predicted_col in group.columns
            ):
                row[metric_name] = _accuracy(group[true_col], group[predicted_col])
        if (
            true_overlap_col is not None
            and predicted_overlap_col is not None
            and true_overlap_col in group.columns
            and predicted_overlap_col in group.columns
        ):
            true_overlap = group[true_overlap_col].to_numpy(dtype=float)
            predicted_overlap = group[predicted_overlap_col].to_numpy(dtype=float)
            if not np.all(np.isfinite(true_overlap)) or not np.all(
                np.isfinite(predicted_overlap)
            ):
                raise ValueError("Overlap-ratio targets and predictions must be finite.")
            row["overlap_ratio_mae"] = float(
                np.mean(np.abs(true_overlap - predicted_overlap))
            )
        fold_rows.append(row)
    return pd.DataFrame(fold_rows).sort_values(["k", "subject", "seed"]).reset_index(drop=True)


def aggregate_fold_scores(
    fold_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Julia-style fold SD and subject-averaged population SEM tables."""

    required = {"subject", "seed", "k"}
    missing = sorted(required.difference(fold_scores.columns))
    if missing:
        raise ValueError(f"Fold-score table is missing required columns: {missing}.")
    metric_columns = [
        column
        for column in fold_scores.columns
        if column not in required
        and not column.startswith("n_")
        and pd.api.types.is_numeric_dtype(fold_scores[column])
    ]
    if not metric_columns:
        raise ValueError("Fold-score table contains no numeric metric columns.")

    julia_rows: list[dict[str, Any]] = []
    for calibration_count, group in fold_scores.groupby("k", sort=True):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            julia_rows.append(
                {
                    "k": int(calibration_count),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd_across_subject_seed_folds": (
                        float(np.std(values, ddof=1)) if values.size > 1 else np.nan
                    ),
                    "n_folds": int(values.size),
                    "n_subjects": int(group["subject"].nunique()),
                    "n_seeds": int(group["seed"].nunique()),
                }
            )
    julia_summary = pd.DataFrame(julia_rows)

    subject_means = (
        fold_scores.groupby(["subject", "k"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values(["k", "subject"])
        .reset_index(drop=True)
    )
    population_rows: list[dict[str, Any]] = []
    for calibration_count, group in subject_means.groupby("k", sort=True):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            population_rows.append(
                {
                    "k": int(calibration_count),
                    "metric": metric,
                    "mean_after_seed_average": float(np.mean(values)),
                    "sem_across_subjects": (
                        float(np.std(values, ddof=1) / np.sqrt(values.size))
                        if values.size > 1
                        else np.nan
                    ),
                    "n_subjects": int(values.size),
                }
            )
    population_summary = pd.DataFrame(population_rows)
    return julia_summary, subject_means, population_summary


def validate_fold_registry(
    fold_scores: pd.DataFrame,
    *,
    expected_subjects: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_k: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate complete subject×seed×k coverage when a registry is known."""

    actual_subjects = tuple(sorted(fold_scores["subject"].astype(str).unique()))
    actual_seeds = tuple(sorted(int(value) for value in fold_scores["seed"].unique()))
    actual_k = tuple(sorted(int(value) for value in fold_scores["k"].unique()))
    expected = {
        "subjects": (
            actual_subjects
            if expected_subjects is None
            else tuple(sorted(str(value) for value in expected_subjects))
        ),
        "seeds": (
            actual_seeds
            if expected_seeds is None
            else tuple(sorted(int(value) for value in expected_seeds))
        ),
        "k": (
            actual_k
            if expected_k is None
            else tuple(sorted(int(value) for value in expected_k))
        ),
    }
    observed = set(
        zip(
            fold_scores["subject"].astype(str),
            fold_scores["seed"].astype(int),
            fold_scores["k"].astype(int),
            strict=True,
        )
    )
    required = {
        (subject, seed, calibration_count)
        for subject in expected["subjects"]
        for seed in expected["seeds"]
        for calibration_count in expected["k"]
    }
    missing = sorted(required.difference(observed))
    unexpected = sorted(observed.difference(required))
    duplicate_count = int(
        fold_scores.duplicated(["subject", "seed", "k"], keep=False).sum()
    )
    if missing or unexpected or duplicate_count:
        raise ValueError(
            "Incomplete or duplicated fold registry: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"duplicate_rows={duplicate_count}."
        )
    return {
        "subjects": list(expected["subjects"]),
        "seeds": list(expected["seeds"]),
        "k": list(expected["k"]),
        "n_folds": len(required),
    }


def _parse_csv(value: str, *, cast=str) -> tuple[Any, ...]:
    result = tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated values must not be empty.")
    return result


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("splits", help="Build candidate trial split manifests.")
    split_parser.add_argument("--trials", required=True)
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--metadata-output")
    split_parser.add_argument("--calibration-counts", default="1,3,5,10,15,20")
    split_parser.add_argument("--seeds", default="0,1,2,3,4")
    split_parser.add_argument("--mode", choices=SPLIT_MODES, default="nested_rest")

    report_parser = subparsers.add_parser("report", help="Aggregate labeled window predictions.")
    report_parser.add_argument("--predictions", required=True)
    report_parser.add_argument("--output-dir", required=True)
    report_parser.add_argument("--press-mask-col")
    report_parser.add_argument("--null-label", default="null")
    report_parser.add_argument("--expected-subjects")
    report_parser.add_argument("--expected-seeds")
    report_parser.add_argument("--expected-k")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "splits":
        trials = pd.read_csv(args.trials)
        manifest, metadata = build_trial_split_manifest(
            trials,
            calibration_counts=_parse_csv(args.calibration_counts, cast=int),
            seeds=_parse_csv(args.seeds, cast=int),
            mode=args.mode,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        compression = "gzip" if output.suffix == ".gz" else None
        manifest.to_csv(output, index=False, compression=compression)
        metadata_path = (
            Path(args.metadata_output)
            if args.metadata_output
            else output.with_suffix(".metadata.json")
        )
        _write_metadata(metadata_path, metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    predictions = pd.read_csv(args.predictions)
    fold_scores = score_window_predictions(
        predictions,
        press_mask_col=args.press_mask_col,
        null_label=args.null_label,
    )
    registry = validate_fold_registry(
        fold_scores,
        expected_subjects=(
            None
            if args.expected_subjects is None
            else _parse_csv(args.expected_subjects)
        ),
        expected_seeds=(
            None
            if args.expected_seeds is None
            else _parse_csv(args.expected_seeds, cast=int)
        ),
        expected_k=(
            None
            if args.expected_k is None
            else _parse_csv(args.expected_k, cast=int)
        ),
    )
    julia_summary, subject_means, population_summary = aggregate_fold_scores(
        fold_scores
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output / "katja_online_fold_scores.csv", index=False)
    julia_summary.to_csv(output / "katja_online_julia_mean_sd.csv", index=False)
    subject_means.to_csv(output / "katja_online_subject_seed_means.csv", index=False)
    population_summary.to_csv(
        output / "katja_online_population_mean_sem.csv", index=False
    )
    _write_metadata(
        output / "katja_online_reporting_metadata.json",
        {
            "format": "neureptrace_katja_online_reporting_v1",
            "fold_registry": registry,
            "julia_aggregation": "mean and sample SD over subject-by-seed folds",
            "neureptrace_aggregation": "mean across seeds within subject, then population mean and SEM",
            "label_function_status": "externally_supplied",
        },
    )
    print(julia_summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
